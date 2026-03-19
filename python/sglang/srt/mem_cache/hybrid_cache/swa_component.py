from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    IncLockRefResult,
    InsertParams,
    InsertResult,
)
from sglang.srt.mem_cache.hybrid_cache.tree_component import (
    ComponentName,
    TreeComponent,
    gen_component_uuid,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.hybrid_cache.hybrid_radix_cache import HybridTreeNode


class SWAComponent(TreeComponent):
    """Sliding window attention component.

    Each SWA node stores translated SWA pool indices as its component
    value, independent of the full attention indices on the same tree node.
    When SWA data is evicted from an internal node the node is tombstoned
    — its SWA component value becomes None while the full attention
    value stays intact.
    """

    @property
    def name(self) -> ComponentName:
        return ComponentName.SWA

    def _translate_full_to_swa(self, full_indices: torch.Tensor) -> torch.Tensor:
        return self.cache.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
            full_indices
        )

    def create_match_validator(self) -> Callable[[HybridTreeNode], bool]:
        sliding_window_size = self.cache.sliding_window_size
        name = self.name
        state = {"len": float("inf")}

        def validator(node: HybridTreeNode) -> bool:
            if node.component_value(name) is None:
                state["len"] = 0
                return False
            state["len"] += len(node.key)
            return state["len"] >= sliding_window_size

        return validator

    def update_component_on_insert_overlap(
        self,
        node: HybridTreeNode,
        prefix_len: int,
        total_prefix_len: int,
        value_slice: torch.Tensor,
        params: InsertParams,
    ) -> None:
        if params.prev_prefix_len >= total_prefix_len + prefix_len:
            return

        is_tombstone = node.component_value(self.name) is None
        if not is_tombstone:
            self.cache.token_to_kv_pool_allocator.free(value_slice)
            return

        swa_evicted_seqlen = params.swa_evicted_seqlen
        assert (
            node.component(self.name).lock_ref == 0
        ), f"tombstone {self.name} lock_ref should be 0, node {node.id}"
        assert (
            swa_evicted_seqlen % self.cache.page_size == 0
        ), f"{self.name}: swa_evicted_seqlen must be page-aligned, {swa_evicted_seqlen=}"

        if swa_evicted_seqlen <= total_prefix_len:
            # Branch 1: all swa tokens of value[:prefix_len] are not evicted, so we can insert it to the tree directly
            # Free full tokens in the original tree node
            self.cache.token_to_kv_pool_allocator.free(node.full_value)
            # Overwrite the new value in request to the tree node
            node.full_value = value_slice.clone()
            swa_value = self._translate_full_to_swa(node.full_value)
            node.set_component_value(self.name, swa_value)
            self.cache.lru_lists[self.name].insert_mru(node)
            self.cache.component_evictable_size_[self.name] += len(swa_value)
        elif swa_evicted_seqlen < total_prefix_len + prefix_len:
            # Branch 2: part of swa tokens of value[:prefix_len] are evicted, so we need to split the node and insert the value to new node
            start_idx = swa_evicted_seqlen - total_prefix_len
            self.cache.token_to_kv_pool_allocator.free(node.full_value[start_idx:])
            self.cache._split_node(node.key, node, start_idx)
            # Here node is the new node after split, so we can overwrite the value to the new node
            # The old node is still swa tombstone and the full token is not freed
            node.full_value = value_slice[start_idx:].clone()
            self.cache.token_to_kv_pool_allocator.free(value_slice[:start_idx])
            swa_value = self._translate_full_to_swa(node.full_value)
            node.set_component_value(self.name, swa_value)
            self.cache.lru_lists[self.name].insert_mru(node)
            self.cache.component_evictable_size_[self.name] += len(swa_value)
        else:
            # Branch 3: all swa tokens of value[:prefix_len] are evicted, so we don't need to update the node
            self.cache.token_to_kv_pool_allocator.free(value_slice)

    def get_tombstone_prefix_len_for_insert(
        self, total_prefix_len: int, new_key_len: int, params: InsertParams
    ) -> int:
        swa_evicted_seqlen = params.swa_evicted_seqlen
        if (
            swa_evicted_seqlen > total_prefix_len
            and swa_evicted_seqlen < total_prefix_len + new_key_len
        ):
            return swa_evicted_seqlen - total_prefix_len
        return 0

    def commit_insert_component_data(
        self,
        node: HybridTreeNode,
        is_new_leaf: bool,
        params: InsertParams,
        result: InsertResult,
    ) -> None:
        if is_new_leaf:
            swa_value = self._translate_full_to_swa(node.full_value)
            node.set_component_value(self.name, swa_value)
            self.cache.lru_lists[self.name].insert_mru(node)
            self.cache.component_evictable_size_[self.name] += len(swa_value)

    def redistribute_on_node_split(
        self, new_parent: HybridTreeNode, child: HybridTreeNode
    ):
        new_parent.component(self.name).lock_ref = child.component(self.name).lock_ref

        child_swa_value = child.component_value(self.name)
        if child_swa_value is not None:
            split_len = len(new_parent.key)
            new_parent.set_component_value(
                self.name, child_swa_value[:split_len].clone()
            )
            child.set_component_value(self.name, child_swa_value[split_len:].clone())
        else:
            new_parent.set_component_value(self.name, None)

        # parent inherits the swa_uuid from child for swa lock ref
        new_parent.component(self.name).metadata["uuid"] = child.component(
            self.name
        ).metadata.get("uuid")
        child.component(self.name).metadata.pop("uuid", None)

    def evict_component(self, node: HybridTreeNode, is_leaf: bool) -> int:
        swa_value = node.component_value(self.name)
        if swa_value is None:
            return 0
        # Direct swa_attn_allocator.free(swa_value) would double-free
        # free_swa(full_value) has the mapping guard to avoid double-free
        # TODO: decoupling full and swa free, need further discussion on mapping necessity
        self.cache.token_to_kv_pool_allocator.free_swa(node.full_value)
        freed = len(swa_value)
        self.cache.component_evictable_size_[self.name] -= freed
        if not is_leaf:
            node.set_component_value(self.name, None)
        return freed

    def drive_eviction(self, params: EvictParams, tracker: dict[str, int]) -> None:
        request = params.swa_num_tokens
        lru = self.cache.lru_lists[self.name]
        x = lru.get_lru_no_lock()
        while tracker[self.name] < request and x is not None and lru.in_list(x):
            assert x.component_value(self.name) is not None
            if len(x.children) > 0:
                # Internal: tombstone SWA, cascade to equal-priority components
                x_next = lru.get_prev_no_lock(x)
                self.cache._evict_component_and_detach_lru(
                    x, self, is_leaf=False, tracker=tracker
                )
                self.cache._cascade_evict(x, self, tracker)
                x = x_next
            else:
                # Leaf: evict SWA, cascade to all components
                self.cache._evict_component_and_detach_lru(
                    x, self, is_leaf=True, tracker=tracker
                )
                self.cache._cascade_evict(x, self, tracker)
                x = lru.get_lru_no_lock()

    def acquire_component_lock(
        self, node: HybridTreeNode, result: IncLockRefResult
    ) -> IncLockRefResult:
        sliding_window_size = self.cache.sliding_window_size
        swa_lock_size = 0
        swa_uuid_for_lock = None

        cur = node
        while cur != self.cache.root_node and swa_lock_size < sliding_window_size:
            assert (
                cur.component_value(self.name) is not None
            ), f"acquire_component_lock({self.name}) on tombstoned node {cur.id}"
            comp = cur.component(self.name)
            if comp.lock_ref == 0:
                self.cache.component_evictable_size_[self.name] -= len(cur.key)
                self.cache.component_protected_size_[self.name] += len(cur.key)
            comp.lock_ref += 1
            swa_lock_size += len(cur.key)
            if swa_lock_size >= sliding_window_size:
                if comp.metadata.get("uuid") is None:
                    comp.metadata["uuid"] = gen_component_uuid()
                swa_uuid_for_lock = comp.metadata["uuid"]
            cur = cur.parent

        result.swa_uuid_for_lock = swa_uuid_for_lock
        return result

    def release_component_lock(
        self, node: HybridTreeNode, params: Optional[DecLockRefParams]
    ) -> None:
        swa_uuid_for_lock = params.swa_uuid_for_lock if params else None
        dec_swa = True

        cur = node
        while cur != self.cache.root_node and dec_swa:
            assert (
                cur.component_value(self.name) is not None
            ), f"release_component_lock({self.name}) on tombstoned node {cur.id}"
            comp = cur.component(self.name)
            assert (
                comp.lock_ref > 0
            ), f"release_component_lock({self.name}) on node with lock_ref=0, node {cur.id}"
            if comp.lock_ref == 1:
                self.cache.component_evictable_size_[self.name] += len(cur.key)
                self.cache.component_protected_size_[self.name] -= len(cur.key)
            comp.lock_ref -= 1
            if swa_uuid_for_lock and comp.metadata.get("uuid") == swa_uuid_for_lock:
                dec_swa = False
            cur = cur.parent
