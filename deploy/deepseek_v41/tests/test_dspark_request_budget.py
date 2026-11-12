"""CPU regressions for DSPARK request budgets and overlap's finishing tick."""

import ast
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from types import MethodType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
from sglang.test.test_utils import CustomTestCase

ROOT = Path(__file__).resolve().parents[1] / "runtime-overlay/sglang/srt"
PLANNER = ROOT / "speculative/dspark_components/dspark_planner.py"
WORKER = ROOT / "speculative/dspark_components/dspark_worker_v2.py"
SCHEDULER = ROOT / "managers/scheduler.py"


def actual(path, name, **extra):
    tree = ast.parse(path.read_text())
    nodes = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    node = nodes[name]
    node.decorator_list = []
    env = {"torch": torch, **extra}
    import __future__
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec", flags=__future__.annotations.compiler_flag), env)
    return env[name]


class StopProbe(Exception):
    pass


class SeqLens(torch.Tensor):
    def record_stream(self, stream):
        pass  # CPU substitute for the stream fence only.


class DraftInput:
    pass


class TestDsparkRequestBudget(CustomTestCase):
    def worker_probe(self, remaining, stop_at):
        captured = {}
        p = 28 if remaining == 1 else 12
        req = NS(origin_input_ids=[0] * 8, output_ids=[1] * (p - 7), sampling_params=NS(max_new_tokens=p - 7 + remaining))
        batch = NS(seq_lens=torch.tensor([p]).as_subclass(SeqLens), seq_lens_cpu=[p], req_pool_indices=torch.tensor([0]), spec_info=DraftInput(), forward_mode=NS(is_idle=lambda: False), reqs=[req], sampling_info=None, global_num_tokens=None, has_grammar=False, can_run_decode_cuda_graph=True)
        layout_factory = NS(from_verify_lens=lambda **kw: NS(verify_lens_cpu=list(kw["verify_lens_cpu"]), verify_lens=torch.tensor(kw["verify_lens_cpu"]), graph_num_tokens=max(sum(kw["verify_lens_cpu"]), min(kw["grid"]))))
        planner_env = {"VerifyWindow": NS, "RaggedVerifyLayout": layout_factory, "assign_extend_cache_locs_func": lambda **kw: torch.arange(100, 100 + kw["batch_size"] * kw["draft_token_num"])}
        alloc = actual(PLANNER, "alloc_verify_window", **planner_env)
        worker_env = {"DFlashDraftInputV2": DraftInput, "alloc_verify_window": alloc, "RaggedVerifyLayout": layout_factory, "InfoSegment": NS(DRAFT="draft", TARGET_VERIFY="target"), "get_parallel": lambda: NS(enable_dp_attention=False), "prepare_mamba_track_for_verify": lambda batch: None, "verify_logits_adjustments_are_noop": lambda info: True}
        names = {n.name for n in ast.walk(ast.parse(PLANNER.read_text())) if isinstance(n, ast.FunctionDef)}
        if "row_verify_budgets" in names:
            # Absent on the unmodified baseline; the full-width fast path then
            # runs through the original alloc and worker without the budget.
            worker_env["row_verify_budgets"] = actual(PLANNER, "row_verify_budgets")
        for name in ("request_verify_lens", "bound_request_verify_layout"):
            if name in names:
                worker_env[name] = actual(PLANNER, name, **planner_env)
        method = actual(WORKER, "_forward_decode", **worker_env)
        def propose(**kw):
            captured["window"] = kw["verify_window"]
            captured["draft_graph"] = kw.get("allow_cuda_graph", True)
            if stop_at == "draft":
                raise StopProbe()
            return NS(draft_block_ids=torch.ones(1, 6, dtype=torch.int64), draft_block=NS(draft_tokens=torch.ones(1, 5, dtype=torch.int64)), confidence=torch.zeros(1), folded=False)
        def target(compact, **kw):
            captured["compact"] = compact
            captured["layout"] = kw.get("layout")
            captured["target_graph"] = kw.get("allow_cuda_graph", True)
            raise StopProbe()
        owner = NS(device="cpu", verify_num_draft_tokens=6, gamma=5, model_runner=NS(model_config=NS(context_len=32), req_to_token_pool=NS(req_to_token=torch.zeros(1, 48, dtype=torch.int32))), target_worker=NS(model_runner=NS(model=object())), _block_pos_offsets=torch.arange(6).view(1, -1), _observers=NS(begin_step=lambda: None, segment=lambda name: nullcontext()), _draft_context=lambda: nullcontext(), _proposer=NS(propose=propose), _draft_is_moe=False, _dp_verify_tier_num_tokens=lambda batch: None, _verify_epilogue=None, _simulate_acc_len=0, _verify_planner=NS(resolve_verify_token_budget=lambda **kw: None, schedule_layout=lambda **kw: None, should_run_compact=lambda **kw: False, mode_value="static"), _verify_executor=NS(verify_epilogue=None, run_compact=lambda **kw: target(True, **kw), run_non_compact=lambda **kw: target(False, **kw)))
        with patch.object(torch, "get_device_module", return_value=NS(current_stream=lambda: None)):
            with self.assertRaises(StopProbe):
                method(owner, batch, None)
        self.assertTrue(batch.can_run_decode_cuda_graph, "boundary must not latch future batches to eager")
        return captured

    def test_real_window_is_bounded_before_draft(self):
        for remaining in (1, 3, 6):
            with self.subTest(remaining=remaining):
                result = self.worker_probe(remaining, "draft")
                window = result["window"]
                self.assertTrue(bool((window.positions_2d < 32).all()), window.positions_2d)
                if remaining < 6:
                    p = window.positions_2d[0, 0].item()
                    self.assertEqual(window.positions_2d[0, remaining:].tolist(), [p] * (6 - remaining))
                    anchor_loc = window.verify_cache_loc_2d[0, 0].item()
                    self.assertEqual(window.verify_cache_loc_2d[0, remaining:].tolist(), [anchor_loc] * (6 - remaining))
                    # Draft graph shape is unchanged (only contents clamped), so
                    # the draft cuda graph stays eligible for capped rows.
                    self.assertTrue(result["draft_graph"])
                else:
                    self.assertEqual(window.verify_cache_loc_2d.tolist(), [[100, 101, 102, 103, 104, 105]])
                    self.assertTrue(result["draft_graph"])

    def test_target_uses_same_budget_and_no_bucket_growth(self):
        result = self.worker_probe(1, "target")
        self.assertTrue(result["compact"])
        self.assertEqual(result["layout"].verify_lens_cpu, [1])
        self.assertEqual(result["layout"].graph_num_tokens, 6)
        self.assertFalse(result["target_graph"])
        # The real accept cap + finalize contract for this identical target layout.
        path = ROOT.parent / "kernels/ops/speculative/dspark/dspark_accept.py"
        cap = actual(path, "cap_correct_len")
        finalize = actual(path, "finalize_accept_lens", FinalizeAcceptLensResult=NS)
        cl, trim = cap(correct_len=torch.tensor([5]), verify_lens=result["layout"].verify_lens)
        output = finalize(correct_len=cl, cap_trim_lens=trim, prefix_lens=torch.tensor([28]))
        self.assertEqual(output.commit_lens.tolist(), [1])
        self.assertEqual(output.new_seq_lens.tolist(), [29])

    def test_overlap_drains_then_filters_zero_rows_but_keeps_fast_path(self):
        for limits, expected in (([1], []), ([1, 3], [1]), ([50, 50], [0, 1])):
            with self.subTest(limits=limits):
                reqs = [NS(rid=i, output_ids=[], sampling_params=NS(max_new_tokens=n), done=False) for i, n in enumerate(limits)]
                trace = []
                class Batch:
                    def __init__(self, extend):
                        self.reqs = reqs[:]
                        self.forward_mode = NS(is_extend=lambda: extend, is_decode=lambda: not extend)
                        self.spec_algorithm = NS(is_none=lambda: False, is_dspark=lambda: True)
                    def grammar_needs_sync(self): return False
                    def copy(self): return self
                    def is_empty(self): return not self.reqs
                    def filter_batch(self): self.reqs = [r for r in self.reqs if not r.done]
                old, current = Batch(True), Batch(False)
                calls = 0
                owner = NS(gracefully_exit=False, last_batch=None, running_batch=old, require_mlp_sync=False, _engine_paused=False, enable_unified_memory=False, is_generation=False, request_receiver=NS(recv_requests=lambda: []), process_input_requests=lambda reqs: None, _apply_war_barrier=lambda: None)
                def next_batch(**kw):
                    nonlocal calls
                    calls += 1
                    if calls == 2: owner.gracefully_exit = True
                    b = old if calls == 1 else current
                    return NS(running_batch=b, batch_to_run=b)
                def forward(b):
                    trace.append(("forward", [reqs.index(r) for r in b.reqs]))
                    return object()
                def process(b, result):
                    trace.append(("process", None))
                    for r in b.reqs:
                        r.output_ids.append(1)
                        r.done = len(r.output_ids) >= r.sampling_params.max_new_tokens
                env = {"deque": deque, "get_spec": lambda: NS(speculative_num_draft_tokens=6), "envs": NS(SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP=NS(get=lambda: False), SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY=NS(get=lambda: False))}
                owner.is_disable_overlap_for_batch = MethodType(actual(SCHEDULER, "is_disable_overlap_for_batch", **env), owner)
                names = {n.name for n in ast.walk(ast.parse(SCHEDULER.read_text())) if isinstance(n, ast.FunctionDef)}
                if "_needs_dspark_budget_sync" in names:
                    owner._needs_dspark_budget_sync = MethodType(actual(SCHEDULER, "_needs_dspark_budget_sync", **env), owner)
                owner.get_next_batch_to_run, owner.run_batch, owner.process_batch_result = next_batch, forward, process
                actual(SCHEDULER, "event_loop_overlap", **env)(owner)
                forwards = [ids for kind, ids in trace if kind == "forward"]
                if expected:
                    self.assertEqual(forwards[-1], expected)
                else:
                    self.assertEqual(len(forwards), 1)
                if min(limits) <= 12:
                    self.assertEqual(trace[1][0], "process")
                else:
                    self.assertEqual(trace[1][0], "forward")


if __name__ == "__main__":
    unittest.main(verbosity=2)
