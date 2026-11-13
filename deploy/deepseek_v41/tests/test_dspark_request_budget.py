"""CPU regressions for DSPARK request budgets and overlap's finishing tick.

Real functions are AST-loaded from this tree; the only seams are
ForwardBatch.init_new (heavy graph-tensor constructor) and native model
execution. The verify executor chain (run_non_compact / _run_ragged /
_forward_prepared_verify / DFlashVerifyInput.prepare_for_verify /
DecodeCudaGraphRunner.can_run_graph / ModelRunner._forward_raw) and the
ragged layout factory run real code, so a parameter-plumbing bug or a
disable-flag the runner ignores fails here rather than in production.
"""

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
VERIFY = ROOT / "speculative/dspark_components/dspark_verify.py"
SCHEDULER = ROOT / "managers/scheduler.py"
RUNNER = ROOT / "model_executor/runner/decode_cuda_graph_runner.py"
MODELRUNNER = ROOT / "model_executor/model_runner.py"

from sglang.srt.speculative.dflash_info import (
    CaptureHiddenMode,
    DFlashVerifyInput,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.srt.speculative.dspark_components.dspark_verify import (
    apply_dflash_verify_logits_adjustments,
)
from sglang.srt.speculative import spec_utils
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


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


VERIFY_W = 6
CONTEXT_LEN = 32


class TestDsparkRequestBudget(CustomTestCase):
    def make_runner(self, trace):
        """Decode-graph runner with the REAL can_run_graph, stubbed execute."""

        def execute(forward_batch, **kw):
            trace.append("execute_graph")
            return NS(next_token_logits=torch.zeros(forward_batch.batch_size * VERIFY_W, 2), hidden_states=None)

        runner = NS(
            ragged_verify_mode=False,
            captured_req_width=VERIFY_W,
            require_mlp_tp_gather=False,
            require_mlp_sync=False,
            max_bs=32,
            disable_padding=False,
            is_encoder_decoder=False,
            enable_two_batch_overlap=False,
            enable_pdmux=False,
            model_runner=NS(spec_algorithm=NS(is_ngram=lambda: False)),
            backend=NS(),
            execute=execute,
            load_batch=lambda forward_batch: trace.append("load_graph"),
            init_forward_metadata=lambda forward_batch: trace.append(
                "init_eager_metadata"
            ),
        )
        can_run = actual(
            RUNNER,
            "can_run_graph",
            resolve_ragged_verify_layout=lambda forward_batch: None,
        )
        runner.can_run_graph = MethodType(can_run, runner)
        runner._make_graph_key = lambda cuda_graph_bs, **kw: ("key", cuda_graph_bs)
        runner._resolve_lora_variant = lambda forward_batch: None
        return runner

    def make_forward_batch(self, spec_info, trace):
        fb = NS(
            replace_embeds=None,
            mm_inputs=None,
            spec_info=spec_info,
            batch_size=2,
            input_ids=torch.zeros(2 * VERIFY_W),
            forward_mode=NS(
                is_cuda_graph=lambda: True,
                is_cpu_graph=lambda: True,
                is_target_verify=lambda: True,
                is_decode=lambda: False,
            ),
            can_run_decode_cuda_graph=True,
            can_run_tbo=True,
            encoder_lens=torch.ones(2),
            capture_hidden_mode=CaptureHiddenMode.FULL,
            positions=torch.zeros(2 * VERIFY_W, dtype=torch.int64),
        )
        return fb

    def worker_probe(self, rows, stop_at):
        captured = {}
        trace = []
        seqs = [p for p, _ in rows]
        reqs = [
            NS(
                origin_input_ids=[0] * 8,
                output_ids=[1] * (p - 7),
                sampling_params=NS(max_new_tokens=p - 7 + remaining),
            )
            for p, remaining in rows
        ]
        batch = NS(
            seq_lens=torch.tensor(seqs).as_subclass(SeqLens),
            seq_lens_cpu=list(seqs),
            seq_lens_sum=sum(seqs),
            req_pool_indices=torch.arange(len(rows)),
            spec_info=DraftInput(),
            forward_mode=NS(is_idle=lambda: False),
            reqs=reqs,
            sampling_info=None,
            global_num_tokens=None,
            has_grammar=False,
            return_logprob=False,
            can_run_decode_cuda_graph=True,
        )

        planner_env = {
            "VerifyWindow": NS,
            "RaggedVerifyLayout": RaggedVerifyLayout,
            "assign_extend_cache_locs_func": lambda **kw: torch.arange(
                100, 100 + kw["batch_size"] * kw["draft_token_num"]
            ),
        }
        alloc = actual(PLANNER, "alloc_verify_window", **planner_env)
        worker_env = {
            "DFlashDraftInputV2": DraftInput,
            "alloc_verify_window": alloc,
            "RaggedVerifyLayout": RaggedVerifyLayout,
            "InfoSegment": NS(DRAFT="draft", TARGET_VERIFY="target"),
            "get_parallel": lambda: NS(enable_dp_attention=False),
            "prepare_mamba_track_for_verify": lambda batch: None,
            "verify_logits_adjustments_are_noop": lambda info: True,
            "make_next_draft_input": lambda **kw: NS(),
            "GenerationBatchResult": lambda **kw: NS(**kw),
        }
        names = {n.name for n in ast.walk(ast.parse(PLANNER.read_text())) if isinstance(n, ast.FunctionDef)}
        if "row_verify_budgets" in names:
            worker_env["row_verify_budgets"] = actual(PLANNER, "row_verify_budgets")
        for name in ("request_verify_lens", "bound_request_verify_layout"):
            if name in names:
                worker_env[name] = actual(PLANNER, name, **planner_env)
        method = actual(WORKER, "_forward_decode", **worker_env)

        def propose(**kw):
            captured["window"] = kw["verify_window"]
            if stop_at == "draft":
                raise StopProbe()
            return NS(
                draft_block_ids=torch.ones(len(rows), VERIFY_W, dtype=torch.int64),
                draft_block=NS(draft_tokens=torch.ones(len(rows), VERIFY_W - 1, dtype=torch.int64)),
                confidence=torch.zeros(len(rows)),
                folded=False,
            )

        run_non_compact = actual(
            VERIFY,
            "run_non_compact",
            DFlashVerifyInput=DFlashVerifyInput,
            CaptureHiddenMode=CaptureHiddenMode,
            TargetVerifyResult=NS,
            apply_dflash_verify_logits_adjustments=apply_dflash_verify_logits_adjustments,
        )
        prepared = actual(
            VERIFY,
            "_forward_prepared_verify",
            TargetVerifyResult=NS,
            _is_npu=False,
        )
        runner = self.make_runner(trace)

        def load_batch(fb):
            trace.append("load_graph")

        def init_metadata(fb):
            trace.append("init_eager_metadata")

        target_worker = NS(
            model_runner=NS(
                model=object(),
                decode_cuda_graph_runner=runner,
                attn_backend=NS(init_forward_metadata=init_metadata),
            ),
            forward_batch_generation=lambda **kw: NS(
                logits_output=NS(next_token_logits=torch.zeros(sum(b.get("w", VERIFY_W) for b in [{}]) * 0 or 4, 2)),
                can_run_cuda_graph=False,
            ),
        )
        executor = NS(
            verify_num_draft_tokens=VERIFY_W,
            _verify_backend_self_adds_seq_lens=lambda: True,
            target_worker=target_worker,
            accept_and_finalize=lambda **kw: (_ for _ in ()).throw(StopProbe()),
            commit_hidden=lambda **kw: None,
            verify_epilogue=None,
        )

        def run_compact_probe(**kw):
            captured["compact"] = True
            captured["layout"] = kw["layout"]
            captured["allow"] = kw.get("allow_cuda_graph", True)
            raise StopProbe()

        executor.run_compact = run_compact_probe
        executor.run_non_compact = MethodType(run_non_compact, executor)
        executor._forward_prepared_verify = MethodType(prepared, executor)

        def init_new_seam(batch, model_runner, capture_hidden_mode=None, **kw):
            trace.append("init_new")
            spec_info = batch.spec_info
            return self.make_forward_batch(spec_info, trace)

        owner = NS(
            device="cpu",
            verify_num_draft_tokens=VERIFY_W,
            gamma=VERIFY_W - 1,
            model_runner=NS(
                model_config=NS(context_len=CONTEXT_LEN),
                req_to_token_pool=NS(req_to_token=torch.zeros(len(rows), 48, dtype=torch.int32)),
                ngram_embedding_manager=NS(update_after_verify=lambda **kw: None),
            ),
            target_worker=target_worker,
            _block_pos_offsets=torch.arange(VERIFY_W).view(1, -1),
            _observers=NS(
                begin_step=lambda: None,
                segment=lambda name: nullcontext(),
                observe_verify_step=lambda **kw: None,
                note_idle_decode_step=lambda: None,
            ),
            _draft_context=lambda: nullcontext(),
            _proposer=NS(propose=propose),
            _draft_is_moe=False,
            _dp_verify_tier_num_tokens=lambda batch: None,
            _verify_epilogue=None,
            _verify_executor=executor,
            _simulate_acc_len=0,
            _verify_planner=NS(
                resolve_verify_token_budget=lambda **kw: None,
                schedule_layout=lambda **kw: None,
                should_run_compact=lambda **kw: False,
                mode_value="static",
            ),
            _commit_target_mamba_states_after_verify=lambda **kw: None,
        )
        with patch.object(torch, "get_device_module", return_value=NS(current_stream=lambda: None)):
            with patch.object(spec_utils, "prepare_mamba_track_for_verify", lambda batch: None):
                with patch.object(ForwardBatch, "init_new", staticmethod(init_new_seam)):
                    with self.assertRaises(StopProbe):
                        method(owner, batch, None)
        self.assertTrue(batch.can_run_decode_cuda_graph, "boundary must not latch future batches to eager")
        captured["trace"] = trace
        captured["seam_allows"] = [
            getattr(batch, "spec_info", None) for batch in [batch]
        ]
        return captured

    # --- draft window bounding (RED on 04d, stays green) ---

    def test_real_window_is_bounded_before_draft(self):
        # Legal positions keep their original per-column values (04d behavior);
        # only positions that actually cross the RoPE boundary clamp to the
        # anchor position, and cache locs always keep the assigned per-column
        # slots (no same-slot aliasing).
        for p, remaining in ((12, 1), (12, 3), (12, 6), (28, 1)):
            with self.subTest(p=p, remaining=remaining):
                result = self.worker_probe([(p, remaining)], "draft")
                window = result["window"]
                self.assertTrue(bool((window.positions_2d < CONTEXT_LEN).all()), window.positions_2d)
                expected_positions = [p + k if p + k < CONTEXT_LEN else p for k in range(VERIFY_W)]
                self.assertEqual(window.positions_2d[0].tolist(), expected_positions)
                # Every column keeps its own assigned slot, budget or not.
                self.assertEqual(
                    window.verify_cache_loc_2d[0].tolist(),
                    [100 + k for k in range(VERIFY_W)],
                )

    # --- static full-width fast path must reach the real executor ---

    def test_static_full_width_reaches_real_executor_and_keeps_graph(self):
        result = self.worker_probe([(12, 6)], "target")
        self.assertNotIn("compact", result)
        trace = result["trace"]
        self.assertIn("init_new", trace)
        self.assertIn("load_graph", trace, f"full-width fast path lost graph preload: {trace}")
        self.assertNotIn("init_eager_metadata", trace)

    # --- mixed budgets go through the REAL layout factory ---

    def test_mixed_budget_layout_from_real_factory(self):
        result = self.worker_probe([(28, 1), (12, 6)], "target")
        self.assertTrue(result["compact"])
        layout = result["layout"]
        self.assertEqual(layout.verify_lens_cpu, [1, 6])
        self.assertEqual(layout.graph_num_tokens, 2 * VERIFY_W)
        self.assertFalse(result["allow"])

    def test_single_capped_budget_layout_from_real_factory(self):
        result = self.worker_probe([(28, 1)], "target")
        self.assertTrue(result["compact"])
        layout = result["layout"]
        self.assertEqual(layout.verify_lens_cpu, [1])
        self.assertEqual(layout.graph_num_tokens, VERIFY_W)
        self.assertFalse(result["allow"])

    # --- ragged budget forward must thread allow/disallow through every layer ---

    def _run_ragged_probe(self, allow):
        trace = []
        seam_inputs = []

        def init_new_seam(batch, model_runner, capture_hidden_mode=None, **kw):
            trace.append("init_new")
            return self.make_forward_batch(batch.spec_info, trace)

        runner = self.make_runner(trace)

        def load_batch(fb):
            trace.append("load_graph")

        def init_metadata(fb):
            trace.append("init_eager_metadata")

        target_worker = NS(
            model_runner=NS(
                model=object(),
                decode_cuda_graph_runner=runner,
                attn_backend=NS(init_forward_metadata=init_metadata),
            ),
            forward_batch_generation=lambda **kw: NS(
                logits_output=NS(next_token_logits=torch.zeros(2, 2)),
                can_run_cuda_graph=False,
            ),
        )
        executor = NS(
            verify_num_draft_tokens=VERIFY_W,
            target_worker=target_worker,
        )
        prepared = actual(VERIFY, "_forward_prepared_verify", TargetVerifyResult=NS, _is_npu=False)
        executor._forward_prepared_verify = MethodType(prepared, executor)
        run_ragged = actual(
            VERIFY,
            "_run_ragged",
            DFlashVerifyInput=DFlashVerifyInput,
            CaptureHiddenMode=CaptureHiddenMode,
            _forward_prepared_verify=executor._forward_prepared_verify,
        )
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[1, 6], device=torch.device("cpu"), grid=[2 * VERIFY_W]
        )
        ragged_window = NS(
            verify_ids=torch.ones(7, dtype=torch.int64),
            positions=torch.arange(7),
            verify_cache_loc=torch.arange(7),
        )
        batch = NS(
            seq_lens_cpu=torch.tensor([28, 12]),
            seq_lens_sum=40,
            out_cache_loc=None,
            spec_info=None,
            forward_mode=NS(is_idle=lambda: False),
        )
        with patch.object(spec_utils, "prepare_mamba_track_for_verify", lambda batch: None):
          with patch.object(ForwardBatch, "init_new", staticmethod(init_new_seam)):
            result = run_ragged(
                executor,
                batch=batch,
                layout=layout,
                ragged_window=ragged_window,
                sampling_info=None,
                allow_cuda_graph=allow,
            )
        return trace, result

    def test_ragged_budget_forward_threads_disallow_to_graph_gate(self):
        trace, result = self._run_ragged_probe(allow=False)
        self.assertIn("init_new", trace, f"real ragged forward did not reach prepare: {trace}")
        self.assertIn("init_eager_metadata", trace, f"allow=False still preloaded graph: {trace}")
        self.assertNotIn("load_graph", trace)
        self.assertFalse(getattr(result, "can_run_cuda_graph", True))

    def test_ragged_full_width_forward_keeps_graph_preload(self):
        trace, result = self._run_ragged_probe(allow=True)
        self.assertIn("init_new", trace)
        self.assertIn("load_graph", trace, f"allow=True lost graph preload: {trace}")
        self.assertNotIn("init_eager_metadata", trace)

    # --- the graph runner itself must honor the spec's disallow flag ---

    def test_graph_runner_respects_verify_disallow_flag(self):
        trace = []
        runner = self.make_runner(trace)
        fb = self.make_forward_batch(None, trace)
        self.assertTrue(runner.can_run_graph(fb), "fixture must reach True without the flag")
        capped_spec = DFlashVerifyInput(
            draft_token=torch.zeros(VERIFY_W, dtype=torch.int64),
            positions=torch.arange(VERIFY_W),
            draft_token_num=VERIFY_W,
        )
        capped_spec.verify_disallow_cuda_graph = True
        fb_capped = self.make_forward_batch(capped_spec, trace)
        self.assertFalse(runner.can_run_graph(fb_capped))

    def test_ragged_mode_cannot_bypass_verify_disallow_flag(self):
        # Compact-mode support face: the ragged early-return in can_run_graph
        # must not answer before the disallow flag is checked.
        from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout

        can_run = actual(
            RUNNER,
            "can_run_graph",
            resolve_ragged_verify_layout=lambda forward_batch: getattr(
                forward_batch.spec_info, "ragged_verify_layout", None
            ),
        )
        can_ragged = actual(RUNNER, "_can_run_ragged_verify_graph")

        def make_ragged_runner():
            return NS(
                ragged_verify_mode=True,
                captured_req_width=VERIFY_W,
                require_mlp_tp_gather=False,
                require_mlp_sync=False,
                max_bs=32,
                disable_padding=False,
                is_encoder_decoder=False,
                enable_two_batch_overlap=False,
                enable_pdmux=False,
                model_runner=NS(spec_algorithm=NS(is_ngram=lambda: False)),
                attn_backend=NS(supports_ragged_verify_graph=True),
                capture_num_tokens=[VERIFY_W, 2 * VERIFY_W],
                capture_hidden_mode=CaptureHiddenMode.FULL,
                backend=NS(),
                _make_graph_key=lambda bs, **kw: ("key", bs),
                _resolve_lora_variant=lambda fb: None,
                _ragged_capture_slots=lambda admission_tokens: 2,
            )

        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[1, 6],
            device=torch.device("cpu"),
            grid=[2 * VERIFY_W],
        )
        plain_spec = NS(
            verify_disallow_cuda_graph=False,
            ragged_verify_layout=layout,
            num_tokens_per_req=VERIFY_W,
        )
        disallowed_spec = NS(
            verify_disallow_cuda_graph=True,
            ragged_verify_layout=layout,
            num_tokens_per_req=VERIFY_W,
        )
        runner = make_ragged_runner()
        runner.can_run_graph = MethodType(can_run, runner)
        runner._can_run_ragged_verify_graph = MethodType(can_ragged, runner)

        def fb_for(spec_info):
            return self.make_forward_batch(spec_info, [])

        # Without the flag the ragged path admits the layout (fixture control).
        self.assertTrue(runner.can_run_graph(fb_for(plain_spec)))
        # With the flag the ragged early-return must not answer around it.
        self.assertFalse(runner.can_run_graph(fb_for(disallowed_spec)))

    # --- ModelRunner._forward_raw must not re-enable a disabled graph ---

    def test_forward_raw_respects_verify_disallow_flag(self):
        trace = []
        runner = self.make_runner(trace)

        def prepare_eager(forward_batch):
            trace.append("prepare_eager")
            raise StopProbe()

        forward_raw = actual(
            MODELRUNNER,
            "_forward_raw",
            has_forward_context=lambda: False,
            forward_context=lambda ctx: nullcontext(),
            ForwardContext=NS,
            ModelRunnerOutput=NS,
            contextlib=NS(nullcontext=nullcontext),
        )
        owner = NS(
            device="cpu",
            attn_backend=NS(),
            decode_cuda_graph_runner=runner,
            hisparse_coordinator=None,
            _prepare_eager_forward_batch=prepare_eager,
        )
        capped_spec = DFlashVerifyInput(
            draft_token=torch.zeros(VERIFY_W, dtype=torch.int64),
            positions=torch.arange(VERIFY_W),
            draft_token_num=VERIFY_W,
        )
        capped_spec.verify_disallow_cuda_graph = True
        fb = self.make_forward_batch(capped_spec, trace)
        with self.assertRaises(StopProbe):
            forward_raw(owner, fb, None)
        self.assertEqual(trace, ["prepare_eager"], f"graph replayed despite disallow flag: {trace}")

    def test_real_verify_prepare_carries_mm_window_metadata(self):
        # Real ScheduleBatch -> ForwardBatch construction and MM embed/count
        # boundary; attention, weights, KV translation, Engram and distributed
        # state are CPU stand-ins. Unlike the old seam, init_new is not replaced.
        from sglang.srt.managers.mm_utils import embed_mm_inputs
        from sglang.srt.managers.schedule_batch import (
            Modality,
            MultimodalDataItem,
            MultimodalInputs,
            ScheduleBatch,
        )
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.speculative.dspark_components.dspark_verify import (
            TargetVerifyExecutor,
        )

        prepare_mm = actual(
            ROOT / "models/deepseek_v4.py",
            "_prepare_mm_embeddings",
            embed_mm_inputs=embed_mm_inputs,
            MultimodalInputs=MultimodalInputs,
        )
        for widths, ragged in (([1], True), ([3], True), ([6], True), ([1, 6], True), ([6], False)):
            with self.subTest(widths=widths, ragged=ragged):
                prefixes = torch.tensor([12 + 8 * i for i in range(len(widths))])
                image = MultimodalDataItem(
                    modality=Modality.IMAGE,
                    pad_value=200,
                    offsets=[(2, 4)],
                    feature=torch.ones(3, 2),
                )
                mm = MultimodalInputs(mm_items=[image])
                batch = ScheduleBatch(
                    reqs=[NS(rid=str(i), lora_id=None, token_type_ids=None) for i in range(len(widths))],
                    device="cpu",
                    forward_mode=ForwardMode.DECODE,
                    input_ids=torch.zeros(len(widths), dtype=torch.int64),
                    req_pool_indices=torch.arange(len(widths)),
                    seq_lens=prefixes.clone(),
                    seq_lens_cpu=prefixes.clone(),
                    seq_lens_sum=int(prefixes.sum()),
                    orig_seq_lens=prefixes.clone(),
                    multimodal_inputs=[mm] + [None] * (len(widths) - 1),
                )
                embed_tokens = torch.nn.Embedding(256, 2)
                model = NS(
                    get_input_embeddings=lambda: embed_tokens,
                    get_image_feature=lambda items: self.fail("cached prompt image was re-embedded"),
                )
                runner = NS(
                    device="cpu",
                    ps=NS(attn_dcp_size=1),
                    kv_index_translator=NS(rebind_write_loc=lambda fb: None),
                    ngram_embedding_manager=NS(enabled=False),
                    lora_manager=None,
                    model_config=NS(model_is_mrope=False),
                    decode_cuda_graph_runner=None,
                    attn_backend=NS(
                        init_forward_metadata=lambda fb: None,
                        make_forward_metadata_from_raw_verify=lambda *args: None,
                    ),
                )
                seen = []

                def forward(**kw):
                    fb = kw["forward_batch"]
                    seen.append(fb)
                    embeds = prepare_mm(model, fb.input_ids, fb)
                    self.assertTrue(torch.equal(embeds, embed_tokens(fb.input_ids)))
                    return NS(logits_output=NS(next_token_logits=torch.zeros(sum(widths), 2)), can_run_cuda_graph=False)

                target = NS(model_runner=runner, forward_batch_generation=forward)
                executor = TargetVerifyExecutor(
                    target_worker=target,
                    gamma=VERIFY_W - 1,
                    verify_num_draft_tokens=VERIFY_W,
                    model_runner=runner,
                    kv_injector=None,
                    tp_sync=None,
                )
                layout = RaggedVerifyLayout.from_verify_lens(
                    verify_lens_cpu=widths,
                    device=torch.device("cpu"),
                    grid=[len(widths) * VERIFY_W],
                )
                window = NS(
                    verify_ids=torch.zeros(sum(widths), dtype=torch.int64),
                    positions=torch.cat([torch.arange(p, p + w) for p, w in zip(prefixes.tolist(), widths)]),
                    verify_cache_loc=torch.arange(sum(widths)),
                )
                with (
                    patch.object(spec_utils, "mamba_extra_buffer_enabled", return_value=False),
                    patch("sglang.srt.model_executor.forward_batch_info.get_parallel", return_value=NS(moe_ep_size=8)),
                ):
                    if ragged:
                        executor._run_ragged(
                            batch=batch,
                            layout=layout,
                            ragged_window=window,
                            sampling_info=None,
                            allow_cuda_graph=False,
                        )
                    else:
                        executor.run_non_compact(
                            batch=batch,
                            draft_input=NS(),
                            verify_ids_2d=window.verify_ids.view(len(widths), VERIFY_W),
                            verify_window=NS(
                                positions_2d=window.positions.view(len(widths), VERIFY_W),
                                verify_cache_loc=window.verify_cache_loc,
                            ),
                            sampling_info=None,
                            allow_cuda_graph=False,
                        )
                fb = seen[0]
                self.assertEqual(fb.extend_prefix_lens_cpu, prefixes.tolist())
                self.assertEqual(fb.extend_seq_lens_cpu, widths)
                self.assertIs(fb.mm_inputs[0], mm)
                self.assertTrue(torch.equal(batch.seq_lens_cpu, prefixes))

    # --- accept cap contract for the budget layout (kept from the first suite) ---

    def test_target_uses_same_budget_and_no_bucket_growth(self):
        path = ROOT.parent / "kernels/ops/speculative/dspark/dspark_accept.py"
        cap = actual(path, "cap_correct_len")
        finalize = actual(path, "finalize_accept_lens", FinalizeAcceptLensResult=NS)
        cl, trim = cap(correct_len=torch.tensor([5]), verify_lens=torch.tensor([1], dtype=torch.int32))
        output = finalize(correct_len=cl, cap_trim_lens=trim, prefix_lens=torch.tensor([28]))
        self.assertEqual(output.commit_lens.tolist(), [1])
        self.assertEqual(output.new_seq_lens.tolist(), [29])

    # --- simulated accept override must obey the budget cutoff too ---

    def test_simulated_accept_respects_budget_cutoff(self):
        from sglang.kernels.ops.speculative.dspark.dspark_accept import (
            AcceptGreedy,
            CapCorrectLen,
            FinalizeAcceptLens,
        )
        from sglang.kernels.ops.speculative.dspark.dspark_verify_window import (
            BuildOutTokens,
        )
        from sglang.srt.speculative.spec_utils import sample_simulated_acc_len

        accept_draft = actual(
            VERIFY, "accept_draft_tokens", AcceptGreedy=AcceptGreedy
        )
        simulated = actual(
            VERIFY,
            "_simulated_correct_len",
            sample_simulated_acc_len=sample_simulated_acc_len,
            SIMULATE_ACC_METHOD="match-expected",
        )
        accept = actual(
            VERIFY,
            "accept_and_finalize",
            accept_draft_tokens=accept_draft,
            FinalizeAcceptLens=FinalizeAcceptLens,
            BuildOutTokens=BuildOutTokens,
            CapCorrectLen=CapCorrectLen,
            AcceptOuts=NS,
            SpecTpSyncSite=NS(
                DSPARK_ACCEPT_GREEDY="greedy", DSPARK_ACCEPT_SAMPLE="sample"
            ),
        )
        owner = NS(
            _simulate_acc_len=6.0,
            _simulated_correct_drafts_buf=None,
            gamma=VERIFY_W - 1,
            verify_num_draft_tokens=VERIFY_W,
            _tp_sync=NS(sync=lambda site, tensor: tensor),
        )
        owner._simulated_correct_len = MethodType(simulated, owner)
        out = accept(
            owner,
            folded_accept=False,
            bs=1,
            verify_ids_2d=torch.zeros(1, VERIFY_W, dtype=torch.int64),
            target_logits=torch.tensor([[1.0, 0.0]] * VERIFY_W),
            draft_block=NS(greedy_mask=torch.ones(1, dtype=torch.bool)),
            sampling_info=None,
            draft_input=NS(),
            layout=NS(verify_lens=torch.tensor([1])),
            prefix_lens=torch.tensor([28]),
            draft_tokens=torch.zeros(1, VERIFY_W - 1, dtype=torch.int64),
        )
        # The simulated 6-accept must be capped to the 1-token budget row:
        # commit stays at the budget instead of jumping past the window.
        self.assertLessEqual(out.commit_lens.tolist(), [1])
        self.assertEqual(out.new_seq_lens.tolist(), [29])

    # --- overlap loop drains and filters zero-budget rows (kept, unchanged) ---

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

                    def grammar_needs_sync(self):
                        return False

                    def copy(self):
                        return self

                    def is_empty(self):
                        return not self.reqs

                    def filter_batch(self):
                        self.reqs = [r for r in self.reqs if not r.done]

                old, current = Batch(True), Batch(False)
                calls = 0
                owner = NS(
                    gracefully_exit=False,
                    last_batch=None,
                    running_batch=old,
                    require_mlp_sync=False,
                    _engine_paused=False,
                    enable_unified_memory=False,
                    is_generation=False,
                    request_receiver=NS(recv_requests=lambda: []),
                    process_input_requests=lambda reqs: None,
                    _apply_war_barrier=lambda: None,
                )

                def next_batch(**kw):
                    nonlocal calls
                    calls += 1
                    if calls == 2:
                        owner.gracefully_exit = True
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

                env = {
                    "deque": deque,
                    "get_spec": lambda: NS(speculative_num_draft_tokens=VERIFY_W),
                    "envs": NS(
                        SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP=NS(get=lambda: False),
                        SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY=NS(get=lambda: False),
                    ),
                }
                owner.is_disable_overlap_for_batch = MethodType(
                    actual(SCHEDULER, "is_disable_overlap_for_batch", **env), owner
                )
                names = {n.name for n in ast.walk(ast.parse(SCHEDULER.read_text())) if isinstance(n, ast.FunctionDef)}
                if "_needs_dspark_budget_sync" in names:
                    owner._needs_dspark_budget_sync = MethodType(
                        actual(SCHEDULER, "_needs_dspark_budget_sync", **env), owner
                    )
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
