"""Exercise the real SM120 prefill adapter's token-slot/count contract on CPU."""

import ast
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from sglang.test.test_utils import CustomTestCase

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "profiles/sm120/kernels/ops/attention/flash_mla_sm120.py"


class TestSm120ExtraTokenSlots(CustomTestCase):
    def test_reblocking_preserves_extra_slots_and_candidate_lengths(self):
        # Execute the actual adapter; replace only GPU page-copy/native calls.
        # The incorrect index expansion remains executable real torch code.
        function = next(
            node
            for node in ast.parse(PROFILE.read_text()).body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_flash_mla_sm120_prefill"
        )
        q = torch.zeros(65, 1, 8, 512, dtype=torch.bfloat16)
        main = torch.zeros(2, 128, 1, 584, dtype=torch.uint8)
        main_indices = torch.tensor([[[0, 63, -1]]] * 65, dtype=torch.int32)
        main_lengths = torch.full((65,), 2, dtype=torch.int32)
        for page_size in (128, 256):
            for with_lengths in (False, True):
                with self.subTest(page_size=page_size, with_lengths=with_lengths):
                    extra = torch.zeros(16, page_size, 1, 584, dtype=torch.uint8)
                    indices = torch.tensor(
                        [
                            [
                                [
                                    page_size + 1,
                                    page_size + 63,
                                    page_size + 64,
                                    16 * page_size - 1,
                                    -1,
                                ]
                            ]
                        ]
                        * 65,
                        dtype=torch.int32,
                    )
                    lengths = (
                        torch.full((65,), 4, dtype=torch.int32)
                        if with_lengths
                        else None
                    )
                    split = Mock(
                        side_effect=lambda cache, size, **kw: torch.empty(
                            cache.shape[0] * size // 64, 64, 1, 584, dtype=torch.uint8
                        )
                    )
                    native = Mock()
                    module = types.ModuleType("flashinfer.mla._sparse_mla_sm120")
                    module._sparse_mla_sm120_paged_attention = native
                    namespace = {
                        "torch": torch,
                        "_PBS_SRC": 256,
                        "_PBS_DST": 64,
                        "_split_kv_pages_to_64": split,
                    }
                    exec(
                        compile(
                            ast.Module(body=[function], type_ignores=[]),
                            str(PROFILE),
                            "exec",
                        ),
                        namespace,
                    )
                    with patch.dict("sys.modules", {module.__name__: module}):
                        output, lse = namespace[function.name](
                            q,
                            main,
                            main_indices,
                            main_lengths,
                            None,
                            512,
                            512**-0.5,
                            extra,
                            indices,
                            lengths,
                        )
                    native.assert_called_once()
                    passed = native.call_args.kwargs
                    self.assertTrue(
                        torch.equal(passed["extra_indices"], indices.squeeze(1))
                    )
                    self.assertIs(passed["extra_topk_length"], lengths)
                    self.assertIs(passed["topk_length"], main_lengths)
                    self.assertEqual(len(split.call_args_list), 2)
                    self.assertEqual(split.call_args_list[1].args[1], page_size)
                    self.assertEqual(
                        split.call_args_list[1].kwargs,
                        {
                            "touched_indices": None,
                            "buf_key": "flash_mla_sm120_split_extra",
                        },
                    )
                    self.assertEqual(list(output.shape), [65, 1, 8, 512])
                    self.assertIsNone(lse)


if __name__ == "__main__":
    unittest.main()
