"""Selected-token logprob normalization must accept both tensor and list rows."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.logprob_processor import LogprobStage, get_token_ids_logprobs_raw
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestSelectedTokenLogprobNormalization(CustomTestCase):
    def _check_normalization(self, path):
        processor = SchedulerBatchResultProcessor.__new__(SchedulerBatchResultProcessor)
        batch = SimpleNamespace(
            return_logprob=True,
            spec_algorithm=SimpleNamespace(is_none=lambda: True),
        )
        logprobs = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]])
        for token_ids in ([None, [1]], [[0], [1]], [None, None]):
            for no_copy_to_cpu in (True, False):
                with self.subTest(token_ids=token_ids, no_copy=no_copy_to_cpu):
                    values, indices = get_token_ids_logprobs_raw(
                        logprobs,
                        token_ids,
                        stage=LogprobStage.DECODE,
                        no_copy_to_cpu=no_copy_to_cpu,
                    )
                    expected = [
                        [] if ids is None else logprobs[i, ids].tolist()
                        for i, ids in enumerate(token_ids)
                    ]
                    list_rows = {
                        i: row for i, row in enumerate(values) if isinstance(row, list)
                    }
                    output = LogitsProcessorOutput(
                        next_token_logits=None,
                        next_token_logprobs=torch.tensor([-1.0, -3.0]),
                        next_token_token_ids_logprobs_val=values,
                        next_token_token_ids_logprobs_idx=indices,
                    )
                    if path == "prefill":
                        processor.move_logprobs_to_cpu(
                            batch=batch, logits_output=output
                        )
                    else:
                        ids, next_logprobs = processor._normalize_decode_outputs(
                            batch=batch,
                            result=None,
                            logits_output=output,
                            next_token_ids=torch.tensor([0, 1]),
                        )
                        self.assertEqual(ids, [[0], [1]])
                        self.assertEqual(next_logprobs, [-1.0, -3.0])

                    self.assertEqual(output.next_token_token_ids_logprobs_val, expected)
                    self.assertIs(output.next_token_token_ids_logprobs_idx, indices)
                    for i, row in list_rows.items():
                        self.assertIs(output.next_token_token_ids_logprobs_val[i], row)

    def test_prefill_selected_token_logprob_rows(self):
        self._check_normalization("prefill")

    def test_decode_selected_token_logprob_rows(self):
        self._check_normalization("decode")


if __name__ == "__main__":
    unittest.main()
