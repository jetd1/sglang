"""Mixin for testing reasoning_tokens in usage (thinking vs non-thinking).

Requires the host test class to set:
    cls.model              - model name
    cls.base_url           - server base URL (e.g. "http://127.0.0.1:30000")
    cls.reasoning_parser_name - reasoning parser name (e.g. "qwen3")

Optional:
    cls.api_key - API key (default: no auth)
"""

import json

import requests

from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.utils.hf_transformers_utils import get_tokenizer


class ReasoningTokenUsageMixin:
    """Test reasoning_tokens field in chat completion and generate API usage.

    Sends paired requests with enable_thinking=True / False and asserts
    reasoning_tokens > 0 or == 0 accordingly.  Also verifies exact token
    count via /generate API against output_ids.
    """

    reasoning_parser_name = None  # host class must override

    @classmethod
    def init_reasoning_token_verifier(cls):
        """Call from host setUpClass to set up tokenizer and think_end_token_id."""
        assert cls.reasoning_parser_name, "reasoning_parser_name must be set"
        cls.tokenizer = get_tokenizer(cls.model)
        parser = ReasoningParser(cls.reasoning_parser_name)
        cls.think_end_token_id = cls.tokenizer.convert_tokens_to_ids(
            parser.detector.think_end_token
        )
        assert cls.think_end_token_id is not None

    # -- helpers --

    def _reasoning_chat_request(self, enable_thinking, stream=False):
        api_key = getattr(self, "api_key", None)
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": "What is 1+3?"}],
            "max_tokens": 1024,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        return requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers=headers,
            json=payload,
            stream=stream,
        )

    def _extract_streaming_usage(self, response):
        usage = None
        for line in response.iter_lines():
            if not line:
                continue
            decoded = line.decode("utf-8")
            if not decoded.startswith("data:") or decoded.startswith("data: [DONE]"):
                continue
            data = json.loads(decoded[len("data:") :].strip())
            if data.get("usage"):
                usage = data["usage"]
        return usage

    # -- chat API tests --

    def test_reasoning_tokens_thinking(self):
        resp = self._reasoning_chat_request(enable_thinking=True)
        self.assertEqual(resp.status_code, 200, resp.text)
        usage = resp.json()["usage"]
        self.assertGreater(usage["reasoning_tokens"], 0)
        self.assertLess(usage["reasoning_tokens"], usage["completion_tokens"])

    def test_reasoning_tokens_non_thinking(self):
        resp = self._reasoning_chat_request(enable_thinking=False)
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["usage"]["reasoning_tokens"], 0)

    def test_reasoning_tokens_thinking_stream(self):
        with self._reasoning_chat_request(enable_thinking=True, stream=True) as resp:
            self.assertEqual(resp.status_code, 200, resp.text)
            usage = self._extract_streaming_usage(resp)
            self.assertIsNotNone(usage, "No usage in stream")
            self.assertGreater(usage["reasoning_tokens"], 0)
            self.assertLess(usage["reasoning_tokens"], usage["completion_tokens"])

    def test_reasoning_tokens_non_thinking_stream(self):
        with self._reasoning_chat_request(enable_thinking=False, stream=True) as resp:
            self.assertEqual(resp.status_code, 200, resp.text)
            usage = self._extract_streaming_usage(resp)
            self.assertIsNotNone(usage, "No usage in stream")
            self.assertEqual(usage["reasoning_tokens"], 0)

    # -- generate API exact count verification --

    def test_reasoning_tokens_generate_exact_count(self):
        """Verify reasoning_tokens matches think_end_token position in output_ids."""
        api_key = getattr(self, "api_key", None)
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        messages = [{"role": "user", "content": "What is 1+3?"}]
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        resp = requests.post(
            f"{self.base_url}/generate",
            headers=headers,
            json={
                "text": prompt,
                "sampling_params": {"max_new_tokens": 1024},
                "require_reasoning": True,
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        data = resp.json()
        reported = data["meta_info"]["reasoning_tokens"]
        output_ids = data["output_ids"]
        actual = output_ids.index(self.think_end_token_id) + 1
        self.assertEqual(
            reported,
            actual,
            f"reasoning_tokens mismatch: reported={reported}, actual={actual}",
        )
