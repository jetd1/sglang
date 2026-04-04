"""Mixin for testing reasoning_tokens in usage (thinking vs non-thinking).

Requires the host test class to set:
    cls.model   - model name
    cls.base_url - server base URL (e.g. "http://127.0.0.1:30000")

Optional:
    cls.api_key - API key (default: no auth)
"""

import json

import requests


class ReasoningTokenUsageMixin:
    """Test reasoning_tokens field in chat completion usage.

    Sends paired requests with enable_thinking=True / False and asserts
    reasoning_tokens > 0 or == 0 accordingly.
    """

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

    def _extract_streaming_reasoning_tokens(self, response):
        reasoning_tokens = None
        for line in response.iter_lines():
            if not line:
                continue
            decoded = line.decode("utf-8")
            if not decoded.startswith("data:") or decoded.startswith("data: [DONE]"):
                continue
            data = json.loads(decoded[len("data:") :].strip())
            usage = data.get("usage")
            if usage and "reasoning_tokens" in usage:
                reasoning_tokens = usage["reasoning_tokens"]
        return reasoning_tokens

    # -- non-streaming tests --

    def test_reasoning_tokens_thinking(self):
        resp = self._reasoning_chat_request(enable_thinking=True)
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertGreater(resp.json()["usage"]["reasoning_tokens"], 0)

    def test_reasoning_tokens_non_thinking(self):
        resp = self._reasoning_chat_request(enable_thinking=False)
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["usage"]["reasoning_tokens"], 0)

    # -- streaming tests --

    def test_reasoning_tokens_thinking_stream(self):
        resp = self._reasoning_chat_request(enable_thinking=True, stream=True)
        self.assertEqual(resp.status_code, 200)
        rt = self._extract_streaming_reasoning_tokens(resp)
        self.assertIsNotNone(rt, "No usage with reasoning_tokens in stream")
        self.assertGreater(rt, 0)

    def test_reasoning_tokens_non_thinking_stream(self):
        resp = self._reasoning_chat_request(enable_thinking=False, stream=True)
        self.assertEqual(resp.status_code, 200)
        rt = self._extract_streaming_reasoning_tokens(resp)
        self.assertIsNotNone(rt, "No usage with reasoning_tokens in stream")
        self.assertEqual(rt, 0)
