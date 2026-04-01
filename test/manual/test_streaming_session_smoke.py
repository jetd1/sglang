"""
Small manual smoke test for Dynamo-compatible streaming sessions using
OpenAI-compatible chat.

Flow:
1. Send a few normal multi-turn chat completion requests.
2. Start a streaming session with `nvext.session_control.action="open"`.
3. Send a few more session-backed chat completion requests.
4. End the session with `nvext.session_control.action="close"`.
5. Send a few more normal chat completion requests.

Examples:
    python test/manual/test_streaming_session_smoke.py --launch-server
    python test/manual/test_streaming_session_smoke.py --base-url http://localhost:8000
"""

import argparse
import json
import sys
import time
import uuid
from typing import Any, Optional

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)


PRE_SESSION_MESSAGES = [
    "Reply with a short greeting.",
    "Reply with a short sentence about Paris.",
]

SESSION_MESSAGES = [
    "My name is Bob. Reply with exactly one short acknowledgment.",
    "What is my name? Reply with one word if you know it.",
    "Now reply with a short sentence about what we just established.",
    "Repeat my name in one word.",
    "What city did we not discuss in this session? Reply with unknown.",
    "Reply with exactly two words acknowledging the session.",
    "What is my name? Reply with one word.",
    "Reply with a short sentence proving you still remember my name.",
]

CLOSE_SESSION_MESSAGE = "Reply with exactly the word closing."

POST_SESSION_MESSAGES = [
    "Reply with a short sentence about Rome.",
    "Reply with a short sentence about Berlin.",
]


def _post_json(base_url: str, route: str, payload: dict, timeout: int = 120):
    return requests.post(
        f"{base_url}{route}",
        json=payload,
        timeout=timeout,
    )


def _chat_completion(
    base_url: str,
    model: str,
    message: str,
    session_id: Optional[str] = None,
    session_action: Optional[str] = None,
    max_tokens: int = 32,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": message}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if session_id is not None:
        session_control = {"session_id": session_id}
        if session_action is not None:
            session_control["action"] = session_action
            if session_action == "open":
                session_control["timeout"] = 60
        payload["nvext"] = {"session_control": session_control}

    response = _post_json(base_url, "/v1/chat/completions", payload)
    if response.status_code != 200:
        raise RuntimeError(
            f"/v1/chat/completions failed: {response.status_code} {response.text}"
        )

    return response.json()


def _run_plain_requests(base_url: str, model: str, messages: list[str], label: str) -> None:
    print(f"{label}:")
    for idx, message in enumerate(messages, start=1):
        body = _chat_completion(base_url, model, message)
        print(f"  req {idx}:")
        print(json.dumps(body, indent=2, sort_keys=True))


def run_smoke_test(
    base_url: str,
    model: str,
    max_tokens: int,
    session_delay_seconds: float,
    pre_close_delay_seconds: float,
) -> None:
    _run_plain_requests(base_url, model, PRE_SESSION_MESSAGES, "pre-session requests")

    session_id = f"smoke-{uuid.uuid4().hex[:12]}"
    print(f"session id: {session_id}")

    for idx, message in enumerate(SESSION_MESSAGES, start=1):
        action = "open" if idx == 1 else None
        body = _chat_completion(
            base_url=base_url,
            model=model,
            message=message,
            session_id=session_id,
            session_action=action,
            max_tokens=max_tokens,
        )
        print(f"  session req {idx}:")
        print(json.dumps(body, indent=2, sort_keys=True))
        if idx != len(SESSION_MESSAGES):
            print(f"  sleeping {session_delay_seconds:.1f}s before next session request")
            time.sleep(session_delay_seconds)

    print(f"holding session open for {pre_close_delay_seconds:.1f}s before close")
    time.sleep(pre_close_delay_seconds)

    body = _chat_completion(
        base_url=base_url,
        model=model,
        message=CLOSE_SESSION_MESSAGE,
        session_id=session_id,
        session_action="close",
        max_tokens=max_tokens,
    )
    print("  session close req:")
    print(json.dumps(body, indent=2, sort_keys=True))
    print("closed session")

    _run_plain_requests(base_url, model, POST_SESSION_MESSAGES, "post-session requests")
    print("streaming session smoke test passed")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Manual smoke test for streaming sessions via /v1/chat/completions."
    )
    parser.add_argument("--base-url", default=DEFAULT_URL_FOR_TEST)
    parser.add_argument("--model", default=DEFAULT_SMALL_MODEL_NAME_FOR_TEST)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--session-delay-seconds", type=float, default=1.0)
    parser.add_argument("--pre-close-delay-seconds", type=float, default=3.0)
    parser.add_argument(
        "--launch-server",
        action="store_true",
        help="Launch a local SGLang server for the test.",
    )
    parser.add_argument(
        "--launch-timeout",
        type=int,
        default=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    )
    args = parser.parse_args()

    process = None
    try:
        if args.launch_server:
            process = popen_launch_server(
                args.model,
                args.base_url,
                timeout=args.launch_timeout,
                other_args=["--enable-streaming-session"],
            )

        run_smoke_test(
            base_url=args.base_url,
            model=args.model,
            max_tokens=args.max_tokens,
            session_delay_seconds=args.session_delay_seconds,
            pre_close_delay_seconds=args.pre_close_delay_seconds,
        )
        return 0
    finally:
        if process is not None:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    sys.exit(main())
