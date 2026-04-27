"""
DeepSeek-V4 reasoning parser patch for SGLang (v2 — carve, don't terminate).

Problem
-------
DS-V4 sometimes emits DSML tool-call markup (<｜DSML｜tool_calls>...
</｜DSML｜tool_calls>) _inside_ a thinking block, before </think> closes.
The stock Qwen3Detector (which DS-V4 maps to in the registry) treats
everything between <think>/</think> as reasoning_content, so DSML inside
thinking never reaches the tool-call detector → downstream sees 0
tool_calls and gets stuck.

Design (Jet, 2026-04-26)
------------------------
"Tool invocation should not depend on thinking-block closure. When a
complete DSML tool call appears inside an unclosed thinking block, invoke
immediately. DSML is a structured action token; it's already committed."

But: Anthropic-style interleaved thinking semantics mean thinking is NOT
terminated by a tool call — it continues across tool calls within a single
assistant turn:

    thinking₁ → tool_use₁ → thinking₂ → tool_use₂ → final_text

So the correct fix is to CARVE the DSML block out of reasoning_text and
hand it to normal_text (for the tool detector), while letting reasoning
continue both before and after the DSML block. The thinking block stays
open until </think> is seen.

Approach
--------
DeepSeekV4ReasoningDetector overrides parse_streaming_increment with its
own buffer/state machine that mirrors the parent's logic but adds DSML
carve-out. It does NOT call super() to avoid the parent's "stream-and-
flush" behavior swallowing partial DSML start tokens.

State per chunk while in reasoning:
  1. Append new_text to internal buffer
  2. If </think> appears: parent semantics — emit reasoning up to it,
     normal_text after, exit reasoning mode
  3. If complete DSML pair (start + end) appears in buffer:
     - emit (reasoning_before, dsml_block_as_normal_text)
     - keep reasoning_after in buffer for next chunk
     - stay in reasoning mode
  4. If DSML start appears but no end yet:
     - emit reasoning_before as reasoning_text
     - keep DSML-start-onward in buffer (waiting for end)
  5. If buffer ends with a partial DSML start prefix:
     - emit (buffer minus prefix) as reasoning_text
     - keep prefix in buffer
  6. Otherwise: emit whole buffer as reasoning_text

Behavioral guarantees
---------------------
- Never injects DSML tokens into the visible content stream when no
  matching closing tag exists (false-positive mention is safe).
- Never terminates reasoning before </think> arrives.
- Carved DSML block goes to normal_text → SGLang's tool detector picks
  it up as a regular tool call.
- Reasoning before AND after the carved DSML continues to stream as
  reasoning_text (interleaved-thinking semantics).

Registry change
---------------
Maps "deepseek-v4" → DeepSeekV4ReasoningDetector instead of Qwen3Detector.
All other model mappings are preserved.

Activation: copy this file into sglang/srt/parser/, then at startup do
    ReasoningParser.DetectorMap['deepseek-v4'] = DeepSeekV4ReasoningDetector
The b300 entrypoint script handles both steps.
"""

from __future__ import annotations

from sglang.srt.parser.reasoning_parser import (
    Qwen3Detector,
    StreamingParseResult,
)

# DS-V4 tool-call wrapper tokens. These are the SAME tokens that
# DeepSeekV4Detector (function_call/deepseekv4_detector.py) uses; we
# duplicate them as constants here to avoid a cross-module dependency
# from the reasoning parser into the function_call package.
_DSML_TOOL_CALL_START = "<\uff5cDSML\uff5ctool_calls>"
_DSML_TOOL_CALL_END = "</\uff5cDSML\uff5ctool_calls>"


def _longest_partial_prefix_len(text: str, tokens) -> int:
    """Return length of longest suffix of `text` that is a strict prefix
    of ANY token in `tokens` (i.e. `text` ends with the start of some
    token). Returns 0 if no overlap.

    Used to decide how many trailing bytes of a streamed chunk to keep
    buffered in case the next chunk completes a boundary token (DSML
    start or </think>).
    """
    if isinstance(tokens, str):
        tokens = (tokens,)
    best = 0
    for token in tokens:
        max_check = min(len(text), len(token) - 1)
        for k in range(max_check, best, -1):
            if text.endswith(token[:k]):
                best = k
                break
    return best


class DeepSeekV4ReasoningDetector(Qwen3Detector):
    """Qwen3Detector with DSML carve-out inside reasoning.

    Inherits init/state from Qwen3Detector (think_start_token=<think>,
    think_end_token=</think>, force_reasoning controllable). Replaces
    parse_streaming_increment with a buffered state machine that carves
    complete DSML blocks out of reasoning into normal_text, without
    terminating the reasoning state.
    """

    # ------------------------------------------------------------------
    # Non-streaming path. OpenClaw's openai-completions API path lands
    # here (one-shot detect_and_parse on the fully assembled response),
    # so we MUST override it too. Stock Qwen3Detector splits on </think>
    # and dumps the entire pre-</think> chunk into reasoning_text, leaving
    # any in-thinking DSML stuck. We carve the same way the streaming
    # path does.
    # ------------------------------------------------------------------
    def detect_and_parse(self, text: str) -> StreamingParseResult:
        in_reasoning = self._in_reasoning or self.think_start_token in text
        if not in_reasoning:
            # Even when no <think> wrapper exists, DSML can leak into
            # the visible stream from a model that mis-emits it. Try to
            # carve from raw text too.
            reasoning_part, normal_part = self._carve_dsml(text)
            # No reasoning context here — so anything outside DSML is
            # normal text. Concatenate carved-out reasoning back into
            # normal_text since there's no reasoning channel to send it on.
            joined = (reasoning_part + normal_part) if normal_part else text
            return StreamingParseResult(normal_text=joined)

        processed = text.replace(self.think_start_token, "")

        if self.think_end_token in processed:
            pre, post = processed.split(self.think_end_token, maxsplit=1)
            reasoning_part, dsml_normal = self._carve_dsml(pre)
            return StreamingParseResult(
                reasoning_text=(reasoning_part.strip() or None),
                normal_text=(dsml_normal + post).strip() or None,
            )

        # No </think> seen — model truncated thinking. Still carve any
        # complete DSML pair so tool calls can fire.
        reasoning_part, dsml_normal = self._carve_dsml(processed)
        return StreamingParseResult(
            reasoning_text=(reasoning_part.strip() or None),
            normal_text=(dsml_normal or None),
        )

    def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:
        self._buffer += new_text
        current = self._buffer

        # ---- Phase 1: handle think_start --------------------------------
        # If we haven't yet stripped the <think> opener and the buffer is
        # short enough to still be a partial of <think>, hold and wait.
        # (Mirrors parent behavior; necessary because we don't call super().)
        if not self.stripped_think_start:
            if any(
                tok.startswith(current) and tok != current
                for tok in (self.think_start_token, self.think_end_token)
            ):
                # current text is a strict prefix of one of the boundary
                # tokens — wait for more text before deciding.
                return StreamingParseResult()

            if self.think_start_token in current:
                current = current.replace(self.think_start_token, "")
                self.stripped_think_start = True
                self._in_reasoning = True
                self._buffer = current
            elif not self._in_reasoning:
                # No <think> ever seen and not force_reasoning → passthrough.
                self._buffer = ""
                return StreamingParseResult(normal_text=current)
            # If _in_reasoning was set by force_reasoning at construction time,
            # we already act as if <think> was seen.
            else:
                self.stripped_think_start = True

        # ---- Phase 2: handle </think> -----------------------------------
        # If </think> is in the buffer, end reasoning at that boundary.
        # Anything before </think> may still contain DSML which we carve;
        # anything after </think> goes to normal_text. We do this carefully:
        # carve DSML from the pre-</think> portion so a stuck-tool DSML
        # inside the LAST thinking block is still recovered.
        if self._in_reasoning and self.think_end_token in current:
            end_idx = current.find(self.think_end_token)
            pre_close = current[:end_idx]
            post_close = current[end_idx + len(self.think_end_token):]

            # Carve DSML from pre_close if a complete pair exists.
            reasoning_part, normal_part = self._carve_dsml(pre_close)

            self._buffer = ""
            self._in_reasoning = False

            return StreamingParseResult(
                reasoning_text=(reasoning_part.rstrip() if reasoning_part else None),
                normal_text=(normal_part + post_close) if (normal_part or post_close) else None,
            )

        # ---- Phase 3: in reasoning, no </think> yet ---------------------
        if self._in_reasoning:
            # Carve ALL complete DSML pairs in `current` left-to-right.
            # One backend chunk can legitimately contain tool₁+tool₂ in a
            # single delta (interleaved thinking) before the stream pauses
            # for tool execution; emitting only the first leaves the second
            # stuck in the buffer indefinitely (codex review P1.2).
            reasoning_chunks: list[str] = []
            normal_chunks: list[str] = []
            cursor = 0
            while True:
                s = current.find(_DSML_TOOL_CALL_START, cursor)
                if s == -1:
                    break
                e = current.find(
                    _DSML_TOOL_CALL_END, s + len(_DSML_TOOL_CALL_START)
                )
                if e == -1:
                    break  # start present, end not yet — stop carving here
                e += len(_DSML_TOOL_CALL_END)
                reasoning_chunks.append(current[cursor:s])
                normal_chunks.append(current[s:e])
                cursor = e

            tail = current[cursor:]

            # Decide how much of `tail` to hold in buffer for the next
            # chunk: an unclosed DSML start (must wait for end), or a
            # partial prefix of any boundary token at the very end.
            unclosed_start = tail.find(_DSML_TOOL_CALL_START)
            if unclosed_start != -1:
                emit_tail = tail[:unclosed_start]
                hold_tail = tail[unclosed_start:]
            else:
                partial_len = _longest_partial_prefix_len(
                    tail, (_DSML_TOOL_CALL_START, self.think_end_token),
                )
                if partial_len > 0:
                    emit_tail = tail[:-partial_len] if partial_len < len(tail) else ""
                    hold_tail = tail[-partial_len:]
                else:
                    emit_tail = tail
                    hold_tail = ""

            reasoning_chunks.append(emit_tail)
            reasoning_out = "".join(reasoning_chunks)
            normal_out = "".join(normal_chunks)

            # Non-streaming mode: hold ALL reasoning in buffer so it accumulates
            # across calls and is finally flushed by Phase 2 when </think>
            # arrives. DSML carved blocks are still emitted now — they're
            # bounded events the tool detector must see synchronously
            # (codex review P1.1: stock parser's contract).
            if not self.stream_reasoning:
                self._buffer = reasoning_out + hold_tail
                if normal_out:
                    return StreamingParseResult(normal_text=normal_out)
                return StreamingParseResult()

            # Streaming mode: emit reasoning + carved DSML now.
            self._buffer = hold_tail
            return StreamingParseResult(
                reasoning_text=reasoning_out if reasoning_out else None,
                normal_text=normal_out if normal_out else None,
            )

        # ---- Phase 4: not in reasoning (post-</think>) ------------------
        self._buffer = ""
        return StreamingParseResult(normal_text=current)

    # ------------------------------------------------------------------
    @staticmethod
    def _carve_dsml(text: str) -> tuple[str, str]:
        """Split `text` (a finalized reasoning slice) into (reasoning_part,
        normal_part) by carving out any complete DSML tool_call blocks.

        Multiple DSML blocks are handled left-to-right. Reasoning content
        between/around DSML blocks is concatenated into reasoning_part.
        DSML blocks are concatenated into normal_part in order.
        """
        reasoning_chunks: list[str] = []
        normal_chunks: list[str] = []
        cursor = 0
        while True:
            s = text.find(_DSML_TOOL_CALL_START, cursor)
            if s == -1:
                reasoning_chunks.append(text[cursor:])
                break
            e = text.find(_DSML_TOOL_CALL_END, s + len(_DSML_TOOL_CALL_START))
            if e == -1:
                # Unclosed DSML inside finalized reasoning slice — leave it
                # in reasoning. (This is a model-side bug; safer than
                # injecting half a tool call into normal_text.)
                reasoning_chunks.append(text[cursor:])
                break
            e += len(_DSML_TOOL_CALL_END)
            reasoning_chunks.append(text[cursor:s])
            normal_chunks.append(text[s:e])
            cursor = e
        return "".join(reasoning_chunks), "".join(normal_chunks)
