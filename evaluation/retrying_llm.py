"""
A judge client that re-asks when the provider returns an empty body.

DeepSeek intermittently answers with HTTP 200 and no content. It is not a
rate limit, not a truncation, and not a malformed reply — the body is simply
empty, so nothing downstream can parse it.

That matters more than the rate suggests. RAGAS treats a parse failure as
fatal: it tries three repair prompts (which cannot help, since there is no
text to repair) and then aborts the entire evaluation, discarding every
result gathered so far. Scoring one question takes roughly six calls; a full
four-stage ablation takes a few hundred. At even a 1-in-50 empty rate, a
single question almost always succeeds and a full run almost never does.

Re-asking the one bad call is enough to fix it, because the emptiness is
transient — the same prompt returns normal output on the next attempt.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
)
from llama_index.llms.openai_like import OpenAILike


def _is_blank(text: str | None) -> bool:
    """True when the provider returned nothing usable."""
    return not (text or "").strip()


class RetryOnEmptyLLM(OpenAILike):
    """
    OpenAILike that retries a call whose response body comes back empty.

    Only empties are retried here. Rate limits, timeouts and transport errors
    are already handled by the underlying client's own retry logic.
    """

    max_empty_retries: int = 4
    empty_retry_delay: float = 2.0

    def _report(self, attempt: int) -> None:
        print(
            f"--- Judge returned an empty response "
            f"(attempt {attempt}/{self.max_empty_retries + 1}); re-asking ---",
            flush=True,
        )

    # --- completion -------------------------------------------------------
    def complete(self, *args: Any, **kwargs: Any) -> CompletionResponse:
        for attempt in range(1, self.max_empty_retries + 2):
            response = super().complete(*args, **kwargs)
            if not _is_blank(response.text):
                return response
            if attempt <= self.max_empty_retries:
                self._report(attempt)
                time.sleep(self.empty_retry_delay * attempt)
        return response

    async def acomplete(self, *args: Any, **kwargs: Any) -> CompletionResponse:
        import asyncio

        for attempt in range(1, self.max_empty_retries + 2):
            response = await super().acomplete(*args, **kwargs)
            if not _is_blank(response.text):
                return response
            if attempt <= self.max_empty_retries:
                self._report(attempt)
                await asyncio.sleep(self.empty_retry_delay * attempt)
        return response

    # --- chat -------------------------------------------------------------
    def chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        for attempt in range(1, self.max_empty_retries + 2):
            response = super().chat(messages, **kwargs)
            if not _is_blank(response.message.content):
                return response
            if attempt <= self.max_empty_retries:
                self._report(attempt)
                time.sleep(self.empty_retry_delay * attempt)
        return response

    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        import asyncio

        for attempt in range(1, self.max_empty_retries + 2):
            response = await super().achat(messages, **kwargs)
            if not _is_blank(response.message.content):
                return response
            if attempt <= self.max_empty_retries:
                self._report(attempt)
                await asyncio.sleep(self.empty_retry_delay * attempt)
        return response
