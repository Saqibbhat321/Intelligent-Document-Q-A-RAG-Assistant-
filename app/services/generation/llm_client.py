"""LLM client — OpenAI-compatible interface targeting NVIDIA NIM."""

import logging
import time
from typing import List

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log

from app.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class LLMClient:
    """
    Thin wrapper around an OpenAI-compatible API client.
    Swap NVIDIA_BASE_URL and NVIDIA_API_KEY to point at any provider
    (OpenAI, Together AI, Groq, Ollama, etc.) without changing this class.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self.model = model or settings.llm_model
        self._client = OpenAI(
            api_key=api_key or settings.nvidia_api_key,
            base_url=base_url or settings.nvidia_base_url,
        )
        logger.info(f"LLM client initialised — model={self.model}, base={base_url or settings.nvidia_base_url}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def generate(
        self,
        system_message: str,
        user_message: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, float]:
        """
        Call the LLM and return (answer_text, response_latency_seconds).

        Retries up to 3 times on transient errors with exponential back-off.
        """
        t0 = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature if temperature is not None else settings.llm_temperature,
            max_tokens=max_tokens or settings.llm_max_tokens,
        )
        latency = time.perf_counter() - t0

        answer = response.choices[0].message.content or ""
        logger.info(f"LLM response received in {latency:.2f}s ({len(answer)} chars)")
        return answer.strip(), latency
