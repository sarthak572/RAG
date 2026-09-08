"""Stage 6: Generation.

Sends the constructed prompt to the LLM and returns the generated answer,
along with token usage and latency -- captured here because bolting on
observability after the fact is much more painful than capturing it at
the one place the API call actually happens.
"""

import time
from dataclasses import dataclass

from google import genai

from rag.config import GEMINI_API_KEY, GENERATION_MODEL

_client = genai.Client(api_key=GEMINI_API_KEY)


@dataclass
class Generation:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float


def generate_answer(prompt: str) -> Generation:
    """Call the LLM with the given prompt and return the answer + metadata.

    Data flow: prompt string -> Gemini generateContent -> answer text
    """
    start = time.monotonic()
    response = _client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
    )
    latency = time.monotonic() - start

    usage = response.usage_metadata
    return Generation(
        text=response.text,
        model=GENERATION_MODEL,
        input_tokens=usage.prompt_token_count if usage else 0,
        output_tokens=usage.candidates_token_count if usage else 0,
        latency_seconds=latency,
    )
