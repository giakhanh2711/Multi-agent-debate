from openai import OpenAI, APIConnectionError, APIError, RateLimitError
from dotenv import load_dotenv
import time
import os
import re
import json
from src.configs import TAMU_BASE_URL, SONNET_MODEL
import src.utils as utils


def extract_final_answer(text: str) -> str:
    text = (text or "").strip()

    matches = re.findall(
        r"(?im)^\s*(?:final\s+answer|answer)\s*[:\-]\s*(.*?)\s*$",
        text
    )

    candidate = matches[-1] if matches else ""

    if not candidate:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        candidate = lines[-1] if lines else ""

    # Remove numbering and common Markdown formatting
    candidate = re.sub(r"^\s*[\*_\-\d.)]+\s*", "", candidate)
    candidate = re.sub(r"[*_`]", "", candidate)

    return candidate.strip().strip('"').strip("'").strip()


class TAMUChatClient:
    def __init__(
        self,
        api_url=TAMU_BASE_URL,
        model_name=SONNET_MODEL,
        min_delay=1.0,
        max_retries=4
    ):
        load_dotenv(".env")

        key = os.environ.get("TAMU_API_KEY")
        cookie = os.environ.get("CF_COOKIE")

        if not key or not cookie:
            raise EnvironmentError("Missing TAMU credentials.")

        self.model = model_name
        self.min_delay = min_delay
        self.max_retries = max_retries
        self.last_call = 0.0

        self.client = OpenAI(
            api_key=key,
            base_url=api_url,
            default_headers={
                "Cookie": utils.normalize_cf_cookie(cookie)
            }
        )


    # 2.1 Parse each LLM's response to extract its chosen answer
    
    def _visible_content(self, response):
        """Extract visible text from TAMU SSE or OpenAI-compatible responses."""

        model = None

        if isinstance(response, str):
            if "Cloudflare Access" in response and "Sign in" in response:
                raise RuntimeError(
                    "TAMU gateway returned the Cloudflare sign-in page; "
                    "refresh CF_Authorization."
                )

            if response.lstrip().startswith("data:"):
                pieces = []

                for line in response.splitlines():
                    if not line.startswith("data:"):
                        continue

                    payload = line[5:].strip()

                    if not payload or payload == "[DONE]":
                        continue

                    try:
                        event = json.loads(payload)
                    except json.JSONDecodeError:
                        continue

                    if not model:
                        model = event.get("model")

                    for choice in event.get("choices", []):
                        piece = choice.get("delta", {}).get("content")

                        if isinstance(piece, str):
                            pieces.append(piece)

                if pieces:
                    content = "".join(pieces)

                    content = re.sub(
                        r"<think>.*?</think>",
                        "",
                        content,
                        flags=re.DOTALL
                    ).strip()

                    return content

                raise ValueError("No text content found in SSE response.")

            return response.strip()

        if isinstance(response, dict):
            choices = response.get("choices", [])

            if choices:
                return choices[0].get("message", {}).get("content", "") or ""

        try:
            return response.choices[0].message.content or ""

        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise TypeError(
                f"Unsupported TAMU completion response: "
                f"{type(response).__name__}"
            ) from exc

           
    def _response_usage(self, response):
        if isinstance(response, str):
            if response.lstrip().startswith("data:"):
                for line in response.splitlines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if not payload or payload == "[DONE]":
                        continue
                    try:
                        event = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    usage = event.get("usage")
                    if usage is not None:
                        return usage
            else:
                try:
                    response = json.loads(response)
                except json.JSONDecodeError:
                    return None
        if isinstance(response, dict):
            return response.get("usage")
        return getattr(response, "usage", None)


    def chat_with_usage(self, messages, temperature=1.0):
        """
        Return only the final answer
        """
        for attempt in range(self.max_retries):

            elapsed = time.monotonic() - self.last_call

            if elapsed < self.min_delay:
                time.sleep(self.min_delay - elapsed)

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=16_384
                )
                
                visible = self._visible_content(response)

                self.last_call = time.monotonic()
                # print(f"llm response {response}")
                return extract_final_answer(visible)

            except (
                RateLimitError,
                APIConnectionError,
                APIError
            ):

                if attempt == self.max_retries - 1:
                    raise

                time.sleep(2 ** attempt)
                