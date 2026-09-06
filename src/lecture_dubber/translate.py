from __future__ import annotations

import json
import time

import httpx

from .models import Config, TranslationUnit
from .utils import extract_json

SYSTEM = """You translate university lectures into natural spoken Mandarin Chinese for dubbing.
Preserve technical meaning. Do not add explanations. Prefer concise spoken Chinese over literal translation.
Preserve variable names, equations, code identifiers, model names, and established English technical terms when appropriate.
Respect the glossary. The spoken duration should fit the target time window.
Return JSON only with the shape: {\"translation\": \"...\"}."""


class OpenAICompatibleTranslator:
    def __init__(self, cfg: Config, glossary: dict[str, str]):
        self.cfg = cfg
        self.glossary = glossary
        self.client = httpx.Client(timeout=180.0)
        import itertools
        self._seq = itertools.count()

    def _chat(self, messages: list[dict[str, str]], retries: int = 2) -> str:
        url = self.cfg.llm_base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.cfg.llm_api_key}"}
        content = ""
        r = None
        for attempt in range(retries + 1):
            # httpx's json= uses ensure_ascii=True, which escapes Chinese into
            # \uXXXX and confuses small models; send UTF-8 text directly.
            # json_schema (not json_object - broken on llama.cpp b10795) makes
            # the server enforce a grammar, so the reply is always valid JSON.
            payload = {
                "model": self.cfg.llm_model,
                "messages": messages,
                "temperature": self.cfg.llm_temperature,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "translation",
                        "schema": {
                            "type": "object",
                            "properties": {"translation": {"type": "string"}},
                            "required": ["translation"],
                        },
                    },
                },
            }
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
            r = self.client.post(url, content=body, headers=headers)
            r.raise_for_status()
            try:
                content = r.json()["choices"][0]["message"]["content"] or ""
            except (ValueError, KeyError, IndexError):
                raise RuntimeError(
                    f"unexpected LLM response: status={r.status_code} body={r.text[:300]!r}"
                )
            if content.strip():
                return content
            # Transient empty replies happen when the GPU is momentarily saturated
            # (e.g. a parallel vision request encoding frames).
            time.sleep(1.0 * (attempt + 1))
        raise RuntimeError(
            f"empty LLM response after {retries + 1} attempts: status={r.status_code} body={r.text[:300]!r}"
        )

    def _chat_json(self, messages: list[dict[str, str]], attempts: int = 3) -> dict:
        # The 2B model occasionally answers in free text instead of JSON despite
        # the response_format grammar; resampling the request recovers it. Each
        # retry adds a nonce so the server's prompt cache serves a fresh path -
        # a poisoned cache entry otherwise reproduces the same bad output.
        raw = ""
        last_error: Exception | None = None
        for attempt in range(attempts):
            raw = self._chat(messages)
            try:
                return extract_json(raw)
            except (ValueError, RuntimeError) as e:
                last_error = e
                time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"unparseable LLM output after {attempts} attempts: {raw[:200]!r}") from last_error


    def translate(self, unit: TranslationUnit, previous: str, next_text: str) -> str:
        glossary_text = "\n".join(f"- {k}: {v}" for k, v in self.glossary.items()) or "(none)"
        user = f"""Target language: Simplified Chinese
Available speaking time: {unit.duration:.2f} seconds
Previous context: {previous or '(none)'}
Current source: {unit.source}
Next context: {next_text or '(none)'}
Glossary:\n{glossary_text}
Translate only the current source. Keep it concise enough for dubbing."""
        result = self._chat_json([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
        return str(result["translation"]).strip()

    def compress(self, unit: TranslationUnit, actual_duration: float) -> str:
        user = f"""The current Chinese dubbing is too long.
Source: {unit.source}
Current translation: {unit.translation}
Current spoken duration: {actual_duration:.2f} seconds
Target duration: {unit.duration:.2f} seconds
Rewrite the translation substantially more concisely without losing technical content. Return JSON only."""
        result = self._chat_json([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
        return str(result["translation"]).strip()
