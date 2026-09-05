from __future__ import annotations

import json
import httpx
from .models import Config, TranslationUnit


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

    def _chat(self, messages: list[dict[str, str]]) -> str:
        url = self.cfg.llm_base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.cfg.llm_model,
            "messages": messages,
            "temperature": self.cfg.llm_temperature,
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self.cfg.llm_api_key}"}
        r = self.client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def translate(self, unit: TranslationUnit, previous: str, next_text: str) -> str:
        glossary_text = "\n".join(f"- {k}: {v}" for k, v in self.glossary.items()) or "(none)"
        user = f"""Target language: Simplified Chinese
Available speaking time: {unit.duration:.2f} seconds
Previous context: {previous or '(none)'}
Current source: {unit.source}
Next context: {next_text or '(none)'}
Glossary:\n{glossary_text}
Translate only the current source. Keep it concise enough for dubbing."""
        raw = self._chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
        return str(json.loads(raw)["translation"]).strip()

    def compress(self, unit: TranslationUnit, actual_duration: float) -> str:
        user = f"""The current Chinese dubbing is too long.
Source: {unit.source}
Current translation: {unit.translation}
Current spoken duration: {actual_duration:.2f} seconds
Target duration: {unit.duration:.2f} seconds
Rewrite the translation substantially more concisely without losing technical content. Return JSON only."""
        raw = self._chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
        return str(json.loads(raw)["translation"]).strip()
