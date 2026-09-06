"""Interpretation-style translation for the realtime mode.

Reuses the offline translator's HTTP layer (json_schema grammar, UTF-8 body,
nonce retries) but with a simultaneous-interpretation prompt, a bounded
context window and latency-mode hints.
"""

from __future__ import annotations

from .translate import OpenAICompatibleTranslator

SYSTEM = """You are a simultaneous interpreter translating an English technical
lecture into concise natural spoken Mandarin Chinese, in real time.
Rules:
- Preserve technical meaning. Never explain, elaborate, or add information.
- Prefer concise spoken Chinese over literal translation.
- Drop English filler expressions (um, so, well, you know) when safe.
- Preserve variable names, equations, code identifiers and model names; follow the glossary.
- Stay consistent with the recent context you were given.
- The output is synthesized to speech immediately; short wording is preferred.
Return JSON only with the shape: {"translation": "..."}."""


class RealtimeTranslator(OpenAICompatibleTranslator):
    def interpret(
        self,
        source_text: str,
        prev_source: str,
        prev_zh: str,
        glossary: dict[str, str],
        mode_hint: str,
    ) -> str:
        glossary_text = "\n".join(f"- {k}: {v}" for k, v in glossary.items()) or "(none)"
        parts = [
            "Target language: Simplified Chinese",
            f"Previous source clause: {prev_source or '(none)'}",
            f"Your previous Chinese output: {prev_zh or '(none)'}",
            f"CURRENT SOURCE CLAUSE (translate ONLY this): <<{source_text}>>",
            f"Glossary:\n{glossary_text}",
        ]
        if mode_hint:
            parts.append(f"Latency mode instruction: {mode_hint}")
        parts.append(
            "Translate only the text between << >>. If it is only a filler or "
            "incomplete fragment with no meaning, return a very short clause."
        )
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "\n".join(parts)},
        ]
        result = self._chat_json(messages, attempts=2)
        return str(result["translation"]).strip()
