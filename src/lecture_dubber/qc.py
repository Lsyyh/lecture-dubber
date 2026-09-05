"""VLM-based quality control: hardsub detection and dubbing render checks.

A small vision-language model (served on the same OpenAI-compatible endpoint as
the translator) inspects sampled video frames. Before rendering it detects
burned-in source subtitles so our subtitles can avoid them; after rendering it
verifies the Chinese subtitles were burned correctly.
"""
from __future__ import annotations

import base64
from pathlib import Path

import httpx

from .media import sample_frames
from .utils import extract_json


class VisionQC:
    def __init__(self, cfg):
        self.cfg = cfg
        self.client = httpx.Client(timeout=180.0)

    def _chat_images(self, prompt: str, images: list[Path]) -> dict:
        content: list[dict] = [{"type": "text", "text": prompt}]
        for image in images:
            b64 = base64.b64encode(image.read_bytes()).decode()
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
        payload = {
            "model": "qc",
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        url = self.cfg.llm_base_url.rstrip("/") + "/chat/completions"
        r = self.client.post(url, json=payload, headers={"Authorization": f"Bearer {self.cfg.llm_api_key}"})
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"]
        return extract_json(raw)

    def detect_subtitles(self, video: Path) -> str:
        """Return where burned-in source subtitles sit: "bottom", "top" or "none"."""
        frames = sample_frames(video, self.cfg.qc_frames)
        if not frames:
            return "none"
        prompt = (
            "These are frames sampled from one lecture video. Does the video show "
            "burned-in subtitles (text overlaid on the picture by the uploader)? "
            "Ignore player UI. Reply JSON only: {\"position\": \"bottom\" | \"top\" | \"none\"}."
        )
        result = self._chat_images(prompt, frames)
        position = str(result.get("position", "none")).lower()
        return position if position in {"bottom", "top", "none"} else "none"

    def check_render(self, final: Path) -> dict:
        """Verify the rendered dubbing: Chinese subtitles present and readable."""
        frames = sample_frames(final, self.cfg.qc_frames)
        if not frames:
            return {"status": "skipped", "issues": []}
        prompt = (
            "These are frames from a dubbed lecture video that must have Chinese "
            "subtitles overlaid. For each frame check: (1) Chinese subtitle text is "
            "present, (2) it is readable (no garbled characters, no blocking boxes, "
            "not overlapping other on-screen text badly). Reply JSON only: "
            "{\"subtitles_present\": true/false, \"issues\": [\"garbled\" | \"missing\" | "
            "\"overlapping\" | ...], \"notes\": \"...\"}."
        )
        try:
            result = self._chat_images(prompt, frames)
        except Exception as e:
            return {"status": "error", "issues": [f"qc request failed: {e}"]}
        return {
            "status": "ok",
            "subtitles_present": bool(result.get("subtitles_present", False)),
            "issues": [str(x) for x in result.get("issues", [])],
            "notes": str(result.get("notes", "")),
        }
