"""OpenRouter VLM wrapper.

Security:
- Never hardcode API keys in source code.
- Set OPENROUTER_API_KEY in the shell or a local .env file.
- .env must remain gitignored.

Environment:
    OPENROUTER_API_KEY=...
    OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
    OPENROUTER_MODEL=google/gemini-2.5-flash-preview
"""
from __future__ import annotations

import base64
import io
import json
import os
from typing import Any


DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "google/gemini-2.5-flash"


class OpenRouterVLM:
    """OpenRouter-backed VLM using the OpenAI-compatible chat API."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        site_url: str | None = None,
        app_name: str = "medical-rag-agent",
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> None:
        self.model = model or os.environ.get("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.base_url = (base_url or os.environ.get("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL)).rstrip("/")
        self.site_url = site_url or os.environ.get("OPENROUTER_SITE_URL", "")
        self.app_name = app_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set. Do not hardcode it; set an environment variable.")

    @staticmethod
    def _image_to_data_url(image: Any) -> str:
        """Convert PIL image to JPEG data URL."""
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=85)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.app_name:
            headers["X-Title"] = self.app_name
        return headers

    def _chat(self, messages: list[dict[str, Any]], max_tokens: int | None = None) -> str:
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("Install requests: pip install requests") from exc

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=120,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"OpenRouter API error {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        return str(data["choices"][0]["message"]["content"])

    def generate(
        self,
        question: str,
        image: Any | None = None,
        context: str | None = None,
        max_new_tokens: int | None = None,
    ) -> str:
        """Generate grounded medical answer from question, optional image and evidence."""
        system = (
            "You are a careful medical multimodal RAG assistant. "
            "Answer only using the provided evidence when possible. "
            "If evidence is insufficient, say so. Do not provide definitive diagnosis; "
            "phrase findings cautiously and include citations like [Evidence 1]."
        )
        prompt = (
            f"Question:\n{question}\n\n"
            f"Retrieved evidence:\n{context or 'No retrieved evidence.'}\n\n"
            "Write a concise, evidence-grounded medical answer."
        )

        if image is None:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": self._image_to_data_url(image)}},
                    ],
                },
            ]
        return self._chat(messages, max_tokens=max_new_tokens)

    def ground_region(self, question: str, image: Any) -> dict[str, Any]:
        """Ask the VLM for a normalized bbox relevant to the medical question.

        Returns:
            {"bbox": [x1, y1, x2, y2], "confidence": float, "description": str}
        """
        prompt = (
            "Given this medical image and question, identify the most relevant visual region.\n"
            f"Question: {question}\n\n"
            "Return ONLY valid JSON with this schema:\n"
            "{\"bbox\": [x1, y1, x2, y2], \"confidence\": 0.0, \"description\": \"...\"}\n"
            "Coordinates must be normalized floats from 0 to 1."
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": self._image_to_data_url(image)}},
                ],
            }
        ]
        raw = self._chat(messages, max_tokens=256)
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            parsed = json.loads(raw[start:end]) if start >= 0 and end > start else json.loads(raw)
            bbox = parsed.get("bbox", [0.25, 0.25, 0.75, 0.75])
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError("invalid bbox")
            bbox = [float(max(0.0, min(1.0, x))) for x in bbox]
            return {
                "bbox": bbox,
                "confidence": float(parsed.get("confidence", 0.5)),
                "description": str(parsed.get("description", "OpenRouter visual grounding")),
            }
        except Exception:
            return {
                "bbox": [0.25, 0.25, 0.75, 0.75],
                "confidence": 0.5,
                "description": f"Fallback center ROI. Raw response: {raw[:200]}",
            }
