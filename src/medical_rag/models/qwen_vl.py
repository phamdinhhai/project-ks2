"""Qwen2.5-VL vision-language model wrapper.

Provides:
- generate(): text + optional image → answer string
- ground_region(): image + question → bounding box dict

Reference: VimRAG (src/support_repo/VimRAG_project) uses Qwen2.5-VL for
visual grounding and region-aware caption generation.

Supports:
- Local 4-bit quantized inference (GPU ≥16GB)
- API mode (DashScope / OpenAI-compatible) as fallback
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MODEL_7B = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_MODEL_3B = "Qwen/Qwen2.5-VL-3B-Instruct"


def _detect_vram_gb() -> float:
    """Return available VRAM in GB, or 0 if no CUDA."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return props.total_memory / 1024 ** 3
    except Exception:
        pass
    return 0.0


def _auto_select_model(vram_gb: float) -> str:
    """Choose 3B for <= 6GB VRAM, 7B otherwise."""
    if vram_gb >= 6.0:
        return DEFAULT_MODEL_7B
    return DEFAULT_MODEL_3B


class QwenVLModel:
    """Qwen2.5-VL wrapper with hardware-aware model selection.

    Automatically selects 3B for VRAM <= 6GB, 7B otherwise.
    Supports local 4-bit inference, CPU offload, and API fallback.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        use_api: bool = False,
        api_key: str | None = None,
        api_base: str | None = None,
        use_cpu_offload: bool = True,
        max_new_tokens: int = 256,
    ):
        vram_gb = _detect_vram_gb()
        if model_name is None:
            model_name = _auto_select_model(vram_gb)
            logger.info(f"Auto-selected model {model_name} for VRAM={vram_gb:.1f}GB")
        self.model_name = model_name
        self._device = device
        self.use_cpu_offload = use_cpu_offload
        self.max_new_tokens = max_new_tokens
        self.vram_gb = vram_gb
        self.use_api = use_api or bool(os.environ.get("QWEN_API_KEY"))
        self.api_key = api_key or os.environ.get("QWEN_API_KEY", "")
        self.api_base = api_base or os.environ.get(
            "QWEN_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self._model: Any = None
        self._processor: Any = None

    @property
    def device(self) -> str:
        if self._device is None:
            import torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    def _load_local(self) -> None:
        """Load model locally with 4-bit quantization and optional CPU offload."""
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

            logger.info(f"Loading {self.model_name} (VRAM={self.vram_gb:.1f}GB, offload={self.use_cpu_offload})")

            # 4-bit quantization
            quant_config = None
            try:
                from transformers import BitsAndBytesConfig
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            except ImportError:
                logger.warning("bitsandbytes not available, using float16")

            load_kwargs: dict[str, Any] = {"torch_dtype": torch.float16}
            if quant_config:
                load_kwargs["quantization_config"] = quant_config

            # CPU offload for low VRAM — keeps GPU free for compute
            if self.use_cpu_offload and torch.cuda.is_available():
                vram_limit = max(2.0, self.vram_gb - 0.5)  # leave 0.5GB headroom
                load_kwargs["device_map"] = "auto"
                load_kwargs["max_memory"] = {
                    0: f"{vram_limit:.1f}GiB",
                    "cpu": "10GiB",
                }
                logger.info(f"CPU offload: GPU limit {vram_limit:.1f}GB, CPU 10GB")
            elif torch.cuda.is_available():
                load_kwargs["device_map"] = "auto"
            else:
                # Full CPU inference — very slow but functional
                logger.warning("No CUDA — loading on CPU. Expect very slow inference.")
                load_kwargs["device_map"] = "cpu"
                load_kwargs["torch_dtype"] = torch.float32

            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name, **load_kwargs
            )
            self._processor = AutoProcessor.from_pretrained(self.model_name)
            logger.info("Qwen2.5-VL loaded successfully")
        except ImportError as exc:
            raise RuntimeError(
                "Qwen2.5-VL dependencies missing. Install: "
                "pip install transformers torch accelerate bitsandbytes qwen-vl-utils"
            ) from exc

    def _build_messages(
        self,
        question: str,
        image: Any | None = None,
        context: str | None = None,
    ) -> list[dict]:
        """Build chat messages for Qwen2.5-VL."""
        content: list[dict] = []
        if image is not None:
            # Convert PIL Image to base64 for message
            import base64
            import io
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })

        prompt = question
        if context:
            prompt = (
                f"Based on the following evidence:\n{context}\n\n"
                f"Answer the question: {question}\n"
                f"Provide a concise, evidence-based answer with citations."
            )
        content.append({"type": "text", "text": prompt})

        return [{"role": "user", "content": content}]

    def generate(
        self,
        question: str,
        image: Any | None = None,
        context: str | None = None,
        max_new_tokens: int = 512,
    ) -> str:
        """Generate answer from question + optional image + optional context.

        Args:
            question: user query
            image: optional PIL.Image
            context: optional retrieved evidence text
            max_new_tokens: max generation length

        Returns:
            Generated answer string
        """
        if self.use_api:
            return self._generate_api(question, image, context, max_new_tokens)
        return self._generate_local(question, image, context, max_new_tokens)

    def _generate_local(
        self,
        question: str,
        image: Any | None,
        context: str | None,
        max_new_tokens: int,
    ) -> str:
        """Generate using local model."""
        import torch

        self._load_local()
        messages = self._build_messages(question, image, context)

        text_input = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Build inputs depending on whether we have an image
        if image is not None:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self._processor(
                text=[text_input],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self._model.device)
        else:
            inputs = self._processor(
                text=[text_input], padding=True, return_tensors="pt"
            ).to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
        # Trim input tokens
        generated = output_ids[:, inputs.input_ids.shape[1]:]
        return self._processor.decode(generated[0], skip_special_tokens=True).strip()

    def _generate_api(
        self,
        question: str,
        image: Any | None,
        context: str | None,
        max_new_tokens: int,
    ) -> str:
        """Generate using API endpoint (DashScope / OpenAI-compatible)."""
        import json
        import urllib.request

        messages = self._build_messages(question, image, context)
        payload = {
            "model": self.model_name.split("/")[-1],
            "messages": messages,
            "max_tokens": max_new_tokens,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        req = urllib.request.Request(
            f"{self.api_base}/chat/completions",
            data=json.dumps(payload).encode(),
            headers=headers,
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read())
            return result["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            logger.error(f"API generation failed: {exc}")
            return f"[Generation failed: {exc}]"

    def ground_region(
        self,
        question: str,
        image: Any,
    ) -> dict[str, Any]:
        """Visual grounding: find region of interest in image.

        Asks Qwen2.5-VL to locate the relevant region for the question.

        Args:
            question: what to look for
            image: PIL.Image

        Returns:
            dict with keys: bbox (list[float]), confidence (float), description (str)
        """
        prompt = (
            f"Given the medical image and the question: '{question}'\n"
            f"Identify the most relevant region. Return a JSON object with:\n"
            f'- "bbox": [x1, y1, x2, y2] as relative coordinates (0.0 to 1.0)\n'
            f'- "confidence": a float between 0 and 1\n'
            f'- "description": a brief description of what is in the region\n'
            f"Return ONLY the JSON object, nothing else."
        )
        raw = self.generate(prompt, image=image, max_new_tokens=256)

        # Parse JSON from response
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[^}]+\}', raw, re.DOTALL)
            if json_match:
                import json
                result = json.loads(json_match.group())
                bbox = result.get("bbox", [0.0, 0.0, 1.0, 1.0])
                # Validate bbox
                if (
                    isinstance(bbox, list)
                    and len(bbox) == 4
                    and all(0.0 <= v <= 1.0 for v in bbox)
                    and bbox[0] < bbox[2]
                    and bbox[1] < bbox[3]
                ):
                    return {
                        "bbox": bbox,
                        "confidence": float(result.get("confidence", 0.5)),
                        "description": str(result.get("description", "")),
                    }
        except Exception:
            pass

        # Fallback: return full image region
        logger.warning("Could not parse grounding result, using full image")
        return {
            "bbox": [0.0, 0.0, 1.0, 1.0],
            "confidence": 0.3,
            "description": raw[:200] if raw else "full image region (grounding fallback)",
        }
