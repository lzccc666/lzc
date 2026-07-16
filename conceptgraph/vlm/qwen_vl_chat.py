from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Iterable, Optional

import torch
from PIL import Image


DEFAULT_QUERY = "Describe the central object in the image."

os.environ.pop("OMP_NUM_THREADS", None)


def _prefer_conda_runtime_libs() -> None:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return
    lib_dir = Path(conda_prefix) / "lib"
    libstdcxx = lib_dir / "libstdc++.so.6"
    if lib_dir.exists():
        current = os.environ.get("LD_LIBRARY_PATH", "")
        paths = [str(lib_dir)] + [item for item in current.split(":") if item and item != str(lib_dir)]
        os.environ["LD_LIBRARY_PATH"] = ":".join(paths)
    if libstdcxx.exists():
        ctypes.CDLL(str(libstdcxx), mode=ctypes.RTLD_GLOBAL)


_prefer_conda_runtime_libs()


class QwenVLChat:
    def __init__(self, model_name_or_path: Optional[str] = None, device: str = "cuda:0"):
        default_model_path = Path(__file__).resolve().parents[2] / "checkpoint" / "Qwen2-VL-2B-Instruct"
        self.model_name_or_path = model_name_or_path or os.getenv("QWEN_VL_CKPT_PATH", str(default_model_path))
        self.device = device
        self.history = None
        self._load_model()

    def _load_model(self) -> None:
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        torch_dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.processor = AutoProcessor.from_pretrained(self.model_name_or_path, trust_remote_code=True)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch_dtype,
            device_map="auto" if self.device.startswith("cuda") else None,
            trust_remote_code=True,
        ).eval()
        if not self.device.startswith("cuda"):
            self.model.to(self.device)

    def reset(self):
        self.history = None

    def _messages(self, query: str, images: Optional[Iterable[Image.Image]]):
        content = []
        if images is not None:
            for image in images:
                content.append({"type": "image", "image": image.convert("RGB")})
        content.append({"type": "text", "text": query})
        return [{"role": "user", "content": content}]

    def chat(self, query: str, images: Optional[Iterable[Image.Image]] = None) -> str:
        messages = self._messages(query, images)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs = [item["image"] for item in messages[0]["content"] if item["type"] == "image"]
        inputs = self.processor(
            text=[text],
            images=image_inputs if image_inputs else None,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=192)
        trimmed = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0].strip()

    def __call__(self, query: str = DEFAULT_QUERY, images: Optional[Iterable[Image.Image]] = None) -> str:
        return self.chat(query=query, images=images)


class QwenTextChat(QwenVLChat):
    def chat(self, query: str, images: Optional[Iterable[Image.Image]] = None) -> str:
        return super().chat(query=query, images=None)
