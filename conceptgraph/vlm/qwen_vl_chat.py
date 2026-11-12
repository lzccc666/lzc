from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Iterable, Optional

import torch
from PIL import Image
from conceptgraph.vlm.image_input import prepare_qwen_image


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
    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        device: str = "cuda:0",
        *,
        adapter_path: str | Path | None = None,
        torch_dtype: torch.dtype | None = None,
        max_image_side: int | None = None,
    ):
        default_model_path = Path(__file__).resolve().parents[2] / "checkpoint" / "Qwen2-VL-2B-Instruct"
        self.model_name_or_path = model_name_or_path or os.getenv("QWEN_VL_CKPT_PATH", str(default_model_path))
        self.adapter_path = Path(adapter_path) if adapter_path is not None else None
        self.device = device
        self.torch_dtype = torch_dtype
        self.max_image_side = max_image_side
        if self.max_image_side is not None and self.max_image_side <= 0:
            raise ValueError("max_image_side must be positive")
        self.history = None
        self._load_model()

    def _load_model(self) -> None:
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        torch_dtype = self.torch_dtype
        if torch_dtype is None:
            torch_dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.processor = AutoProcessor.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch_dtype,
            device_map="auto" if self.device.startswith("cuda") else None,
            trust_remote_code=True,
            local_files_only=True,
        ).eval()
        if self.adapter_path is not None:
            adapter_config = self.adapter_path / "adapter_config.json"
            if not adapter_config.exists():
                raise FileNotFoundError(f"LoRA adapter config not found: {adapter_config}")
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, str(self.adapter_path)).eval()
        if not self.device.startswith("cuda"):
            self.model.to(self.device)

    def reset(self):
        self.history = None

    def _messages(self, query: str, images: Optional[Iterable[Image.Image]]):
        content = []
        if images is not None:
            for image in images:
                rgb = prepare_qwen_image(image, self.max_image_side)
                content.append({"type": "image", "image": rgb})
        content.append({"type": "text", "text": query})
        return [{"role": "user", "content": content}]

    def chat(
        self,
        query: str,
        images: Optional[Iterable[Image.Image]] = None,
        generation_config: Optional[dict] = None,
    ) -> str:
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
            parameters = {
                "do_sample": False,
                "max_new_tokens": 192,
                "temperature": None,
                "top_k": None,
                "top_p": None,
            }
            if generation_config:
                parameters.update(generation_config)
            parameters["do_sample"] = False
            parameters.setdefault("pad_token_id", self.processor.tokenizer.eos_token_id)
            generated_ids = self.model.generate(**inputs, **parameters)
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

    def __call__(
        self,
        query: str = DEFAULT_QUERY,
        images: Optional[Iterable[Image.Image]] = None,
        generation_config: Optional[dict] = None,
    ) -> str:
        return self.chat(query=query, images=images, generation_config=generation_config)


class QwenTextChat(QwenVLChat):
    def chat(
        self,
        query: str,
        images: Optional[Iterable[Image.Image]] = None,
        generation_config: Optional[dict] = None,
    ) -> str:
        return super().chat(query=query, images=None, generation_config=generation_config)
