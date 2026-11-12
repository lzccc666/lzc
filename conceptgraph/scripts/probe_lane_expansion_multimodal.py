"""Actual processor/generation tensor evidence on a frozen real object crop."""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from conceptgraph.scripts.prepare_lane_expansion import DEFAULT_ROOT
from conceptgraph.scripts.run_incremental_semantic_navigation import DEFAULT_CONFIG, write_json
from conceptgraph.vlm.qwen_semantic_delegate import QwenDelegateConfig, QwenSemanticDelegate
from conceptgraph.vlm.qwen_vl_chat import QwenVLChat


class ProcessorRecorder:
    def __init__(self, wrapped, records):
        self.wrapped, self.records = wrapped, records
    def __getattr__(self, name):
        return getattr(self.wrapped, name)
    def __call__(self, *args, **kwargs):
        result = self.wrapped(*args, **kwargs)
        self.records.append({'stage': 'processor', 'real_image_count': len(kwargs.get('images') or []), 'image_sizes': [list(i.size) for i in kwargs.get('images') or []], 'tensor_shapes': {k: list(v.shape) for k, v in result.items() if hasattr(v, 'shape')}, 'image_grid_thw': result['image_grid_thw'].tolist() if 'image_grid_thw' in result else None})
        return result


class ModelRecorder:
    def __init__(self, wrapped, records):
        self.wrapped, self.records = wrapped, records
    def __getattr__(self, name):
        return getattr(self.wrapped, name)
    def generate(self, *args, **kwargs):
        self.records.append({'stage': 'actual_model_generate', 'tensor_devices': {k: str(v.device) for k, v in kwargs.items() if torch.is_tensor(v)}, 'pixel_values_shape': list(kwargs['pixel_values'].shape) if 'pixel_values' in kwargs else None, 'pixel_values_dtype': str(kwargs['pixel_values'].dtype) if 'pixel_values' in kwargs else None, 'do_sample': kwargs['do_sample'], 'max_new_tokens': kwargs['max_new_tokens']})
        return self.wrapped.generate(*args, **kwargs)


def main():
    root = DEFAULT_ROOT
    destination = root / 'runtime_probes/multimodal_processor_report.json'
    if destination.exists():
        raise FileExistsError(destination)
    nodes = json.loads((root/'frame0_inputs/scene-0004/nodes.json').read_text())
    node = next(n for n in nodes if n['metadata'].get('crop_paths'))
    crop = Path(node['metadata']['crop_paths'][0])
    settings = OmegaConf.load(DEFAULT_CONFIG)
    results = []
    backend = None
    for variant in ('adapter', 'base'):
        backend = None
        gc.collect()
        torch.cuda.empty_cache()
        records = []
        backend = QwenVLChat(str(settings.qwen_online.model_name_or_path), adapter_path=str(settings.qwen_online.adapter_path) if variant == 'adapter' else None, torch_dtype=getattr(torch, str(settings.qwen_online.torch_dtype)), max_image_side=int(settings.qwen_online.max_image_side))
        backend.processor = ProcessorRecorder(backend.processor, records)
        backend.model = ModelRecorder(backend.model, records)
        config = QwenDelegateConfig(enabled=True, model_name_or_path=str(settings.qwen_online.model_name_or_path), adapter_path=str(settings.qwen_online.adapter_path) if variant == 'adapter' else None, use_qwen_object_attributes=True, prompt_profile=str(settings.qwen_online.prompt_profile), timeout_s=30.0, max_new_tokens=192, log_dir=root/'runtime_probes'/variant)
        delegate = QwenSemanticDelegate(config, backend=backend)
        try:
            result = delegate.describe_object([str(crop)], {'object_tag': node['category'], 'caption': node['category'], 'bbox_extent': node['extent']}, object_version='frozen-real-crop-probe-v1')
            cuda_pixels = any(r['stage'] == 'actual_model_generate' and r['tensor_devices'].get('pixel_values', '').startswith('cuda') for r in records)
            passed = delegate.backend_call_count == 1 and cuda_pixels and any(r['stage'] == 'processor' and r['real_image_count'] == 1 for r in records)
            results.append({'variant': variant, 'passed': passed, 'schema_success': bool(result['used_qwen']) and bool(result['output_schema_valid']), 'backend_calls': delegate.backend_call_count, 'processor_and_generate_records': records, 'strict_result': result, 'crop_path': str(crop), 'crop_sha256': hashlib.sha256(crop.read_bytes()).hexdigest(), 'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(), 'counts_as_online_application': False})
        finally:
            delegate.close()
            delegate = None
            backend = None
            gc.collect()
            torch.cuda.empty_cache()
    write_json(destination, {'passed': all(r['passed'] for r in results), 'results': results, 'scope': 'Actual real-crop processor/CUDA tensor evidence only; schema success is separate and application evidence must come from real replay events. Not synthetic, not cached, not counted as online application.'})
    if not all(r['passed'] for r in results):
        raise RuntimeError('actual multimodal processor probe failed')


if __name__ == '__main__':
    main()
