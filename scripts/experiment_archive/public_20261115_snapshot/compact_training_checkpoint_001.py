"""未来检查点不复制固定基础processor；保留全部Adapter和可恢复训练状态。"""
from pathlib import Path
import torch
from conceptgraph.completion.stage9 import dump,digest
from scripts.train_qwen_lora_semantic_delegate_v2 import _rng_state
BASE=Path('/root/autodl-tmp/lizichong/checkpoint/Qwen2-VL-2B-Instruct')

def save_checkpoint(path,model,processor,optimizer,scheduler,completed_steps,curve):
 path.mkdir(parents=True,exist_ok=False);model.save_pretrained(path)
 # 所有调用者原本也从固定BASE加载processor，引用其准确指纹，不复制相同的17MiB文件。
 files=['preprocessor_config.json','tokenizer_config.json','tokenizer.json','special_tokens_map.json','added_tokens.json','merges.txt','vocab.json','chat_template.json']
 refs={str(BASE/name):digest(BASE/name) for name in files if (BASE/name).exists()}
 dump(path/'processor_reference.json',dict(path=str(BASE),files_sha256=refs,loading_command='AutoProcessor.from_pretrained(processor_reference.path, local_files_only=True)',copied_processor=False))
 torch.save(dict(completed_steps=completed_steps,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),rng_state=_rng_state(),curve=curve),path/'trainer_state.pt')
 dump(path/'checkpoint_manifest.json',dict(schema_version='qwen-recoverable-processor-reference-v1',completed_steps=completed_steps,recoverable=True,contains_optimizer=True,contains_scheduler=True,contains_rng_state=True,processor_reference_sha256=digest(path/'processor_reference.json'),save_source_sha256=digest(__file__)))
