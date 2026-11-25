"""有界开发pilot只保留每次验证的Adapter与处理文件引用，明确不声称可恢复optimizer。"""
import shutil
from pathlib import Path
from conceptgraph.completion.stage9 import dump,digest
R=Path(__file__).resolve().parents[1];BASE=Path('/root/autodl-tmp/lizichong/checkpoint/Qwen2-VL-2B-Instruct')
def save_checkpoint(path,model,processor,optimizer,scheduler,completed_steps,curve):
 if shutil.disk_usage(R).free<20*2**30 or sum(p.stat().st_size for p in R.rglob('*') if p.is_file() and not p.is_symlink())+5*2**20>10*2**30:raise OSError('bounded pilot checkpoint reserve reached; no old file deleted')
 path.mkdir(parents=True,exist_ok=False);model.save_pretrained(path);names=['preprocessor_config.json','tokenizer_config.json','tokenizer.json','special_tokens_map.json','added_tokens.json','merges.txt','vocab.json','chat_template.json'];dump(path/'processor_reference.json',dict(path=str(BASE),files_sha256={str(BASE/n):digest(BASE/n) for n in names if (BASE/n).exists()}));dump(path/'checkpoint_manifest.json',dict(completed_steps=completed_steps,adapter_retained=True,optimizer_retained=False,scheduler_retained=False,RNG_retained=False,resumable=False,reason='最多240步、三次Adapter的空间受限开发pilot；完整三种子正式训练的恢复状态不变。若中断只允许用新run-id从固定配置重跑。',source_sha256=digest(__file__)))
