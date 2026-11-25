"""训练时提高最终编号/拒答值的损失权重；不改变任何输入、标签或评测。"""
import json
import torch

def weights_for(processor,full_text,row,input_ids,labels):
 tokenized=processor.tokenizer(full_text,return_offsets_mapping=True,add_special_tokens=False)
 if tokenized['input_ids']!=input_ids.tolist():raise ValueError('token offsets differ from actual processor tokens; refuse approximate weighting')
 response=json.dumps(row['response'],ensure_ascii=False,sort_keys=True,separators=(',',':'));begin=full_text.rfind(response)
 if begin<0:raise ValueError('assistant response span absent')
 weights=(labels!=-100).to(torch.float32);counts={}
 for key,factor in [('selected_node_id',16.),('no_match',4.)]:
  value=json.dumps(row['response'][key],ensure_ascii=False,separators=(',',':'));prefix='"'+key+'":';start=begin+response.index(prefix+value)+len(prefix);end=start+len(value);chosen=[i for i,(a,b) in enumerate(tokenized['offset_mapping']) if a<end and b>start and labels[i]!=-100]
  if not chosen:raise ValueError('decision value has no supervised token')
  for i in chosen:weights[i]=factor
  counts[key]=len(chosen)
 if not torch.all(weights[labels==-100]==0):raise ValueError('prompt acquired loss weight')
 return weights,counts

def weighted_causal_loss(model,batch):
 labels=batch['labels'];weights=batch['loss_weights'].reshape_as(labels);output=model(**{k:v for k,v in batch.items() if k not in {'labels','loss_weights'}});logits=output.logits[:,:-1,:].float();target=labels[:,1:];w=weights[:,1:];losses=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),target.reshape(-1),ignore_index=-100,reduction='none').reshape_as(target)
 return (losses*w).sum()/w.sum()
