"""按事前图形契约读取冻结开发统计，导出论文矢量图和源数据。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib import font_manager
import numpy as np

from conceptgraph.completion.stage9 import OUTPUT, digest, dump


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    dest=OUTPUT/'thesis'/args.run_id
    contract=json.loads((dest/'figure_contract.json').read_text())
    if (dest/'manifest.json').exists():
        raise ValueError('finished figure package cannot be overwritten')
    include_second=any(f['id']=='6-5' for f in contract['figures'])
    if include_second and not (OUTPUT/'statistics/fair_three_task_full_validation_001/report.json').exists():
        raise ValueError('complete five-model statistics required for planned figure 6-5')
    font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    font_manager.fontManager.addfont(str(font))
    font_manager.fontManager.addfont(str(font.with_name('NotoSansCJK-Bold.ttc')))
    warnings.filterwarnings('error',message='Glyph .* missing from font',category=UserWarning)
    resolved=Path(font_manager.findfont(FontProperties(family=FontProperties(fname=font).get_name())))
    if resolved.resolve()!=font.resolve():
        raise RuntimeError('CJK font resolved to a different fallback font')
    plt.rcParams.update({'font.family':FontProperties(fname=font).get_name(),
        'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7.5,
        'ytick.labelsize':8,'svg.fonttype':'none','pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
        'legend.frameon':False,'lines.linewidth':1.2,'savefig.facecolor':'white'})
    sources=[]
    qa=[]
    source_rows=[]
    captions=[]
    def read(path):
        full=OUTPUT/path
        sources.append(dict(path=path,sha256=digest(full)))
        return json.loads(full.read_text())
    def save(fig,number,caption):
        fig.canvas.draw()
        renderer=fig.canvas.get_renderer()
        bounds=fig.bbox
        out=[]
        for obj in fig.findobj(matplotlib.text.Text):
            if obj.get_visible() and obj.get_text().strip():
                b=obj.get_window_extent(renderer)
                if b.x0 < bounds.x0-2 or b.y0 < bounds.y0-2 or b.x1 > bounds.x1+2 or b.y1 > bounds.y1+2:
                    out.append(obj.get_text())
        stem=dest/f'图{number}'
        fig.savefig(stem.with_suffix('.svg'))
        fig.savefig(stem.with_suffix('.pdf'))
        fig.savefig(stem.with_suffix('.png'),dpi=300)
        fig.savefig(stem.with_suffix('.tiff'),dpi=600,pil_kwargs={'compression':'tiff_lzw'})
        svg=stem.with_suffix('.svg').read_text()
        qa.append(dict(figure=number,width_mm=float(fig.get_figwidth()*25.4),
            height_mm=float(fig.get_figheight()*25.4),text_outside_canvas=out,
            svg_editable_text='<text ' in svg,visual_review='待逐图检查PNG',
            minimum_font_pt=7.5,formats=['svg','pdf','png','tiff']))
        captions.append(f'图{number} {caption}')
        plt.close(fig)
    blue,orange,gray='#286995','#C1742F','#60717C'
    mapping=read('statistics/robust_motion_001/report.json')
    fig,axes=plt.subplots(1,3,figsize=(183/25.4,83/25.4))
    fig.subplots_adjust(left=.18,right=.97,bottom=.31,top=.80,wspace=.7)
    methods=[('one_to_one_depth_quality','深度支持簇',blue,'o'),
             ('one_to_one_depth_robust_motion','深度＋稳健运动',orange,'s')]
    metrics=[('recall','召回率差（百分点）',100),('center_error_m_mean','中心误差差（m）',1),
             ('identity_switch_rate','身份切换率差（百分点）',100)]
    for a,(metric,label,scale) in enumerate(metrics):
        ax=axes[a]
        for j,(method,name,color,marker) in enumerate(methods):
            row=next(v for v in mapping['paired_log_cluster_bootstrap'] if v['method']==method and v['metric']==metric)
            value=row['difference']*scale
            lo,hi=np.array(row['ci95'])*scale
            ax.errorbar(value,j,xerr=[[value-lo],[hi-value]],fmt=marker,color=color,capsize=3)
            source_rows.append(dict(figure='6-1',panel=metric,method=method,value=value,low=lo,high=hi,
                numerator='',denominator=5723,unit='百分点' if scale==100 else 'm'))
        ax.axvline(0,color=gray,lw=.7,ls='--')
        ax.set(ylim=(-.65,1.65),yticks=[0,1],yticklabels=[m[1] for m in methods] if a==0 else ['', ''],xlabel=label)
        ax.invert_yaxis()
        ax.set_title(f'{chr(97+a)}  '+['覆盖提高','定位误差下降','身份改善不确定'][a],loc='left')
        ax.spines['left'].set_visible(False)
    fig.suptitle('与原融合配对比较：定位改善不等于身份改善',y=.98,fontsize=10)
    fig.text(.04,.10,'38场景，11个开发log，1508帧；误差线：4000次配对log聚类bootstrap的95%区间。',fontsize=7.5)
    fig.text(.04,.04,'召回率基于5723次可见GT对象观测；中心误差仅对已匹配对象计算，匹配组成可能不同。',fontsize=7.5)
    save(fig,'6-1','对象建图相对原融合的配对差。正召回差、负中心误差差表示改善；身份切换率差的区间跨零。所有条件使用同一修正前端。')

    navigation=read('statistics/interactive_rule_full_002/report.json')
    outcomes=navigation['summary']['by_kind']['reachable_positive']['four_outcomes']
    fields=[('route_generated','生成目标路线'),('selected_object_evaluated_correct','选对对象'),
        ('stable_correct_lock','稳定正确锁定'),('actual_full_perception_arrival','实际正确到达')]
    fig,ax=plt.subplots(figsize=(183/25.4,88/25.4))
    fig.subplots_adjust(left=.22,right=.90,bottom=.28,top=.82)
    for j,(key,label) in enumerate(fields):
        row=outcomes[key]
        value=row['estimate']*100
        lo,hi=np.array(row['ci95'])*100
        color=orange if key=='actual_full_perception_arrival' else blue
        ax.errorbar(value,j,xerr=[[value-lo],[hi-value]],fmt='s' if j==3 else 'o',color=color,capsize=4)
        ax.text(105,j,f"{int(row['numerator'])}/90",va='center',ha='left',fontsize=8)
        source_rows.append(dict(figure='6-2',panel=key,method='真实规则感知闭环',value=value,low=lo,high=hi,
            numerator=int(row['numerator']),denominator=90,unit='%'))
    ax.set(yticks=range(4),yticklabels=[r[1] for r in fields],ylim=(3.6,-.6),xlim=(0,114),
        xticks=[0,25,50,75,100],xlabel='占全部可达正任务试验（%）')
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x',color='#D5DDE1',lw=.5)
    ax.set_axisbelow(True)
    fig.suptitle('四项结果分别计数：正确选中目标仍可能未到达',y=.97,fontsize=10)
    fig.text(.035,.11,'30个可达正任务×3种子＝90次；10配置来自3类共享走廊；误差线为配置聚类95%区间。',fontsize=7.5)
    fig.text(.035,.05,'21次到达覆盖13个不同正任务、8个配置；四项不是严格时序漏斗。其余60次负任务另表报告。',fontsize=7.5)
    save(fig,'6-2','完整规则感知闭环的四项独立结果。未到达的69次正任务保留在分母；100%经验聚类区间不代表未来总体一定成功。')

    semantic=read('main_experiments/fair_calibrated_full_validation_001/report.json')
    fig,axes=plt.subplots(1,2,figsize=(183/25.4,92/25.4))
    fig.subplots_adjust(left=.18,right=.96,bottom=.27,top=.81,wspace=.40)
    models=[('base_fullschema','原始Qwen'),('old_adapter_fullschema','旧Adapter'),
        ('new_adapter_seed17','首版新Adapter·17'),('new_adapter_seed29','首版新Adapter·29'),
        ('new_adapter_seed43','首版新Adapter·43')]
    for a,(task,title) in enumerate([('object_attributes','真实crop类别（n＝243）'),('query_parse','完整解析（n＝180）')]):
        ax=axes[a]
        for j,(key,name) in enumerate(models):
            row=semantic['variants'][key]['metrics'][task]
            value=row['accuracy']*100
            ax.plot(value,j,'s' if j>=2 else 'o',color=orange if j>=2 else blue,ms=5)
            ax.annotate(f"{value:.1f}",(value,j),xytext=(5,0),textcoords='offset points',va='center',fontsize=7.5)
            source_rows.append(dict(figure='6-3',panel=task,method=key,value=value,low='',high='',
                numerator=row['semantic_correct'],denominator=row['requests'],unit='%'))
        ax.set(xlim=(-3,108),xticks=[0,25,50,75,100],ylim=(4.6,-.6),yticks=range(5),
            yticklabels=[v[1] for v in models] if a==0 else ['']*5,xlabel='固定验证集准确率（%）')
        ax.set_title(f'{chr(97+a)}  {title}',loc='left')
        ax.spines['left'].set_visible(False)
    fig.suptitle('首版训练负结果：统一v3提示下未获得全面改善',y=.98,fontsize=10)
    fig.text(.035,.11,'相同输入、提示、schema和预算；三个新种子全部显示。固定验证集描述值，未估计置信区间。',fontsize=7.5)
    fig.text(.035,.05,'类别不等于可靠颜色或运动状态；不代表三任务第二版Adapter的结果；回退不获得准确率信用。',fontsize=7.5)
    save(fig,'6-3','首版模型公平比较。点表示固定请求集合上的准确率；未以调用数或crop数推断独立样本显著性，三种子不合并挑优。')

    retrieval=read('statistics/retrieval_baselines_dev_full_001/report.json')
    fig,axes=plt.subplots(1,2,figsize=(183/25.4,90/25.4))
    fig.subplots_adjust(left=.21,right=.96,bottom=.30,top=.81,wspace=.5)
    names=[('category_nearest_current_detection','类别最近检测'),('clip_current_crop','当前crop CLIP'),
        ('vlmaps_style_sparse_clip_grid','稀疏CLIP网格适配'),('existing_object_joint_rule','联合规则选择')]
    for a,(metric,title) in enumerate([('positive_selection','正目标选对（n＝733）'),('no_match_false_selection','无匹配误选（n＝382）')]):
        ax=axes[a]
        for j,(key,label) in enumerate(names):
            row=retrieval['methods'][key][metric]
            value=row['estimate']*100
            lo,hi=np.array(row['ci95'])*100
            ax.errorbar(value,j,xerr=[[value-lo],[hi-value]],fmt='s' if j==3 else 'o',
                color=orange if j==3 else blue,capsize=3)
            source_rows.append(dict(figure='6-4',panel=metric,method=key,value=value,low=lo,high=hi,
                numerator=row['numerator'],denominator=row['denominator'],unit='%'))
        ax.set(xlim=(-3,100),ylim=(3.6,-.6),yticks=range(4),
            yticklabels=[v[1] for v in names] if a==0 else ['']*4,xlabel='比例（%）')
        ax.set_title(f'{chr(97+a)}  {title}',loc='left')
        ax.spines['left'].set_visible(False)
    fig.suptitle('检索取舍：正目标覆盖与无匹配误选需要同时报告',y=.98,fontsize=10)
    fig.text(.035,.15,'38场景、11log、1115条配对查询；误差线为4000次配对log聚类bootstrap的95%区间。',fontsize=7.5)
    fig.text(.035,.09,'检测/网格与图节点的独立实例关联口径不同；未知关联作为正任务失败，不能将差异全归因于选择器。',fontsize=7.5)
    fig.text(.035,.03,'网格是VLMaps式简化适配（稀疏crop CLIP／二维格），不等同原方法；本图不评价路径、锁定或到达。',fontsize=7.5)
    save(fig,'6-4','开发检索方法的正目标与无匹配结果。全部方法读取当前及过去证据；未依据性能挑选查询或只报告正目标。')
    if include_second:
        second=read('statistics/fair_three_task_full_validation_001/report.json')
        fig,axes=plt.subplots(1,3,figsize=(183/25.4,100/25.4))
        fig.subplots_adjust(left=.20,right=.98,bottom=.31,top=.78,wspace=.42)
        choices=[('base_fullschema','原始Qwen'),('old_adapter_fullschema','旧Adapter'),
            ('new_adapter_seed17','第二版·17'),('new_adapter_seed29','第二版·29'),('new_adapter_seed43','第二版·43')]
        tasks=[('object_attributes','crop类别（243）'),('query_parse','完整解析（180）'),
            ('candidate_rerank','受控排序（180）')]
        for a,(task,title) in enumerate(tasks):
            ax=axes[a]
            info=second['tasks'][task]
            for j,(name,label) in enumerate(choices):
                row=info['models'][name]
                value=row['accuracy']*100
                lo,hi=np.array(row['ci95'])*100
                ax.errorbar(value,j,xerr=[[value-lo],[hi-value]],fmt='s' if j>=2 else 'o',
                    color=orange if j>=2 else blue,capsize=3)
                source_rows.append(dict(figure='6-5',panel=task,method=name,value=value,low=lo,high=hi,
                    numerator=row['correct'],denominator=row['requests'],unit='%'))
            ax.set(xlim=(-4,104),xticks=[0,50,100],ylim=(4.6,-.6),yticks=range(5),
                yticklabels=[v[1] for v in choices] if a==0 else ['']*5,xlabel='准确率（%）')
            ax.set_title(f'{chr(97+a)}  {title}',loc='left',fontsize=8.5)
            ax.spines['left'].set_visible(False)
        fig.suptitle('第二版三任务验证：保留全部种子及独立分母',y=.97,fontsize=10)
        fig.text(.03,.18,'解析与crop：仅2个验证log；排序：6个受控符号bank，去置换44种证据。',fontsize=7.5)
        fig.text(.03,.12,'误差线为4000次配对聚类95%区间；2个log的区间离散且极不稳定，不作充分泛化/显著性证据。',fontsize=7.5)
        fig.text(.03,.06,'种子17依训练侧规则事前选定；符号排序与类别准确率不能替代真实道路对象或实际到达表现。',fontsize=7.5)
        save(fig,'6-5','第二版三任务公平验证。每模型相同603个请求及0新增调用缓存复跑；三个种子分别呈现，类别、完整解析与受控排序采用不同证据及聚类单位。')
    with (dest/'source_data.csv').open('x') as f:
        writer=csv.DictWriter(f,fieldnames=['figure','panel','method','value','low','high','numerator','denominator','unit'])
        writer.writeheader()
        writer.writerows(source_rows)
    (dest/'图注与统计说明.md').write_text('\n\n'.join(captions)+'\n\n数据均为开发阶段；无假设检验、p值或多重比较显著性声明。\n')
    shutil.copy2(__file__,dest/'generator_source.py')
    dump(dest/'qa_automatic.json',dict(backend='python',figures=qa,
        passed=all(not q['text_outside_canvas'] and q['svg_editable_text'] for q in qa),visual_review_pending=True))
    files={p.name:digest(p) for p in dest.iterdir() if p.is_file()}
    dump(dest/'manifest.json',dict(contract=contract,sources=sources,files=files,
        source_script_sha256=digest(__file__),font_path=str(font),font_sha256=digest(font),
        independent_test=False,goal_complete=False))
    print(json.dumps(qa,ensure_ascii=False))


if __name__=='__main__':
    main()
