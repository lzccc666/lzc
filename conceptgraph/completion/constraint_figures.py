"""从已完成开发消融生成中文矢量图，保留失败和评价分母。"""
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
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
import numpy as np

from conceptgraph.completion.stage9 import OUTPUT, digest, dump


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    dest = OUTPUT/'thesis'/args.run_id
    contract = json.loads((dest/'figure_contract.json').read_text())
    if (dest/'manifest.json').exists() or {v['id'] for v in contract['figures']}!={'6-6','6-7'}:
        raise ValueError('requires new two-figure contract')
    font = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    for p in (font, font.with_name('NotoSansCJK-Bold.ttc')):
        font_manager.fontManager.addfont(str(p))
    family = FontProperties(fname=font).get_name()
    if Path(font_manager.findfont(FontProperties(family=family))).resolve()!=font.resolve():
        raise RuntimeError('CJK font fallback')
    plt.rcParams.update({'font.family':family,'font.size':8,'axes.titlesize':9,'axes.labelsize':8,
        'xtick.labelsize':7.5,'ytick.labelsize':8,'svg.fonttype':'none','pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
        'legend.frameon':False,'savefig.facecolor':'white'})
    warnings.filterwarnings('error', message='Glyph .* missing from font', category=UserWarning)
    sources, data, qa, captions = [], [], [], []
    def read(relative):
        path = OUTPUT/relative
        sources.append(dict(path=relative,sha256=digest(path)))
        return json.loads(path.read_text())
    def point(ax, row, y, color, marker):
        value = row['estimate']*100
        lo, hi = np.asarray(row['ci95'])*100
        ax.errorbar(value,y,xerr=[[value-lo],[hi-value]],color=color,fmt=marker,capsize=2.5,markersize=4)
    def save(fig, number, caption):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        outside = []
        for obj in fig.findobj(matplotlib.text.Text):
            if obj.get_visible() and obj.get_text().strip():
                b = obj.get_window_extent(renderer)
                if b.x0 < -2 or b.y0 < -2 or b.x1 > fig.bbox.x1+2 or b.y1 > fig.bbox.y1+2:
                    outside.append(obj.get_text())
        for suffix in ['svg','pdf','png','tiff']:
            options = dict(dpi=600,pil_kwargs={'compression':'tiff_lzw'}) if suffix=='tiff' else dict(dpi=300) if suffix=='png' else {}
            fig.savefig(dest/f'图{number}.{suffix}', **options)
        qa.append(dict(figure=number,text_outside_canvas=outside,minimum_font_pt=7.5,
            svg_editable_text='<text ' in (dest/f'图{number}.svg').read_text(),
            width_mm=fig.get_figwidth()*25.4,height_mm=fig.get_figheight()*25.4))
        captions.append(f'图{number} {caption}')
        plt.close(fig)

    blue, orange, gray = '#286995', '#C1742F', '#60717C'
    maps = read('statistics/map_constraints_dev_full_001/report.json')
    names = ['free_space','drivable_area','lane_area','lane_direction','lane_connection','lane_direction_connection']
    labels = ['自由空间','可行驶区域','车道区域','车道＋方向','车道＋连接','车道＋方向＋连接']
    fig, axes = plt.subplots(1,2,figsize=(183/25.4,116/25.4))
    fig.subplots_adjust(left=.20,right=.97,bottom=.27,top=.78,wspace=.34)
    metrics = [[('route_generated','路线覆盖',gray,'o'),('full_lane_legal_all_queries','完整车道合法',blue,'s')],
               [('direction_violation_among_routes','方向违规',orange,'o'),('connection_violation_among_routes','连接违规',blue,'s')]]
    for a, ax in enumerate(axes):
        for j, method in enumerate(names):
            for k, (metric, label, color, marker) in enumerate(metrics[a]):
                row = maps['conditions'][method][metric]
                point(ax,row,j+(k-.5)*.26,color,marker)
                data.append(dict(figure='6-6',panel=a,condition=method,metric=metric,
                    value=row['estimate']*100,low=row['ci95'][0]*100,high=row['ci95'][1]*100,
                    numerator=row['count'],denominator=row['denominator'],unit='%'))
        for _, label, color, marker in metrics[a]:
            ax.plot([],[],marker=marker,color=color,linestyle='none',label=label)
        ax.legend(loc='lower left',bbox_to_anchor=(0,1.02),fontsize=7.5,handletextpad=.3)
        ax.set(xlim=(-3,103),ylim=(5.6,-.6),yticks=range(6),
            yticklabels=labels if a==0 else ['']*6,xticks=[0,25,50,75,100],
            xlabel='占全部212条预选查询（%）' if a==0 else '占该条件已生成路线（%）')
        ax.spines['left'].set_visible(False)
        ax.grid(axis='x',color='#D5DDE1',lw=.5)
        ax.set_axisbelow(True)
        ax.set_title(['a  覆盖与完整车道合法','b  条件违规率（分母不同）'][a],loc='left',y=1.17)
    fig.suptitle('地图约束减少违规，但点路径仍不保证整车可执行',y=.98,fontsize=10)
    fig.text(.04,.15,'38场景、11log；212条语义查询含176个不同几何查询。误差线为log聚类95%区间。',fontsize=7.5)
    fig.text(.04,.095,'已生成路线分母依次为196、162、158、111、118、91；完整方向＋连接仍有3条连接违规。',fontsize=7.5)
    fig.text(.04,.04,'各条件矩形静态姿态通过均为0；GT给定对象参考点，未执行控制，也未评价正确锁定或到达。',fontsize=7.5)
    save(fig,'6-6','六种地图约束的开发离线诊断。左图所有失败保留在212分母；右图只针对已生成路线，不能把条件分母当成同一任务子集。4000次log聚类bootstrap的95%区间不作多重显著性筛选。完整车道合法不包含整车运动学；车道与可行驶多边形同时合法另见原始统计。')

    dynamic = read('statistics/mapping_modules_dev_full_001/report.json')
    fig, axes = plt.subplots(2,2,figsize=(183/25.4,126/25.4))
    fig.subplots_adjust(left=.26,right=.96,bottom=.22,top=.84,wspace=.34,hspace=.68)
    conditions = [('without_dynamic_maintenance','减动态维护',orange,'o'),('without_short_term_expiry','减短期过期',blue,'s')]
    metrics = [('recall','召回率差（百分点）',100),('precision','匹配精确率差（百分点）',100),
               ('center_error_m_mean','中心误差差（m）',1),('identity_switch_rate','身份切换率差（百分点）',100)]
    for a, (metric, label, scale) in enumerate(metrics):
        ax = axes.flat[a]
        for j, (condition, name, color, marker) in enumerate(conditions):
            row = next(r for r in dynamic['paired_differences'] if r['condition']==condition and r['metric']==metric)
            value = row['difference']*scale
            lo, hi = np.asarray(row['ci95'])*scale
            ax.errorbar(value,j,xerr=[[value-lo],[hi-value]],fmt=marker,color=color,capsize=3)
            data.append(dict(figure='6-7',panel=metric,condition=condition,metric=metric,
                value=value,low=lo,high=hi,numerator='',denominator='',unit='百分点' if scale==100 else 'm'))
        ax.axvline(0,color=gray,lw=.7,ls='--')
        low, high = ax.get_xlim()
        ax.set_xticks([v for v in ax.get_xticks() if low <= v <= high])
        ax.set(ylim=(1.65,-.65),yticks=[0,1],yticklabels=[c[1] for c in conditions] if a%2==0 else ['',''],xlabel=label)
        ax.set_title(chr(97+a)+'  '+['覆盖下降','精确率下降','定位误差上升','身份变化仍不确定'][a],loc='left')
        ax.spines['left'].set_visible(False)
    fig.suptitle('动态维护与短期过期的独立作用：相对完整配置的配对差',y=.98,fontsize=10)
    fig.text(.04,.12,'38场景、1508不同帧、11log；4000次配对log聚类bootstrap的95%区间。',fontsize=7.5)
    fig.text(.04,.075,'完整条件复用38份同配置记录，其余76次实际重建；不同指标分母见源CSV及完整统计。',fontsize=7.5)
    fig.text(.04,.03,'关闭动态模块后运动状态为unknown；召回下降时身份切换减少不能单独解释为关联改善。',fontsize=7.5)
    save(fig,'6-7','动态维护与短期过期消融相对完整配置的配对差。召回/精确率以全部官方对象观测/预测为分母，中心误差以匹配对象为分母，身份切换率以相邻已匹配转换为分母；各组组成可能变化。零线是完整配置；未证明真实车辆到达收益。')
    with (dest/'source_data.csv').open('x') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    (dest/'图注与统计说明.md').write_text('\n\n'.join(captions)+'\n')
    shutil.copy2(OUTPUT/'statistics/mapping_modules_dev_full_001/source_data_by_log.csv',
        dest/'source_data_dynamic_by_log.csv')
    dump(dest/'qa_automatic.json',dict(passed=all(not q['text_outside_canvas'] and q['svg_editable_text'] for q in qa),
        figures=qa,visual_qa_pending=True,goal_complete=False))
    shutil.copy2(__file__,dest/'generator_source.py')
    dump(dest/'manifest.json',dict(sources=sources,backend='Python/matplotlib',
        files={p.name:digest(p) for p in dest.iterdir() if p.is_file()},goal_complete=False))
    print(json.dumps(dict(figures=2,automatic_passed=all(not q['text_outside_canvas'] for q in qa)),ensure_ascii=False))


if __name__ == '__main__':
    main()
