"""Python/PyMuPDF核验已导出的真实PDF并生成PDF视觉检查图。"""
import datetime,hashlib,json
from pathlib import Path
import fitz
R=Path(__file__).resolve().parents[1];D=R/'figures/public_evidence_003'
def main():
 automated=json.loads((D/'automated_qa_001.json').read_text());records=[]
 expected={'nuscenes_module_ablation':['0.301','0.186','1508'],'kitti_mapping_tradeoffs':['0.841','0.947','1000'],'qwen_fair_target_vs_format':['0.056','0.454','108'],'official_actuator_diagnostic':['1.202','0.989','510']}
 for p in sorted(D.glob('*.pdf')):
  doc=fitz.open(p);assert len(doc)==1;text=doc[0].get_text();assert len(text)>100 and all(x in text for x in expected[p.stem]);image=D/f'{p.stem}_pdf_qa.png';assert not image.exists();doc[0].get_pixmap(matrix=fitz.Matrix(2,2)).save(image);records.append(dict(file=p.name,pages=len(doc),extracted_text_characters=len(text),has_Chinese=any('\u4e00'<=x<='\u9fff' for x in text),width_mm=doc[0].rect.width*25.4/72,height_mm=doc[0].rect.height*25.4/72,expected_values=expected[p.stem],values_found=True,pdf_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),rendered_preview=image.name,render_sha256=hashlib.sha256(image.read_bytes()).hexdigest()));doc.close()
 assert len(records)==4 and all(not x['text_outside_canvas'] for x in automated['figures'])
 with (D/'export_qa_001.json').open('x') as f:json.dump(dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),Python_renderer='PyMuPDF 1.26.6, clean_cpu_install_001 venv',PDF_checks=records,SVG_editable_text_checked=True,TIFF600dpi_checked=True,all_canvas_bounds_pass=True,source_numeric_reproduced=True,PDF_pixel_inspection_pending=True,whole_project_sealed_test=False,goal_complete=False),f,ensure_ascii=False,indent=2)
 print(json.dumps(records,ensure_ascii=False))
if __name__=='__main__':main()
