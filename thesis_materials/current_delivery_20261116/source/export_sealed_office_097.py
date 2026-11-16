"""纠正表格比较条件说明；沿用冻结数字，不改方法、模型或原081产物。"""
import hashlib
import json
from pathlib import Path
import zipfile

import export_sealed_office_081 as original

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    corrected = ROOT/'tables/sealed_office_097'
    predecessor = ROOT/'tables/sealed_office_081/封存公开数据实验表.docx'
    before = sha(predecessor)
    old_text = '每方法使用相同候选、指令和评价分母；这是模板查询，不能称人工自然指令。'
    new_text = ('五组Qwen模型使用相同候选、提示与推理预算；全部方法使用相同指令与评价分母。'
                '当前帧图像裁剪和稀疏网格使用各自冻结的表示及候选，不声称它们与对象图候选完全相同。'
                '这是模板查询，不能称人工自然指令。')
    render = original.paragraph
    original.paragraph = lambda value: render(str(value).replace(old_text, new_text))
    original.OUT = corrected
    original.main()
    with zipfile.ZipFile(predecessor) as z:
        old_xml = z.read('word/document.xml').decode()
    with zipfile.ZipFile(corrected/'封存公开数据实验表.docx') as z:
        new_xml = z.read('word/document.xml').decode()
    assert old_xml.count(old_text) == 2
    assert old_text not in new_xml and new_xml.count(new_text) == 2
    assert new_xml == old_xml.replace(old_text, new_text)
    assert sha(predecessor) == before
    with (corrected/'correction_receipt_097.json').open('x') as stream:
        json.dump(dict(source_sha256=sha(Path(__file__)), predecessor_sha256=before,
                       predecessor_unchanged=True, corrected_method_input_note_occurrences=2,
                       document_XML_identical_except_two_method_input_notes=True,
                       all_metric_values_unchanged=True, new_GPU_calls=0,
                       new_CARLA_calls=0, PDF_visual_QA_pending=True, goal_complete=False),
                  stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
