#!/usr/bin/env bash
set -euo pipefail
draft_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
draft_python="${draft_dir}/.venv/bin/python"
if [[ ! -x "${draft_python}" ]]; then
  python3 -m venv --system-site-packages "${draft_dir}/.venv"
  "${draft_python}" -m pip install -r "${draft_dir}/scripts/requirements.txt"
fi
mkdir -p "${draft_dir}/qa" "${draft_dir}/figures" "${draft_dir}/figure_data"
"${draft_python}" "${draft_dir}/scripts/build_references.py"
"${draft_python}" "${draft_dir}/scripts/build_literature_report.py"
if [[ "${1:-}" == "--extract-evidence" ]]; then
  "${draft_python}" "${draft_dir}/scripts/build_evidence_figures.py"
fi
for pass in 1 2 3 4; do
  "${draft_python}" "${draft_dir}/scripts/build_word.py"
  libreoffice "-env:UserInstallation=file://${draft_dir}/qa/lo_rebuild_profile" --headless --convert-to pdf --outdir "${draft_dir}" "${draft_dir}/硕士论文初稿.docx" > "${draft_dir}/qa/rebuild_conversion_${pass}.log" 2>&1
  if ! grep -q 'using filter' "${draft_dir}/qa/rebuild_conversion_${pass}.log"; then
    cat "${draft_dir}/qa/rebuild_conversion_${pass}.log"
    exit 1
  fi
  "${draft_python}" "${draft_dir}/scripts/qa_document.py"
  if "${draft_python}" -c 'import json,sys; j=json.load(open(sys.argv[1]));sys.exit(j["page_map_changed"] or bool(j["missing_headings"]))' "${draft_dir}/qa/page_inspection.json"; then
    break
  fi
done
"${draft_python}" "${draft_dir}/scripts/audit_evidence.py"
"${draft_python}" "${draft_dir}/scripts/finish_delivery.py"
printf '%s\n' 'Word和PDF已重新生成；修改内容后仍须重新进行逐页视觉检查。'
