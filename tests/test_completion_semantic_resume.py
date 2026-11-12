import json

import pytest

from conceptgraph.completion.semantic_evaluation_resume import load_prefix


def test_resume_rejects_partial_line_and_different_order_before_appending(tmp_path):
    path=tmp_path/'raw_records.jsonl'
    rows=[dict(sample_id='a',task='query_parse',scene_id='s',log_token='l'),
          dict(sample_id='b',task='query_parse',scene_id='s',log_token='l')]
    record=dict(rows[0],envelope=dict(raw_output='{}'))
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='incomplete raw line'):
        load_prefix(path,rows)
    path.write_text(json.dumps(record)+'\n')
    assert load_prefix(path,rows)==[record]
    before=path.read_bytes()
    with pytest.raises(ValueError,match='request prefix differs'):
        load_prefix(path,list(reversed(rows)))
    assert path.read_bytes()==before
