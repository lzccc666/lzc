"""保护人工标签隔离及未知/无效输出的计分边界，不加载大模型。"""
import unittest

from human_crop_fair_065 import safe_input, score_one


class HumanCropBoundaryTests(unittest.TestCase):
    def reference(self, color='white', category='car'):
        return dict(sample_number=1, scene='example', crop_file='/example.png',
            crop_sha256='abc', predicted_category='truck', human_category=category,
            human_color=color, category_comparable=category not in {'other', 'unknown'},
            note='不可进入模型的备注', human_body_type='厢式货车')

    def prediction(self, colors=None, valid=True):
        return dict(sample_id='human-crop-001', schema_valid=valid,
            normalized_output=dict(colors=colors or ['white'], object_tag='car',
                                   motion_state='unknown', body_type='sedan'))

    def test_label_export_whitelist(self):
        row = safe_input(self.reference())
        self.assertEqual(set(row), {'sample_id', 'scene', 'task', 'crop_path', 'crop_sha256', 'metadata'})
        self.assertEqual(row['metadata'], {'detection_label': 'truck'})

    def test_exact_known_color(self):
        self.assertTrue(score_one(self.prediction(), self.reference())['color_correct'])

    def test_extra_color_cannot_receive_correct_credit(self):
        self.assertFalse(score_one(self.prediction(['white', 'black']), self.reference())['color_correct'])

    def test_schema_invalid_is_failure(self):
        result = score_one(self.prediction(valid=False), self.reference())
        self.assertFalse(result['color_correct'])
        self.assertFalse(result['category_correct'])

    def test_abstention_on_known_color_is_failure(self):
        self.assertFalse(score_one(self.prediction(['unknown']), self.reference())['color_correct'])

    def test_unknown_and_multiple_have_no_accuracy_credit(self):
        for color in ['unknown', 'multiple']:
            result = score_one(self.prediction(), self.reference(color=color))
            self.assertFalse(result['color_evaluable'])
            self.assertIsNone(result['color_correct'])

    def test_taxonomy_unresolved_is_not_relabelled(self):
        result = score_one(self.prediction(), self.reference(category='other'))
        self.assertFalse(result['category_evaluable'])
        self.assertIsNone(result['category_correct'])

    def test_fine_body_type_cannot_receive_credit(self):
        self.assertFalse(score_one(self.prediction(), self.reference())['fine_body_type_accuracy_credit'])


if __name__ == '__main__':
    unittest.main()
