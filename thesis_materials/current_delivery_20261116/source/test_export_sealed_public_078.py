"""图表统计的边界测试；以下数值只作软件测试，不进入实验表或图。"""
from collections import Counter
import unittest

import numpy as np

from export_sealed_public_078 import LABELS, metric, paired_retrieval, totals


class StatisticsTests(unittest.TestCase):
    def test_empty_denominator_stays_unknown(self):
        for name in ['object_recall', 'object_precision', 'IDF1', 'center_error_m',
                     'relation_f1', 'duplicate_fraction']:
            self.assertIsNone(metric({}, name))

    def test_recall_uses_counts_not_mean_of_scene_rates(self):
        rows = [dict(totals=dict(truth=1, true_positive=1), identity={}),
                dict(totals=dict(truth=100, true_positive=10), identity={})]
        self.assertAlmostEqual(metric(totals(rows), 'object_recall'), 11/101)
        self.assertNotAlmostEqual(metric(totals(rows), 'object_recall'), .55)

    def test_unresolved_edges_penalize_f1(self):
        resolved = Counter(relation_true_positive=2, relation_predicted_resolved=2, relation_truth=2)
        self.assertEqual(metric(resolved, 'relation_f1'), 1.)
        resolved['relation_predicted_unresolved'] = 4
        self.assertEqual(metric(resolved, 'relation_f1'), .5)

    def test_identity_false_negative_and_false_positive_both_count(self):
        self.assertEqual(metric(dict(IDTP=3, IDFN=2, IDFP=4), 'IDF1'), .5)

    def test_center_error_is_match_weighted(self):
        rows = [dict(totals=dict(center_error_sum_m=2., true_positive=1), identity={}),
                dict(totals=dict(center_error_sum_m=3., true_positive=9), identity={})]
        self.assertEqual(metric(totals(rows), 'center_error_m'), .5)

    def fixtures(self):
        records = []
        for cohort in ['distribution_preserving', 'challenge']:
            for method in ['base', 'real_graph_seed17']:
                for log, positive in [('a', True), ('b', False)]:
                    records.append(dict(cohort=cohort, method=method, query_id=cohort+log,
                        log=log, positive=positive, correct=(method == 'real_graph_seed17'),
                        no_match_false_selection=(not positive and method == 'base')))
        return records

    def test_paired_positive_and_no_match_have_opposite_desired_signs(self):
        draws = np.asarray([[0, 0], [1, 1], [0, 1], [1, 0]])
        result = paired_retrieval(self.fixtures(), ['base', 'real_graph_seed17'], ['a', 'b'], draws)
        self.assertEqual(len(result), 6)
        positive = next(r for r in result if r['metric'] == 'positive_target_correct')
        negative = next(r for r in result if r['metric'] == 'no_match_false_selection')
        self.assertEqual(positive['method_minus_base'], 1.)
        self.assertEqual(positive['ci95_low'], 1.)
        self.assertEqual(negative['method_minus_base'], -1.)
        self.assertEqual(negative['desirable_direction'], 'negative')
        self.assertEqual(positive['valid_bootstrap_draws'], 3)

    def test_paired_methods_must_have_identical_query_ids(self):
        data = self.fixtures()
        data[-1]['query_id'] = 'different'
        with self.assertRaises(AssertionError):
            paired_retrieval(data, ['base', 'real_graph_seed17'], ['a', 'b'], np.asarray([[0, 1]]))

    def test_actual_controlled_baselines_have_chinese_labels(self):
        self.assertEqual(LABELS['clip_current_crop'], '当前帧CLIP')
        self.assertEqual(LABELS['controlled_sparse_clip_grid'], '稀疏语义格')


if __name__ == '__main__':
    unittest.main()
