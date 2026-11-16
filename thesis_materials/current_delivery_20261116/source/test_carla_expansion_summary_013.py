"""验证汇总的有效分母和审计入口，不运行模型/仿真或触碰正式结果。"""
import json
from pathlib import Path
import tempfile
import unittest

from summarize_carla_expansion_013 import aggregate, classify, validate_closed, METRICS


class SummaryTests(unittest.TestCase):
    def test_timeout_with_two_audits_is_included_but_partial_audit_is_not(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            dest = root / 'episode'
            dest.mkdir()
            (dest / 'summary.json').write_text('{}')
            job = root / 'job.json'
            self.assertEqual(classify(dest, job, {})[0], 'executed_pending_audits')
            receipt = dict(exit_code=0, audits=[dict(source='audit_carla_expansion_009.py', exit_code=0)])
            job.write_text(json.dumps(receipt))
            self.assertEqual(classify(dest, job, {})[0], 'execution_or_audit_error')
            receipt['audits'].append(dict(source='audit_carla_guarded_execution_002.py', exit_code=0))
            job.write_text(json.dumps(receipt))
            self.assertEqual(classify(dest, job, {})[0], 'closed_independently_audited')

    def test_nonarrival_task_and_unclosed_positive_do_not_dilute_arrival_denominator(self):
        def row(name, kind, state, arrived):
            return dict(phase='primary', mode='adapter', condition='full',
                        episode=name, config_id='same_config', seed=17,
                        task_kind=kind, completion_state=state,
                        status='ARRIVED' if arrived else 'TIMEOUT', wall_s=10.,
                        **{m: arrived if m == 'strict_safe_arrival' else False for m in METRICS})
        rows = [row('a', 'positive', 'closed_independently_audited', True),
                row('b', 'positive', 'closed_independently_audited', False),
                row('c', 'no_match', 'closed_independently_audited', False),
                row('d', 'positive', 'started_unclosed', False)]
        groups = {r['task_kind']: r for r in aggregate(rows)}
        self.assertEqual(groups['positive']['arrival_task_denominator'], 2)
        self.assertEqual(groups['positive']['positive_arrival_fraction_of_closed'], .5)
        self.assertEqual(groups['no_match']['arrival_task_denominator'], 0)
        self.assertIsNone(groups['no_match']['positive_arrival_fraction_of_closed'])
        self.assertEqual(groups['positive']['distinct_configs'], 1)
        self.assertIsNone(groups['positive']['CI'])

    def test_unsafe_arrival_cannot_be_counted_even_when_summary_claims_arrived(self):
        summary = dict(frames=2, status='ARRIVED', GT_in_agent=False,
                       actually_arrived=True, collisions=[])
        audit = dict(physical_ticks=2, status='ARRIVED', GT_not_sent_to_agent=True,
                     strict_safe_arrival=True, correct_lock=True, wrong_lock=False,
                     lane_violation_ticks=[], actuation_limits={'speed_violations': 1})
        guard = dict(physical_ticks=2, every_tick_guard_executed=True,
                     every_tick_obstacles_reconstructed=True, raw_LiDAR_frames_verified=2,
                     next_controls_match_next_actual_applied=True)
        with self.assertRaises(AssertionError):
            validate_closed(summary, audit, guard)


if __name__ == '__main__':
    unittest.main()
