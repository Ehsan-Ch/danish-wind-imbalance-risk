from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import math
import tempfile
import unittest
import numpy as np

from windrisk.data import normalize, stamp
from windrisk.model import features, fit_bundle, predict
from windrisk.pipeline import evaluate, future
from windrisk.risk import settlement, stress_cost, tail_loss

UTC = timezone.utc


def history(days=100):
    start = datetime(2026, 1, 1, tzinfo=UTC)
    return {start + timedelta(hours=i): 1200 + 700 * math.sin(i / 45) + 100 * math.sin(i / 6)
            for i in range(days * 24)}


def payloads(days=1):
    generation, imbalance, dayahead = [], [], []
    for t, v in history(days).items():
        for area in ('DK1', 'DK2'):
            generation.append({'TimeUTC': t.isoformat(), 'PriceArea': area, 'Version': 'Synthetic',
                               'OnshoreWindPower': v * .6, 'OffshoreWindPower': v * .4})
            for m in (0, 15, 30, 45):
                base = {'TimeUTC': (t + timedelta(minutes=m)).isoformat(), 'PriceArea': area}
                imbalance.append({**base, 'ImbalancePriceEUR': 20 + m})
                dayahead.append({**base, 'DayAheadPriceEUR': -10 + m})
    return [{'records': r} for r in (generation, imbalance, dayahead)]


class DataTests(unittest.TestCase):
    def test_hourly_mwh_not_summed_four_times(self):
        series, market, q = normalize(*payloads())
        t = min(series['DK1'])
        self.assertAlmostEqual(series['DK1'][t], 1200)
        self.assertEqual(market['DK1'][t]['imbalance_eur_mwh'], 42.5)
        self.assertEqual(market['DK1'][t]['dayahead_eur_mwh'], 12.5)
        self.assertEqual(q['generation_versions']['Synthetic'], 48)

    def test_missing_price_excludes_only_affected_hour(self):
        p = payloads()
        p[1]['records'].pop(0)
        series, market, q = normalize(*p)
        self.assertEqual(len(series['DK1']), 24)
        self.assertEqual(len(market['DK1']), 23)
        self.assertEqual(q['counts']['DK1_unpriced_hours'], 1)

    def test_null_price_reported_not_zero_filled(self):
        p = payloads()
        p[1]['records'][0]['ImbalancePriceEUR'] = None
        _, _, quality = normalize(*p)
        self.assertEqual(quality['counts']['imbalance_invalid_quarters'], 1)
        self.assertEqual(quality['counts']['DK1_unpriced_hours'], 1)

    def test_duplicate_generation_rejected(self):
        p = payloads()
        p[0]['records'].append(p[0]['records'][0])
        with self.assertRaises(ValueError):
            normalize(*p)

    def test_duplicate_price_rejected(self):
        p = payloads()
        p[1]['records'].append(p[1]['records'][0])
        with self.assertRaises(ValueError):
            normalize(*p)

    def test_invalid_generation_rejected(self):
        for value in (-1, float('nan'), None):
            p = payloads()
            p[0]['records'][0]['OnshoreWindPower'] = value
            with self.assertRaises(ValueError):
                normalize(*p)

    def test_missing_generation_reported_without_fill(self):
        p = payloads()
        p[0]['records'].pop(10)
        series, _, q = normalize(*p)
        self.assertEqual(len(series['DK1']), 23)
        self.assertEqual(q['counts']['DK1_missing_generation_hours'], 1)

    def test_dst_offsets_preserve_distinct_utc_hours(self):
        a, b = stamp('2026-10-25T02:00:00+02:00'), stamp('2026-10-25T02:00:00+01:00')
        self.assertEqual(b - a, timedelta(hours=1))
        self.assertEqual(a.tzinfo, UTC)


class ModelTests(unittest.TestCase):
    def test_future_observations_cannot_change_features(self):
        s = history()
        target = datetime(2026, 3, 20, tzinfo=UTC)
        for h in (1, 6, 24):
            latest = target - timedelta(hours=h+2)
            altered = {t: v if t <= latest else v + 1e6 for t, v in s.items()}
            self.assertEqual(features(s, target, h), features(altered, target, h))

    def test_test_period_cannot_change_model_selection_or_calibration(self):
        s = history()
        start = datetime(2026, 3, 20, tzinfo=UTC)
        cutoff = start - timedelta(hours=8)
        changed = {t: v if t < cutoff else v + 1e6 for t, v in s.items()}
        one, two = fit_bundle(s, 6, start), fit_bundle(changed, 6, start)
        self.assertEqual(one['audit'], two['audit'])
        np.testing.assert_array_equal(one['model'].coefficients, two['model'].coefficients)

    def test_split_embargo_and_nonoverlap(self):
        start = datetime(2026, 3, 20, tzinfo=UTC)
        audit = fit_bundle(history(), 24, start)['audit']
        self.assertLess(stamp(audit['fit_last']), stamp(audit['selection_first']))
        self.assertLess(stamp(audit['selection_last']), stamp(audit['calibration_first']))
        self.assertLess(stamp(audit['calibration_last']) + timedelta(hours=2), start - timedelta(hours=24))

    def test_future_forecast_without_actual_and_ordered_quantiles(self):
        s = history()
        target = max(s) + timedelta(hours=1)
        bundle = fit_bundle(s, 6, target)
        p = predict(bundle, s, target, 6)
        self.assertLessEqual(0, p['q10_mwh'])
        self.assertLessEqual(p['q10_mwh'], p['q50_mwh'])
        self.assertLessEqual(p['q50_mwh'], p['q90_mwh'])
        self.assertLessEqual(p['lower90_mwh'], p['forecast_mwh'])
        self.assertLessEqual(p['forecast_mwh'], p['upper90_mwh'])

    def test_invalid_horizon_and_insufficient_history(self):
        with self.assertRaises(ValueError):
            features(history(), datetime(2026, 3, 20, tzinfo=UTC), 2)
        with self.assertRaises(ValueError):
            fit_bundle(history(20), 1, datetime(2026, 2, 1, tzinfo=UTC))


class RiskTests(unittest.TestCase):
    def test_settlement_identity_and_signs(self):
        r = settlement(100, 110, 50, 100, share=1)
        self.assertEqual(r['revenue_eur'], 4500)
        self.assertEqual(r['imbalance_effect_eur'], -500)
        self.assertEqual(r['revenue_eur'] - r['perfect_volume_revenue_eur'], r['imbalance_effect_eur'])
        self.assertEqual(r['shortfall_mwh'], 10)
        self.assertEqual(r['surplus_mwh'], 0)

    def test_negative_prices_and_surplus_can_lose_money(self):
        r = settlement(110, 100, -10, -100, share=1)
        self.assertEqual(r['imbalance_effect_eur'], -900)
        self.assertEqual(r['surplus_mwh'], 10)

    def test_share_scales_all_exposure(self):
        a, b = settlement(100, 90, 50, 80, .01), settlement(100, 90, 50, 80, .02)
        for k in a:
            self.assertAlmostEqual(2*a[k], b[k])

    def test_tail_loss_ignores_gains_and_uses_worst_days(self):
        r = tail_loss([100, -200] + [0] * 38)
        self.assertEqual(r['tail_observations'], 2)
        self.assertEqual(r['historical_tail_mean95_eur'], 100)
        self.assertEqual(tail_loss([1, 2, 3])['historical_tail_mean95_eur'], 0)

    def test_stress_separate_from_historical_settlement(self):
        self.assertEqual(stress_cost(10, 20, 100, 25), 1500)
        self.assertEqual(stress_cost(10, 20, 0, 0), 0)

    def test_invalid_risk_inputs(self):
        for share in (0, -1, 2, float('nan')):
            with self.assertRaises(ValueError):
                settlement(100, 100, 50, 50, share)
        with self.assertRaises(ValueError):
            tail_loss([])
        with self.assertRaises(ValueError):
            stress_cost(-1, 2)


class PipelineTests(unittest.TestCase):
    def test_full_pipeline_and_prospective_forecast(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for name, p in zip(('generation', 'imbalance', 'dayahead'), payloads(85)):
                (folder / (name + '.json')).write_text(json.dumps(p), encoding='utf-8')
            r = evaluate(folder, folder / 'result', days=2)
            self.assertEqual(len(r['metrics']), 6)
            self.assertEqual(r['metrics']['DK1_6']['forecast_count'], 48)
            self.assertEqual(r['metrics']['DK2_24']['policies']['q50']['complete_days'], 2)
            self.assertTrue((folder / 'result' / 'report.html').is_file())
            self.assertNotIn('NaN', (folder / 'result' / 'metrics.json').read_text())
            prospective = future(folder, '2026-03-27T00:00:00Z', folder / 'next.csv')
            self.assertEqual(prospective['forecasts'], 6)
            with self.assertRaises(ValueError):
                evaluate(folder, folder / 'result', days=2)


if __name__ == '__main__':
    unittest.main()
