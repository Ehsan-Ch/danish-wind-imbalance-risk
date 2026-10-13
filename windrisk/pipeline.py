from collections import defaultdict
from datetime import timedelta
from pathlib import Path
import csv
import json
import math
import numpy as np
from .data import load, stamp
from .model import fit_bundle, predict
from .risk import settlement, tail_loss, stress_cost


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(data_dir, output_dir, days=30, share=.01):
    if not isinstance(days, int) or days < 1 or not math.isfinite(share) or not 0 < share <= 1:
        raise ValueError('Positive integer days and 0 < share <= 1 required')
    series, prices, quality = load(data_dir)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty output directory to preserve existing results')
    latest = min(max(s) for s in series.values())
    end = latest.replace(hour=0) + (timedelta(days=1) if latest.hour == 23 else timedelta())
    start = end - timedelta(days=days)
    forecast_rows, positions, audit, metrics = [], [], {}, {}
    skipped = defaultdict(int)
    for area, history in series.items():
        for horizon in (1, 6, 24):
            key = f'{area}_{horizon}'
            bundle = fit_bundle(history, horizon, start)
            audit[key] = bundle['audit']
            rows = []
            for i in range(days * 24):
                target = start + timedelta(hours=i)
                try:
                    row = {'area': area, **predict(bundle, history, target, horizon), 'actual_mwh': history[target]}
                except KeyError:
                    skipped[key + '_missing_generation_or_lags'] += 1
                    continue
                rows.append(row)
                forecast_rows.append(row)
                if target not in prices[area]:
                    skipped[key + '_missing_prices'] += 1
                    continue
                market = prices[area][target]
                for policy in ('q10', 'q50', 'q90'):
                    exposure = settlement(row['actual_mwh'], row[policy + '_mwh'],
                                          market['dayahead_eur_mwh'], market['imbalance_eur_mwh'], share)
                    positions.append({'area': area, 'horizon_hours': horizon, 'policy': policy,
                        'target_utc': row['target_utc'], 'issue_utc': row['issue_utc'],
                        'record_type': 'HISTORICAL_SCENARIO', 'share': share, **market, **exposure})
            if not rows:
                raise ValueError(f'No valid test forecasts for {key}')
            errors = {name: np.array([r[name] - r['actual_mwh'] for r in rows])
                      for name in ('forecast_mwh', 'ridge_mwh', 'persistence_mwh')}
            metrics[key] = {'area': area, 'horizon_hours': horizon, 'model': bundle['selected'],
                'forecast_count': len(rows), 'expected_forecasts': days * 24,
                'mean_actual_mwh': float(np.mean([r['actual_mwh'] for r in rows])),
                'mae_mwh': float(np.mean(np.abs(errors['forecast_mwh']))),
                'rmse_mwh': float(np.sqrt(np.mean(errors['forecast_mwh'] ** 2))),
                'bias_mwh': float(np.mean(errors['forecast_mwh'])),
                'ridge_mae_mwh': float(np.mean(np.abs(errors['ridge_mwh']))),
                'persistence_mae_mwh': float(np.mean(np.abs(errors['persistence_mwh']))),
                'coverage90': sum(r['lower90_mwh'] <= r['actual_mwh'] <= r['upper90_mwh'] for r in rows) / len(rows),
                'interval_mean_width_mwh': float(np.mean([r['upper90_mwh'] - r['lower90_mwh'] for r in rows])),
                'policies': {}}
            for q in (.1, .5, .9):
                error = np.array([r['actual_mwh'] - r[f'q{int(q*100)}_mwh'] for r in rows])
                metrics[key][f'pinball_q{int(q*100)}'] = float(np.mean(np.maximum(q * error, (q - 1) * error)))
    daily = defaultdict(lambda: {'effect_eur': 0., 'shortfall_mwh': 0., 'surplus_mwh': 0., 'hours': 0})
    for r in positions:
        k = (r['area'], r['horizon_hours'], r['policy'], r['target_utc'][:10])
        daily[k]['effect_eur'] += r['imbalance_effect_eur']
        daily[k]['shortfall_mwh'] += r['shortfall_mwh']
        daily[k]['surplus_mwh'] += r['surplus_mwh']
        daily[k]['hours'] += 1
    daily_rows = [{'area': a, 'horizon_hours': h, 'policy': p, 'day': day, **v}
                  for (a, h, p, day), v in sorted(daily.items())]
    for key, m in metrics.items():
        for policy in ('q10', 'q50', 'q90'):
            rows = [r for r in positions if r['area'] == m['area'] and r['horizon_hours'] == m['horizon_hours'] and r['policy'] == policy]
            if not rows:
                raise ValueError(f'No priced positions for {key}')
            complete = [r for r in daily_rows if r['area'] == m['area'] and r['horizon_hours'] == m['horizon_hours'] and r['policy'] == policy and r['hours'] == 24]
            shortfall = sum(r['shortfall_mwh'] for r in rows)
            surplus = sum(r['surplus_mwh'] for r in rows)
            m['policies'][policy] = {'priced_hours': len(rows), 'share': share,
                'imbalance_effect_eur': sum(r['imbalance_effect_eur'] for r in rows),
                'gross_revenue_eur': sum(r['revenue_eur'] for r in rows),
                'perfect_volume_revenue_eur': sum(r['perfect_volume_revenue_eur'] for r in rows),
                'shortfall_mwh': shortfall, 'surplus_mwh': surplus,
                'stress_cost_eur': stress_cost(shortfall, surplus),
                'complete_days': len(complete),
                'daily_loss': tail_loss([r['effect_eur'] for r in complete]) if complete else None}
    result = {'test_start_utc': start.isoformat(), 'test_end_exclusive_utc': end.isoformat(),
              'days': days, 'share': share, 'data_quality': quality, 'skipped': dict(skipped),
              'method': 'Frozen direct ridge vs persistence, chronological model selection and held-out calibration',
              'availability_assumption': 'Hourly interval start + 2 hours (1 hour delivery + 1 hour reporting)',
              'settlement_assumption': 'Counterfactual nomination; flat intra-hour energy at quarterly DA/imbalance prices',
              'metrics': metrics, 'model_audit': audit}
    manifest = Path(data_dir) / 'manifest.json'
    if manifest.exists():
        result['provenance'] = json.loads(manifest.read_text(encoding='utf-8'))
    output.mkdir(parents=True, exist_ok=True)
    (output / 'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    write_csv(output / 'forecasts.csv', forecast_rows)
    write_csv(output / 'positions.csv', positions)
    write_csv(output / 'daily_exposure.csv', daily_rows)
    from .report import render
    (output / 'report.html').write_text(render(result, forecast_rows, daily_rows), encoding='utf-8')
    return result


def future(data_dir, target_value, destination):
    series, _, _ = load(data_dir)
    target = stamp(target_value)
    if target.minute or target.second or target.microsecond:
        raise ValueError('Target must be a full UTC hour')
    rows = []
    for area, history in series.items():
        for horizon in (1, 6, 24):
            bundle = fit_bundle(history, horizon, target)
            rows.append({'area': area, **predict(bundle, history, target, horizon)})
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output, rows)
    return {'forecasts': len(rows), 'path': str(output), 'target_utc': target.isoformat()}
