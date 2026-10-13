"""Direct multi-horizon forecasting with explicit observation availability."""
from datetime import timedelta
import math
import numpy as np

AVAILABILITY_HOURS = 2  # one hour delivery, then one hour reporting lag (assumption)
FEATURE_LAGS = (0, 1, 2, 3, 6, 12, 24, 48, 168)


def features(prices, target, horizon):
    if horizon not in (1, 6, 24):
        raise ValueError('Supported horizons: 1, 6, 24 hours')
    issue = target - timedelta(hours=horizon)
    last = issue - timedelta(hours=AVAILABILITY_HOURS)
    lagged = [prices[last - timedelta(hours=h)] for h in FEATURE_LAGS]
    window = [prices[last - timedelta(hours=h)] for h in range(24)]
    return lagged + [lagged[0] - lagged[1], lagged[0] - lagged[4], float(np.mean(window)), float(np.std(window)),
                    math.sin(2 * math.pi * target.hour / 24), math.cos(2 * math.pi * target.hour / 24)], lagged[0]


class Ridge:
    def fit(self, x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        self.mean, self.scale = x.mean(axis=0), x.std(axis=0)
        self.scale[self.scale < 1e-8] = 1.
        z = np.column_stack((np.ones(len(x)), (x - self.mean) / self.scale))
        regularizer = np.eye(z.shape[1]) * 20
        regularizer[0, 0] = 0
        self.coefficients = np.linalg.solve(z.T @ z + regularizer, z.T @ y)
        return self

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        z = np.column_stack((np.ones(len(x)), (x - self.mean) / self.scale))
        return np.maximum(0., z @ self.coefficients)


def rowset(series, horizon, before=None):
    rows = []
    for t in sorted(series):
        if before is not None and t >= before:
            continue
        try:
            x, persistence = features(series, t, horizon)
        except KeyError:
            continue
        rows.append((t, x, series[t], persistence))
    return rows


def fit_bundle(series, horizon, test_start):
    """Freeze model, selection and calibration before the FIRST test issue.

    The two-hour availability embargo applies to all fitting/selection labels.
    Three disjoint chronological blocks: fit, 14-day selection, 14-day calibration.
    """
    cutoff = test_start - timedelta(hours=horizon + AVAILABILITY_HOURS)
    calibration_start = cutoff - timedelta(days=14)
    selection_start = calibration_start - timedelta(days=14)
    rows = rowset(series, horizon, before=cutoff)
    train = [r for r in rows if r[0] < selection_start]
    validation = [r for r in rows if selection_start <= r[0] < calibration_start]
    calibration = [r for r in rows if calibration_start <= r[0] < cutoff]
    if len(train) < 24 * 30 or min(len(validation), len(calibration)) < 24 * 10:
        raise ValueError('Insufficient complete history: need at least about 70 days before testing')
    model = Ridge().fit([r[1] for r in train], [r[2] for r in train])
    val_ridge = model.predict([r[1] for r in validation])
    truth = np.array([r[2] for r in validation])
    errors = {'ridge': float(np.mean(np.abs(truth - val_ridge))),
              'persistence': float(np.mean(np.abs(truth - np.array([r[3] for r in validation]))))}
    selected = min(errors, key=errors.get)
    cal_pred = model.predict([r[1] for r in calibration]) if selected == 'ridge' else np.array([r[3] for r in calibration])
    residual = np.array([r[2] for r in calibration]) - cal_pred
    # Signed residual adjustments for asymmetric quantile estimates.
    adjustments = {str(q): float(np.quantile(residual, q)) for q in (.1, .5, .9)}
    n = len(residual)
    radius = float(np.sort(np.abs(residual))[min(n - 1, math.ceil((n + 1) * .9) - 1)])
    return {'model': model, 'selected': selected, 'adjustments': adjustments, 'radius': radius,
            'audit': {'horizon_hours': horizon, 'selected_model': selected, 'validation_mae_mwh': errors,
                      'fit_rows': len(train), 'selection_rows': len(validation), 'calibration_rows': len(calibration),
                      'fit_first': train[0][0].isoformat(), 'fit_last': train[-1][0].isoformat(),
                      'selection_first': validation[0][0].isoformat(), 'selection_last': validation[-1][0].isoformat(),
                      'calibration_first': calibration[0][0].isoformat(), 'calibration_last': calibration[-1][0].isoformat(),
                      'label_availability_cutoff_exclusive': cutoff.isoformat(), 'residual_adjustments_mwh': adjustments,
                      'interval_radius_mwh': radius}}


def predict(bundle, series, target, horizon):
    x, persistence = features(series, target, horizon)
    ridge = float(bundle['model'].predict([x])[0])
    point = ridge if bundle['selected'] == 'ridge' else persistence
    return {'target_utc': target.isoformat(), 'issue_utc': (target - timedelta(hours=horizon)).isoformat(),
            'latest_observation_utc': (target - timedelta(hours=horizon + AVAILABILITY_HOURS)).isoformat(),
            'horizon_hours': horizon, 'selected_model': bundle['selected'],
            'forecast_mwh': point, 'ridge_mwh': ridge, 'persistence_mwh': persistence,
            'q10_mwh': max(0., point + bundle['adjustments']['0.1']),
            'q50_mwh': max(0., point + bundle['adjustments']['0.5']),
            'q90_mwh': max(0., point + bundle['adjustments']['0.9']),
            'lower90_mwh': max(0., point - bundle['radius']), 'upper90_mwh': point + bundle['radius']}
