"""Counterfactual volume nominations and a transparent settlement identity."""
import math
import numpy as np


def settlement(actual_mwh, nominated_mwh, day_ahead, imbalance, share=.01):
    values = (actual_mwh, nominated_mwh, day_ahead, imbalance, share)
    if not all(math.isfinite(v) for v in values) or min(actual_mwh, nominated_mwh) < 0 or not 0 < share <= 1:
        raise ValueError('Finite inputs, nonnegative energy, and 0 < share <= 1 required')
    actual, nominated = actual_mwh * share, nominated_mwh * share
    deviation = actual - nominated  # + surplus; - shortfall
    benchmark = actual * day_ahead
    revenue = nominated * day_ahead + deviation * imbalance
    return {'actual_mwh': actual, 'nominated_mwh': nominated, 'deviation_mwh': deviation,
            'revenue_eur': revenue, 'perfect_volume_revenue_eur': benchmark,
            'imbalance_effect_eur': deviation * (imbalance - day_ahead),
            'shortfall_mwh': max(-deviation, 0.), 'surplus_mwh': max(deviation, 0.)}


def tail_loss(values, level=.95):
    """Positive losses only. Empirical quantile and top ceil((1-level)*n) mean."""
    if not values or not 0 < level < 1 or not all(math.isfinite(v) for v in values):
        raise ValueError('A nonempty finite sample and level in (0,1) are required')
    losses = np.maximum(0., -np.asarray(values))
    k = max(1, math.ceil((1 - level) * len(losses) - 1e-12))
    return {'historical_var95_eur': float(np.quantile(losses, level)),
            'historical_tail_mean95_eur': float(np.sort(losses)[-k:].mean()),
            'tail_observations': k, 'observations': len(losses)}


def stress_cost(shortfall, surplus, shortfall_price=100., surplus_price=25.):
    if not all(math.isfinite(v) and v >= 0 for v in (shortfall, surplus, shortfall_price, surplus_price)):
        raise ValueError('Stress inputs must be finite and nonnegative')
    return shortfall * shortfall_price + surplus * surplus_price
