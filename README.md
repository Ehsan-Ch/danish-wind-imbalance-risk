# WindRisk — Danish wind forecasting & imbalance exposure

An end-to-end energy analytics project for **DK1 and DK2**: forecast wind generation at **1, 6 and 24 hours**, quantify forecast uncertainty, and compare the imbalance exposure of alternative nominated volumes.

The project combines real Energinet generation, day-ahead prices and imbalance prices. Its interactive report lets you change the price area, forecast horizon, portfolio share and stress penalties. **All nominations are hypothetical research scenarios. There is no live trading connection.**

## What this project demonstrates

- Public energy-data ingestion, source hashes and dataset-version tracking.
- UTC interval alignment, quarter-hour price aggregation and explicit missing-data accounting.
- Direct ridge forecasts versus a persistence baseline, selected before the test period.
- Separate chronological fitting, model-selection, calibration and test blocks.
- Residual quantile estimates (Q10/Q50/Q90), 90% calibration intervals and observed coverage.
- Volume nomination, surplus/shortfall exposure, historical settlement effects and tail-loss summaries.
- An offline interactive HTML report and auditable CSV exports.
- 20 automated tests, including leakage checks, missing-data handling and settlement identities.

## Quick start

Python 3.11+ and NumPy are required. No API key or paid data subscription is needed.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e .

python -m windrisk.cli download --start 2026-04-01 --end 2026-10-01 --output data
python -m windrisk.cli evaluate --data data --output reports/my-run --days 30 --share 0.01
python -m unittest discover -s tests -v
```

Open `reports/my-run/report.html` in a browser. After downloading the repository, you can also open the included [benchmark report](reports/benchmark/report.html) directly, without Python or a server. GitHub displays its source; download it to use the interactive controls.

`--share 0.01` means a hypothetical **1% share of the entire zone's wind generation**, not a specific wind farm or a 1 MW asset. Evaluation requires an empty output folder to preserve previous results. Network access is needed only for downloading data; the cached evaluation and report work offline.

To produce forecasts for a target hour without supplying its actual generation:

```bash
python -m windrisk.cli forecast --data data --target 2026-10-01T00:00:00Z --output reports/next_forecast.csv
```

The command produces six forecasts: two zones × three horizons, each with its own simulated issue time. The example is historical; use an appropriate recent cache and target for a new experiment.

## Recorded benchmark

Data window: April 1–October 1, 2026, Danish local dates, end exclusive. Test window: **August 31–September 29, 2026**, 720 hourly predictions per zone/horizon and **4,320 forecasts** in total. The last incomplete UTC day is excluded. No missing generation or price intervals were encountered in this run.

| Area | Lead time | Selected before testing | Selected MAE, MWh | Persistence MAE, MWh | 90% interval coverage |
|---|---:|---|---:|---:|---:|
| DK1 | 1 h | Ridge | 350.42 | 414.87 | 87.78% |
| DK1 | 6 h | Persistence | 873.72 | 873.72 | 81.11% |
| DK1 | 24 h | Ridge | 1,079.77 | 1,211.94 | 84.72% |
| DK2 | 1 h | Ridge | 142.03 | 154.22 | 91.67% |
| DK2 | 6 h | Ridge | 275.13 | 309.28 | 89.86% |
| DK2 | 24 h | Ridge | 346.77 | 423.38 | 86.39% |

Errors refer to aggregate zone energy in a one-hour delivery interval. They should not be interpreted as a particular plant's error.

**An instructive result:** persistence won the earlier DK1 six-hour selection block. Ridge subsequently achieved a lower test MAE of 748.84 MWh. The project retains the preselected persistence model rather than selecting the winner using test outcomes. Model choice can become stale after a regime change.

Most intervals under-cover relative to the nominal 90% target. At 24 hours, both models also show substantial negative bias. Weather forecasts and more extensive multi-regime evaluation are natural next steps; these results do not establish production forecasting quality.

## Evaluation and availability rules

1. **Target:** hourly onshore plus offshore generation, in MWh, for each price area.
2. **Issue time:** target interval start minus the chosen lead time. A one-hour forecast is issued one hour before delivery starts.
3. **Availability assumption:** an observation becomes usable at its interval start plus two hours: one hour of delivery and one assumed reporting hour. The newest input is therefore 3, 8 or 26 hours before the target for the respective horizons.
4. **Features:** generation lags 0/1/2/3/6/12/24/48/168 hours relative to that newest available observation; recent changes; trailing 24-hour mean and standard deviation; and target-hour sine/cosine terms. No future realized weather, prices or generation enter the features.
5. **Chronological splits:** reserve the last 14 eligible pre-test days for calibration and the preceding 14 for model selection. Fit ridge only on earlier history. The label cutoff accounts for the first test issue time and the two-hour availability assumption.
6. **Selection:** choose ridge or persistence by MAE on the selection block. Ridge uses standardized features and fixed penalty 20. It is not refitted after selection. Coefficients, selected model and calibration adjustments stay frozen during testing; observations used as lag features update as they become eligible.
7. **Uncertainty:** signed calibration residual quantiles produce Q10, Q50 and Q90 forecasts; a finite-sample absolute-residual rank produces the separate symmetric 90% interval. Negative generation predictions/bounds are clipped at zero. Q10–Q90 spans nominally 80%, not 90%. Temporal dependence and nonstationarity prevent a general coverage guarantee.
8. **Evaluation:** MAE, RMSE, bias, interval coverage/width and quantile pinball losses. Missing generation history causes the affected forecast to be skipped and counted, never interpolated.

Exact split dates, sample sizes and validation scores appear in [metrics.json](reports/benchmark/metrics.json).

## Imbalance-risk calculation

For each forecast, compare three alternative nominated volumes: Q10, Q50 and Q90. They are **mutually exclusive scenarios**, not three positions to add together. Each horizon is also a separate experiment.

```text
actual_energy    = aggregate_zone_actual × portfolio_share
nominated_energy = aggregate_zone_quantile × portfolio_share
deviation       = actual_energy − nominated_energy

scenario_revenue = nominated_energy × day_ahead_price
                 + deviation × imbalance_price

perfect_volume_reference = actual_energy × day_ahead_price
imbalance_effect         = deviation × (imbalance_price − day_ahead_price)
```

A positive deviation is surplus production; a negative deviation is a shortfall. A positive imbalance effect benefits the scenario relative to selling actual output at day-ahead prices. Surplus is not automatically profitable: the sign also depends on the price spread, including when prices are negative.

Generation is hourly. Prices are quarterly. Assuming actual generation and nomination are flat across all four quarters, using equal-weight hourly average prices gives the same settlement arithmetic as summing four quarter-hour allocations. It does **not** reproduce actual within-hour generation variability.

The report separately offers stress costs:

```text
stress_cost = shortfall_MWh × assumed_shortfall_charge
            + surplus_MWh × assumed_surplus_charge
```

Default illustrative charges are EUR 100/MWh for shortfalls and EUR 25/MWh for surplus. Changing them does not change historical market prices or historical imbalance effects. The portfolio-share slider scales all exposure linearly, assuming the portfolio exactly tracks aggregate zone generation.

Daily loss is `max(0, -daily_imbalance_effect)`. The 95% empirical value-at-risk and the mean of the worst `ceil(5% × complete_days)` losses are reported. With only 30 test days, the tail mean uses two observations: this is descriptive historical exposure, not a reliable estimate of future tail risk. Only complete 24-hour UTC days enter these tail summaries; partial priced days remain visible in the ordinary exposure totals.

## Outputs

| File | Contents |
|---|---|
| [report.html](reports/benchmark/report.html) | Standalone interactive dashboard with local scenario controls |
| [metrics.json](reports/benchmark/metrics.json) | Forecast and risk metrics, split audit, provenance and data quality |
| [forecasts.csv](reports/benchmark/forecasts.csv) | All 4,320 test forecasts, issue times, bounds and actual outcomes |
| [positions.csv](reports/benchmark/positions.csv) | 12,960 historical nomination scenarios; explicitly labelled `HISTORICAL_SCENARIO` |
| [daily_exposure.csv](reports/benchmark/daily_exposure.csv) | Daily imbalance effect, shortfall/surplus and priced-hour counts |

The raw download cache stays local and is excluded from Git. SHA-256 source hashes are preserved in the metrics. Source revisions can cause a later download of the same dates to differ.

## Data sources and limitations

- [Generation per Production Type and Exchange](https://www.energidataservice.dk/tso-electricity/GenerationProdTypeExchange)
- [Imbalance Price](https://www.energidataservice.dk/tso-electricity/ImbalancePrice)
- [Day-Ahead Prices](https://www.energidataservice.dk/tso-electricity/DayAheadPrices)
- [Official API guide](https://www.energidataservice.dk/guides/api-guides)

Energinet generation combines Initial, Preliminary and Final versions and returns the best currently available value. This benchmark contains **7,728 Preliminary and 1,056 Initial records**. It is a retrospective forecasting experiment on an archive, not a reconstruction of the exact information available at each historical issue time. A reporting-delay assumption does not remove revision leakage. The downloaded metadata also records the provider's announced calculation-method change associated with DataHub 3.0.

API date parameters use Danish local dates; all internal joins and horizons use UTC. Duplicate intervals and invalid generation are rejected. Incomplete/invalid market-price hours are excluded from settlement and counted. The current adapter supports hourly generation and quarter-hour prices from October 2025 onward; a schema/resolution change requires adapting and validating the ingestion logic.

The nomination experiments deliberately do not implement auction deadlines or intraday rebalancing. In particular, a one- or six-hour forecast generally cannot be treated as an executable day-ahead nomination at the historical day-ahead price. These scenarios isolate volume and price-spread sensitivity. There are no execution claims, fees, taxes, curtailment model, plant-capacity constraint or physical wind-farm model.

## Tests

The 20 tests cover time-zone/DST handling, units, duplicate/missing/invalid data, feature and model look-ahead protection, split embargoes, forecast quantile ordering, forecasts without future actuals, negative prices, settlement accounting, share scaling, tail-loss arithmetic, and the full export pipeline. GitHub Actions is configured for Python 3.11, 3.12 and 3.13.

## License

Code: MIT, see [LICENSE](LICENSE). Energinet source data remains subject to the provider's terms and attribution requirements.
