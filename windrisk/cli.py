import argparse
import json
from .data import download
from .pipeline import evaluate, future


def main():
    parser = argparse.ArgumentParser(description='Wind generation forecasting and imbalance exposure research')
    commands = parser.add_subparsers(dest='command', required=True)
    get = commands.add_parser('download', help='Download and cache public Energinet data')
    get.add_argument('--start', required=True)
    get.add_argument('--end', required=True)
    get.add_argument('--output', default='data')
    run = commands.add_parser('evaluate', help='Run forecasting benchmark and historical nomination scenarios')
    run.add_argument('--data', default='data')
    run.add_argument('--output', default='reports/run')
    run.add_argument('--days', type=int, default=30)
    run.add_argument('--share', type=float, default=.01, help='Share of aggregate zone wind generation, e.g. 0.01 = 1%%')
    predict = commands.add_parser('forecast', help='Forecast an hourly target without actual target generation')
    predict.add_argument('--data', default='data')
    predict.add_argument('--target', required=True, help='UTC timestamp, e.g. 2026-10-01T00:00:00Z')
    predict.add_argument('--output', default='reports/next_forecast.csv')
    args = parser.parse_args()
    try:
        if args.command == 'download':
            result = download(args.start, args.end, args.output)
        elif args.command == 'evaluate':
            result = evaluate(args.data, args.output, args.days, args.share)
            result = {'period': [result['test_start_utc'], result['test_end_exclusive_utc']],
                      'data_quality': result['data_quality'], 'skipped': result['skipped'],
                      'metrics': {k: {key: value for key, value in m.items() if key != 'policies'} for k, m in result['metrics'].items()}}
        else:
            result = future(args.data, args.target, args.output)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
