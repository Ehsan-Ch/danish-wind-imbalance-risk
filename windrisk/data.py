"""Strict UTC joins; missing market intervals are reported, never interpolated."""
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
import hashlib
import json
import math

BASE = 'https://api.energidataservice.dk/'
DATASETS = {
    'generation': ('GenerationProdTypeExchange', 'TimeUTC,PriceArea,Version,OffshoreWindPower,OnshoreWindPower'),
    'imbalance': ('ImbalancePrice', 'TimeUTC,PriceArea,ImbalancePriceEUR'),
    'dayahead': ('DayAheadPrices', 'TimeUTC,PriceArea,DayAheadPriceEUR'),
}


def stamp(value):
    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def download(start, end, destination):
    if start < '2025-10-01' or datetime.fromisoformat(start) >= datetime.fromisoformat(end):
        raise ValueError('Use start >= 2025-10-01 and end > start (local Danish dates)')
    folder = Path(destination)
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {'start_local_inclusive': start, 'end_local_exclusive': end, 'sources': {}}
    for name, (dataset, columns) in DATASETS.items():
        url = BASE + 'dataset/' + dataset + '?' + urlencode({
            'start': start, 'end': end, 'filter': json.dumps({'PriceArea': ['DK1', 'DK2']}),
            'sort': 'TimeUTC,PriceArea', 'columns': columns, 'limit': 0})
        with urlopen(url, timeout=90) as response:
            raw = response.read()
        payload = json.loads(raw)
        if not payload.get('records'):
            raise ValueError(f'Empty {dataset} response')
        (folder / f'{name}.json').write_bytes(raw)
        with urlopen(BASE + 'meta/dataset/' + dataset, timeout=30) as response:
            metadata = json.load(response)
        (folder / f'{name}.metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        manifest['sources'][name] = {'dataset': dataset, 'url': url,
            'sha256': hashlib.sha256(raw).hexdigest(), 'records': len(payload['records']),
            'retrieved_utc': datetime.now(timezone.utc).isoformat(),
            'resolution': metadata.get('resolution'), 'caution': metadata.get('caution', '')}
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def normalize(generation, imbalance, dayahead):
    """Generation is hourly MWh. Prices are four equal quarter-hour products."""
    wind, quarters, quality = {}, {}, Counter()
    versions = Counter()
    for row in generation['records']:
        if row['PriceArea'] not in ('DK1', 'DK2'):
            continue
        t, area = stamp(row['TimeUTC']), row['PriceArea']
        key = (area, t)
        if key in wind:
            raise ValueError(f'Duplicate generation interval {key}')
        if t.minute or t.second or t.microsecond:
            raise ValueError('Generation adapter requires hourly MWh data')
        values = [row['OffshoreWindPower'], row['OnshoreWindPower']]
        if any(v is None or not math.isfinite(float(v)) or float(v) < 0 for v in values):
            raise ValueError(f'Invalid generation {key}')
        wind[key] = sum(map(float, values))
        versions[row.get('Version', 'unspecified')] += 1
    for label, payload, column in [('imbalance', imbalance, 'ImbalancePriceEUR'),
                                    ('dayahead', dayahead, 'DayAheadPriceEUR')]:
        seen = set()
        for row in payload['records']:
            if row['PriceArea'] not in ('DK1', 'DK2'):
                continue
            t, area = stamp(row['TimeUTC']), row['PriceArea']
            if (area, t) in seen:
                raise ValueError(f'Duplicate {label} interval')
            seen.add((area, t))
            if t.minute not in (0, 15, 30, 45) or t.second or t.microsecond:
                raise ValueError('Prices must be aligned to quarter hours')
            value = row[column]
            if value is None or not math.isfinite(float(value)):
                quality[label + '_invalid_quarters'] += 1
                continue
            quarters.setdefault((label, area, t.replace(minute=0)), []).append(float(value))
    series, market = {}, {}
    for area in ('DK1', 'DK2'):
        series[area] = {t: v for (a, t), v in wind.items() if a == area}
        if not series[area]:
            raise ValueError(f'No generation observations for {area}')
        market[area] = {}
        times = sorted(series[area])
        quality[area + '_missing_generation_hours'] = sum(int((b - a).total_seconds() / 3600) - 1 for a, b in zip(times, times[1:]))
        for t in times:
            ib = quarters.get(('imbalance', area, t), [])
            da = quarters.get(('dayahead', area, t), [])
            if len(ib) != 4 or len(da) != 4:
                quality[area + '_unpriced_hours'] += 1
                continue
            market[area][t] = {'imbalance_eur_mwh': sum(ib) / 4, 'dayahead_eur_mwh': sum(da) / 4}
    quality['generation_records'] = len(wind)
    return series, market, {'counts': dict(quality), 'generation_versions': dict(versions)}


def load(folder):
    folder = Path(folder)
    payloads = [json.loads((folder / f'{name}.json').read_text(encoding='utf-8')) for name in DATASETS]
    return normalize(*payloads)
