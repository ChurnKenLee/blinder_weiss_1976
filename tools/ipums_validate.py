#!/usr/bin/env python3
"""Validate completed IPUMS fixed-width extracts against their own DDI files.

This checks acquisition integrity, not survey-to-model measurement assumptions.
It writes aggregate validation results only; no microdata rows are printed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal, InvalidOperation
import gzip
import hashlib
import json
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ipums_download import atomic_json, now  # noqa: E402


def validate(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    if not manifest.get('microdata_downloaded'):
        raise ValueError('The completed microdata manifest is missing')
    for file in manifest['files']:
        path = directory / file['file']
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != file['sha256'] or path.stat().st_size != file['bytes']:
            raise ValueError(f"Checksum/size validation failed: {path.name}")
    ddi = next(directory.glob('*.xml'))
    archive = next(directory.glob('*.dat.gz'))
    columns = {}
    for node in ET.parse(ddi).getroot().findall('.//{*}var'):
        position = node.find('{*}location')
        columns[node.attrib['name']] = {
            'start': int(position.attrib['StartPos']) - 1,
            'end': int(position.attrib['EndPos']),
            'decimals': int(node.attrib.get('dcml', '0')),
        }
    width = max(item['end'] for item in columns.values())
    collection = manifest['extract']['collection']
    weight_name = 'PERWT' if collection == 'usa' else 'WT06'
    years, ages = Counter(), Counter()
    minimum_weight, maximum_weight, weight_sum = None, None, 0
    respondents = {}
    record_count = 0
    with gzip.open(archive, 'rb') as stream:
        for record_count, raw in enumerate(stream, 1):
            row = raw.rstrip(b'\r\n')
            if len(row) != width:
                raise ValueError(f'Unexpected fixed-width record length at line {record_count}')
            def number(name):
                col = columns[name]
                try:
                    return int(row[col['start']:col['end']])
                except ValueError:
                    raise ValueError(f'Invalid integer in {name} at record {record_count}') from None
            weight_column = columns[weight_name]
            try:
                weight = Decimal(row[weight_column['start']:weight_column['end']].decode().strip())
            except InvalidOperation:
                raise ValueError(f'Invalid weight at record {record_count}') from None
            year, age = number('YEAR'), number('AGE')
            if weight < 0:
                raise ValueError('Negative survey weight')
            years[year] += 1
            ages[age] += 1
            minimum_weight = weight if minimum_weight is None else min(minimum_weight, weight)
            maximum_weight = weight if maximum_weight is None else max(maximum_weight, weight)
            if collection == 'usa':
                weight_sum += weight
            else:
                caseid = (year, number('CASEID'))
                attributes = (age, number('SEX'), number('EMPSTAT'), weight)
                if caseid not in respondents:
                    respondents[caseid] = [0, 0, attributes]
                    weight_sum += weight
                entry = respondents[caseid]
                if entry[2] != attributes:
                    raise ValueError('Respondent attributes or weight vary between activities')
                duration = number('DURATION')
                if not 0 <= duration <= 1440:
                    raise ValueError('Activity duration outside a 24-hour diary')
                entry[0] += duration
                entry[1] += 1
    if record_count == 0:
        raise ValueError('Microdata archive is empty')
    result = {
        'validated_at': now(),
        'collection': collection,
        'extract_number': manifest['extract']['number'],
        'all_archive_and_dictionary_checksums_verified': True,
        'record_count': record_count,
        'record_type': 'person' if collection == 'usa' else 'activity',
        'record_width': width,
        'column_count': len(columns),
        'year_record_counts': dict(sorted(years.items())),
        'age_range': [min(ages), max(ages)],
        'weight_variable': weight_name,
        'weight_decimals': columns[weight_name]['decimals'],
        'unscaled_weight_range': [str(minimum_weight), str(maximum_weight)],
        'survey_weight_sum': float(weight_sum / 10**columns[weight_name]['decimals']),
        'survey_weight_sum_unit': 'persons' if collection == 'usa' else 'respondent-days; apply weights once per respondent',
        'empirical_calibration_moments_constructed': False,
    }
    if collection == 'atus':
        totals = [x[0] for x in respondents.values()]
        if any(total != 1440 for total in totals):
            raise ValueError('An ATUS respondent diary does not total 1440 minutes')
        result.update(
            respondent_count=len(respondents),
            diary_minutes_range=[min(totals), max(totals)],
            all_respondent_attributes_consistent=True,
            every_diary_totals_1440_minutes=True,
        )
    atomic_json(directory / 'validation.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.directory), indent=2))


if __name__ == '__main__':
    main()
