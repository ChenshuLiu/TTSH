#!/usr/bin/env python3
"""Create a reproducible, entirely artificial dataset using the production pipeline."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
from datetime import datetime, timedelta


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'synthetic_demo')
    parser.add_argument('--seed', type=int, default=20261023)
    args = parser.parse_args()
    root = args.output.resolve()
    if root.exists():
        parser.error('Output already exists. Choose a new --output directory to preserve existing data.')
    os.environ['SEB_DATA_ROOT'] = str(root)
    import server as alignment
    import cgm_feature_engineering as features
    rng = random.Random(args.seed)
    payload = {'meta': {'synthetic': True, 'seed': args.seed, 'description': 'Artificial demonstration only; no study records used.'},
               'saved_segments': {}, 'peak_pair_features': {'cases': {}}}
    columns = ['Timestamp', 'Channel1', 'Channel2', 'Channel3', 'Delta_Channel2_minus_Channel1',
               'Delta_Channel2_minus_Channel3', 'CGM', 'Normalized_Delta_Channel2_minus_Channel1',
               'Normalized_Delta_Channel2_minus_Channel3', 'Normalized_CGM']
    for subject in range(1, 4):
        # P901-P903 satisfy the application's participant parser and are fictional IDs.
        participant = f'P{900 + subject}'
        start = datetime(2030, 1, subject, 8)
        raw, cgm = [], []
        centers = [70 + subject * 3, 245 + subject * 4]
        for minute in range(361):
            ts = (start + timedelta(minutes=minute)).strftime('%Y-%m-%d %H:%M:%S')
            def pulse(lag=0):
                return sum(math.exp(-0.5 * ((minute - center - lag) / 23) ** 2) for center in centers)
            baseline = 0.15 + 0.00001 * minute
            seg = 0.04 + (0.008 + subject * 0.001) * pulse(8) + rng.gauss(0, 0.00003)
            sel = 0.006 + (0.002 + subject * 0.0003) * pulse(15) + rng.gauss(0, 0.00001)
            raw.append({'Timestamp': ts, 'Channel1': baseline - seg, 'Channel2': baseline, 'Channel3': baseline - sel})
            if minute % 5 == 0:
                cgm.append({'timestamp': ts, 'glucose_mmol_L': round(4.8 + subject * 0.3 + (1.3 + subject * 0.4) * pulse(), 5)})
        source = f'Organized Data/{participant}/Day01/{participant}_Day01_2030010{subject}_SEB.csv'
        write_csv(root / source, raw, ['Timestamp', 'Channel1', 'Channel2', 'Channel3'])
        write_csv(root / f'cleaned_cgm_files/{participant}_cleaned.csv', cgm, ['timestamp', 'glucose_mmol_L'])
        processed = alignment.tool2_process_rows(alignment.load_tool2_series(source), source, 0, 360)
        case = 'analysis/SEB_CGM_alignment/processed_delta_files/' + source
        write_csv(root / case, processed, columns)
        write_json(root / (case + '.metadata.json'), {'tool': 'tool2', 'source_path': source, 'start_index': 0, 'end_index': 360, 'synthetic': True})
        state = {'labels_glu_cgm': [], 'labels_glu_lac': []}
        for signal, lag in [('cgm', 0), ('seg', 8), ('sel', 15)]:
            for i, center in enumerate(centers):
                peak = round((center + lag) / 5) * 5 if signal == 'cgm' else center + lag
                for kind, index, counter in [('trough', i * 180, i + 1), ('peak', peak, i + 1), ('trough', (i + 1) * 180, i + 2)]:
                    label = {'name': f'{signal.upper()} {kind}', 'index': index, 'counter': counter, 'timestamp': raw[index]['Timestamp']}
                    for lane in (['labels_glu_cgm', 'labels_glu_lac'] if signal == 'seg' else ['labels_glu_cgm' if signal == 'cgm' else 'labels_glu_lac']):
                        if label not in state[lane]: state[lane].append(label.copy())
        aggregates = alignment.compute_aggregated_peaks_for_case(case, state, processed)
        assert len(aggregates) == 6
        state['peak_aggregates'] = aggregates
        payload['saved_segments'][case] = state
        pairs = []
        for i in range(2):
            for category, signals, panel in [('CGM_vs_SEG', ['cgm', 'seg'], 'glu_cgm'), ('SEG_vs_SEL', ['seg', 'sel'], 'glu_lac')]:
                name = f'{category}_{i + 1}'
                pair = {'name': name, 'instance_id': f'{case}::{name}', 'case_path': case, 'category': category, 'panel': panel, 'selected_peak_metadata': {}}
                for signal in signals:
                    agg = [a for a in aggregates if a['signal'] == signal][i]
                    pair[signal + '_id'] = agg['id']
                    pair['selected_peak_metadata'][signal] = {'aggregate_id': agg['id'], 'signal': signal, 'pair_context': category, 'label_panel': panel,
                        'trough1_timestamp': agg['left_trough']['timestamp'], 'peak_timestamp': agg['peak']['timestamp'], 'trough2_timestamp': agg['right_trough']['timestamp'], 'preceding_instance': None}
                pairs.append(pair)
        payload['peak_pair_features']['cases'][case] = {'paired_features': pairs}
    write_json(root / 'annotations_input.json', payload)
    output, summary = features.process_annotation_payload(payload)
    assert not summary['case_errors'], summary
    assert summary['cases_seen'] == 3 and summary['peak_metadata_updated'] == 24, summary
    write_json(root / 'analysis/SEB_CGM_alignment/state/alignment_annotations.json', output)
    write_json(root / 'feature_engineering_summary.json', summary)
    write_json(root / 'manifest.json', {'synthetic': True, 'seed': args.seed, 'participants': 3, 'sensor_rows_per_participant': 361,
        'cgm_rows_per_participant': 73, 'paired_records': 12, 'peak_metadata_records': 24,
        'pipeline': ['raw sensor CSV + cleaned CGM CSV', 'tool2_process_rows: baseline correction, CGM matching, normalization', 'processed CSV + metadata JSON', 'peak/trough labels and paired annotation JSON', 'process_annotation_payload: durations, AUC, derivatives, concentration features'],
        'limitations': 'Illustrative Gaussian responses, not a physiological model. No real participant data, lab workbook, meal photos, or clinical validation. Timestamps in feature-engineering provenance reflect execution time; signal generation is seeded.'})
    print(json.dumps(summary, indent=2))
    print(f'Dataset: {root}')


if __name__ == '__main__':
    main()
