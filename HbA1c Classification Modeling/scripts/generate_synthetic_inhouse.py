#!/usr/bin/env python3
"""Generate schema-compatible fictional exports without reading clinical records."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
from datetime import datetime, timedelta

BUNDLE = Path(__file__).resolve().parent.parent
SCHEMA = BUNDLE / 'cgm data/synthetic_inhouse/schema.json'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def nested(flat):
    result = {}
    for key, value in flat.items():
        parts = key.split('.')
        cursor = result
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New output directory; existing directories are not overwritten.')
    parser.add_argument('--seed', type=int, default=20261023)
    args = parser.parse_args()
    root = args.output.resolve()
    if root.exists():
        parser.error('Output already exists; choose a new directory.')
    schema = json.loads(SCHEMA.read_text())
    tables = {name: [] for name in schema['tables']}
    rng = random.Random(args.seed)
    inference = []
    for index in range(9):
        subject = f'SYNTHETIC_{index + 1:03d}'
        group = index % 3
        patient = {key: None for key in schema['tables']['patient_metadata.csv']}
        patient.update(subject_id=subject, metadata_available=True, hba1c_available=True)
        patient.update({'patient.age': 30 + index * 3, 'patient.height': 165 + index,
                        'patient.weight': 60 + index * 2, 'patient.hba1c': [5.2, 6.1, 7.4][group]})
        patient['patient.bmi'] = round(patient['patient.weight'] / (patient['patient.height'] / 100) ** 2, 3)
        tables['patient_metadata.csv'].append(patient)
        subject_events = []
        for event_index in range(3):
            start = datetime(2035, 1, index + 1, 8) + timedelta(days=event_index)
            stamp = lambda minute: (start + timedelta(minutes=minute)).strftime('%Y-%m-%d %H:%M:%S')
            eid = f'{subject}_event_{event_index + 1}'
            uid = hashlib.sha256(eid.encode()).hexdigest()[:20]
            baseline = [5.0, 6.0, 7.5][group] + rng.uniform(-0.15, 0.15)
            amplitude = 1.5 + 0.6 * group + 0.1 * event_index
            values = [baseline + amplitude * math.sin(math.pi * i / 36) ** 2 for i in range(37)]
            predictions = [v + 0.08 * math.sin(2 * math.pi * i / 36) for i,v in enumerate(values)]
            case_path = f'synthetic/{subject}/Day{event_index + 1:02d}/{eid}.csv'
            common = dict(event_uid=uid, event_id=eid, subject_id=subject, day=f'Day{event_index + 1:02d}',
                          event_name=f'CGM_vs_SEG_{event_index + 1}', split='train' if event_index==0 else 'validation')
            event = {key: None for key in schema['tables']['event_features.csv']}
            event.update(common, case_path=case_path, pair_index=event_index)
            for signal in ['seg', 'cgm']:
                series = values if signal=='cgm' else [0.04 + 0.012 * math.sin(math.pi*i/36)**2 for i in range(37)]
                prefix = signal + '.'
                base = dict(aggregate_id=f'{signal}_p18_l0_r36', signal=signal, pair_context='CGM_vs_SEG', label_panel='glu_cgm',
                            trough1_timestamp=stamp(0), peak_timestamp=stamp(90), trough2_timestamp=stamp(180),
                            duration_trough1_to_trough2_seconds=10800, duration_trough1_to_peak_seconds=5400,
                            duration_peak_to_trough2_seconds=5400, duration_trough1_to_trough2_points=36,
                            duration_trough1_to_peak_points=18, num_preceding_peaks_since_instance=0, num_presceding_peak=0)
                event.update({prefix+k:v for k,v in base.items()})
                for point,j in [('trough1',0),('peak',18),('trough2',36)]:
                    point_base=f'{prefix}point_values.{point}.'
                    for key,value in dict(label_timestamp=stamp(j*5),matched_timestamp=stamp(j*5),matched_index=j,time_offset_seconds=0,signal=signal,
                                          value_source='CGM' if signal=='cgm' else 'Delta_Channel2_minus_Channel1',
                                          raw_delta_value=series[j] if signal=='seg' else None,cgm_value=series[j] if signal=='cgm' else None).items():
                        event[point_base+key]=value
                auc=sum((a+b)*150 for a,b in zip(series,series[1:]))
                event.update({prefix+'trough1_to_trough2_auc.'+k:v for k,v in dict(start_timestamp=stamp(0),end_timestamp=stamp(180),signal=signal,
                              value_source='CGM' if signal=='cgm' else 'Delta_Channel2_minus_Channel1',method='trapezoidal',point_count=37,duration_seconds=10800,
                              raw_delta_auc_seconds=auc if signal=='seg' else None,cgm_auc_seconds=auc if signal=='cgm' else None).items()})
                for interval,left,right in [('trough1_to_peak',0,18),('peak_to_trough2',18,36)]:
                    derivatives=[(series[j+1]-series[j])*(right-left) for j in range(left,right)]
                    feature=dict(max_derivative=max(derivatives),min_derivative=min(derivatives),average_derivative=sum(derivatives)/len(derivatives),
                                 point_count=right-left+1,derivative_count=right-left,time_normalized_start=0,time_normalized_end=1,
                                 start_timestamp=stamp(left*5),end_timestamp=stamp(right*5),signal=signal,
                                 value_source='CGM' if signal=='cgm' else 'Delta_Channel2_minus_Channel1',time_normalization='interval normalized to 0..1')
                    event.update({prefix+'interval_derivative_features.'+interval+'.'+k:v for k,v in feature.items()})
            pair=dict(name=common['event_name'],instance_id=f'{case_path}::{common["event_name"]}',case_path=case_path,panel='glu_cgm',category='CGM_vs_SEG',
                      seg_id='seg_p18_l0_r36',cgm_id='cgm_p18_l0_r36',saved_at_utc=stamp(180))
            event.update({'pair.'+k:v for k,v in pair.items()})
            for key in schema['tables']['event_features.csv']:
                if key.startswith('pair.') and ('difference_seconds' in key or 'delay_' in key):event[key]=0.0
            mse=sum((a-p)**2 for a,p in zip(values,predictions))/37
            mean=sum(values)/37
            metrics=dict(n=37,rmse=math.sqrt(mse),mard_percent=sum(abs(a-p)/a for a,p in zip(values,predictions))/37*100,
                         r_squared=1-sum((a-p)**2 for a,p in zip(values,predictions))/sum((a-mean)**2 for a in values))
            event.update({'prediction_metrics.'+k:v for k,v in metrics.items()})
            tables['event_features.csv'].append(event)
            samples=[]
            for j,(actual,predicted) in enumerate(zip(values,predictions)):
                row={key:None for key in schema['tables']['predicted_cgm_timeseries.csv']}
                row.update(common,cgm_timestamp=stamp(j*5),elapsed_seconds_from_cgm_event_start=j*300,
                           interval='trough1_to_peak' if j<=18 else 'peak_to_trough2',actual_cgm_mmol_L=actual,predicted_cgm_mmol_L=predicted)
                row['model_input.intercept']=1.0
                row['model_input.mean_seg_normalized']=math.sin(math.pi*j/36)**2
                row['model_input.mean_seg_normalized_squared']=row['model_input.mean_seg_normalized']**2
                row['model_input.mean_seg_normalized_cubed']=row['model_input.mean_seg_normalized']**3
                samples.append(row);tables['predicted_cgm_timeseries.csv'].append(row)
                tables['timeseries_with_features_and_metadata.csv'].append({**event,**patient,**row})
            subject_events.append({**common,'case_path':case_path,'pair_index':event_index,
                                   'features':nested({k:v for k,v in event.items() if k.startswith(('seg.','cgm.'))}),
                                   'pair_metadata':pair,'prediction_metrics':metrics,'timeseries':samples})
            interpolated=[]
            for j in range(64):
                position=j*36/63;left=int(position);right=min(left+1,36);fraction=position-left
                interpolated.append(predictions[left]*(1-fraction)+predictions[right]*fraction)
            inference.append(dict(subject_id=subject,event_id=eid,values=interpolated))
        # No real fitted coefficients or private provenance are carried over.
        write_json(root/'by_subject'/f'{subject}.json',dict(subject_id=subject,patient_metadata=patient,
                    cgm_model_coefficients={k:None for k in ['baseline','seg_norm','seg_norm_squared','seg_norm_cubed','diffusion_flux_term','signed_sigmoid_phase_term']},events=subject_events))
    dictionary={}
    for name,rows in tables.items():
        columns=schema['tables'][name]
        with (root/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='raise');writer.writeheader();writer.writerows(rows)
        dictionary[name]={key:dict(nonmissing_count=sum(row.get(key) not in (None,'') for row in rows),source='Synthetic generator; unused fields intentionally null') for key in columns}
    write_json(root/'schema.json',schema)
    write_json(root/'data_dictionary.json',dictionary)
    write_json(root/'inference_events.json',inference)
    write_json(root/'manifest.json',dict(synthetic=True,seed=args.seed,contains_real_patient_data=False,
        generation='Analytic curves and fictional metadata; only column names are retained from the original interface.',
        counts=dict(subjects=9,events=27,timeseries_points=999),
        table_shapes={name:dict(rows=len(rows),columns=len(schema['tables'][name])) for name,rows in tables.items()},
        limitations='Toy class-dependent curves; no privacy-preserving statistical fit, clinical validity, or performance claims. Coefficients and unused clinical/meal fields are null. Split is an illustrative CGM-event split, not an HbA1c evaluation split.'))
    print(json.dumps({'subjects':9,'events':27,'timeseries_points':999,'output':str(root)},indent=2))


if __name__=='__main__':
    main()
