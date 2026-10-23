"""Portable implementation of the finalized retrospective SEG-only model.

Numerical smoothing, binning, splitting, and ridge routines are extracted from
original study code; no study records or fitted coefficients are bundled.
"""
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

MIN_TRAIN_POINTS = 12
FEATURE_NAMES = ['intercept', 'seg_norm', 'seg_norm_squared', 'seg_norm_cubed',
                 'seg_rate_per_minute', 'signed_sigmoid_phase']

def finite_float(value):
    try:
        if value is None:
            return None
        if isinstance(value, str) and value.strip().upper() in ("", "NA", "NAN", "NONE"):
            return None
        number = float(value)
        if not math.isfinite(number):
            return None
        return number
    except Exception:
        return None


def smooth_values(values, ratio=0.04):
    if not values:
        return []
    try:
        ratio = max(0.0, min(0.5, float(ratio)))
    except Exception:
        ratio = 0.04
    if ratio <= 0 or len(values) < 5:
        return list(values)
    window = max(5, int(round(len(values) * ratio)))
    if window % 2 == 0:
        window += 1
    window = min(window, 301)
    half = window // 2
    filtered = []
    for idx, value in enumerate(values):
        lo = max(0, idx - half)
        hi = min(len(values), idx + half + 1)
        neighbors = [v for v in values[lo:hi] if v is not None and math.isfinite(v)]
        if value is None or not math.isfinite(value) or len(neighbors) < 3:
            filtered.append(value)
            continue
        ordered = sorted(neighbors)
        med = ordered[len(ordered) // 2]
        deviations = sorted(abs(v - med) for v in neighbors)
        mad = deviations[len(deviations) // 2]
        filtered.append(med if mad > 0 and abs(value - med) > 6 * 1.4826 * mad else value)
    out = []
    for idx, value in enumerate(filtered):
        lo = max(0, idx - half)
        hi = min(len(filtered), idx + half + 1)
        neighbors = [v for v in filtered[lo:hi] if v is not None and math.isfinite(v)]
        out.append(sum(neighbors) / len(neighbors) if neighbors else value)
    return out


def minmax(values):
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if not finite:
        return []
    lo, hi = min(finite), max(finite)
    span = hi - lo
    if span <= 0:
        return [0.5 if v is not None and math.isfinite(v) else None for v in values]
    return [(v - lo) / span if v is not None and math.isfinite(v) else None for v in values]


def sigmoid(value):
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def phase_sigmoid(progress, steepness=10.0):
    return sigmoid(steepness * (progress - 0.5))


def solve_linear_system(matrix, vector):
    n = len(vector)
    aug = [list(matrix[i]) + [vector[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(aug[row][col]))
        if abs(aug[pivot][col]) < 1e-12:
            return [0.0 for _ in range(n)]
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [v / scale for v in aug[col]]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            aug[row] = [v - factor * aug[col][i] for i, v in enumerate(aug[row])]
    return [aug[i][-1] for i in range(n)]


def predict(coefficients, features):
    return sum(a * b for a, b in zip(coefficients, features))


def average_datetime(datetimes):
    valid = [dt for dt in datetimes if dt is not None]
    if not valid:
        return None
    origin = valid[0]
    mean_seconds = sum((dt - origin).total_seconds() for dt in valid) / len(valid)
    return origin + timedelta(seconds=mean_seconds)


def average_numeric(values, default=None):
    clean = [finite_float(value) for value in values]
    clean = [value for value in clean if value is not None]
    if not clean:
        return default
    return sum(clean) / len(clean)


def average_rows_to_cgm_samples(rows, cgm_points, cgm_start, cgm_end):
    cgm_samples = [(dt, value) for dt, value in cgm_points if cgm_start <= dt <= cgm_end]
    if not rows or not cgm_samples:
        return rows
    rows = sorted(rows, key=lambda row: row["mapped_dt"])
    downsampled = []
    for idx, (cgm_dt, cgm_value) in enumerate(cgm_samples):
        left = cgm_start if idx == 0 else cgm_samples[idx - 1][0] + (cgm_dt - cgm_samples[idx - 1][0]) / 2
        right = cgm_end if idx == len(cgm_samples) - 1 else cgm_dt + (cgm_samples[idx + 1][0] - cgm_dt) / 2
        bucket = [row for row in rows if left <= row["mapped_dt"] < right or (idx == len(cgm_samples) - 1 and left <= row["mapped_dt"] <= right)]
        if not bucket:
            continue
        features = []
        feature_count = len(bucket[0].get("features", []))
        for col in range(feature_count):
            values = [finite_float(row["features"][col]) for row in bucket]
            clean = [value for value in values if value is not None]
            features.append(sum(clean) / len(clean) if clean else 0.0)
        if features:
            features[0] = 1.0
        progress = sum(row.get("progress", 0.0) for row in bucket) / len(bucket)
        item = {
            "dt": average_datetime([row.get("dt") for row in bucket]) or bucket[0]["dt"],
            "mapped_dt": cgm_dt,
            "target": cgm_value,
            "features": features,
            "split": "train" if progress <= 0.5 else "validation",
            "seg_norm": average_numeric([row.get("seg_norm") for row in bucket], default=0.0),
            "seg_delta_smooth": average_numeric([row.get("seg_delta_smooth") for row in bucket], default=0.0),
            "seg_delta_raw": average_numeric([row.get("seg_delta_raw") for row in bucket], default=0.0),
            "progress": progress,
            "phase_sigmoid": average_numeric([row.get("phase_sigmoid") for row in bucket], default=0.0),
            "signed_sigmoid_phase": average_numeric([row.get("signed_sigmoid_phase") for row in bucket], default=0.0),
            "limb_direction": bucket[0].get("limb_direction"),
            "interval": bucket[0].get("interval", ""),
            "source_point_count": len(bucket),
        }
        downsampled.append(item)
    return downsampled


def ridge_fit(rows, alpha=1e-6):
    """Standardize on training residual rows; fixed ridge, no held-out tuning."""
    n, p = len(rows), len(rows[0]['features'])
    means = [0.] + [sum(r['features'][j] for r in rows)/n for j in range(1, p)]
    scales = [1.] + [max(1e-8, math.sqrt(sum((r['features'][j]-means[j])**2 for r in rows)/n))
                     for j in range(1, p)]
    xs = [[(x-m)/s for x, m, s in zip(r['features'], means, scales)] for r in rows]
    matrix = [[sum(x[i]*x[j] for x in xs) + (alpha if i == j and i else 0.)
               for j in range(p)] for i in range(p)]
    rhs = [sum(x[j]*r['target'] for x, r in zip(xs, rows)) for j in range(p)]
    scaled = solve_linear_system(matrix, rhs)
    coefficients = [b/s for b, s in zip(scaled, scales)]
    coefficients[0] -= sum(b*m for b, m in zip(coefficients[1:], means[1:]))
    return coefficients


def chronological_split(events, sources, train_count):
    grouped = defaultdict(list)
    for event in events:
        grouped[event['subject_id']].append(event)
    included, exclusions = {}, []
    for subject, items in sorted(grouped.items()):
        ordered = sorted(items, key=lambda e: (min(r['mapped_dt'] for r in sources[e['id']]), str(e['id'])))
        if len(ordered) <= train_count:
            exclusions.append(dict(subject_id=subject, event_id='', error='No later event after requested calibration events'))
            continue
        training = ordered[:train_count]
        rows = [r for e in training for r in sources[e['id']]]
        if len(rows) < MIN_TRAIN_POINTS:
            exclusions.append(dict(subject_id=subject, event_id='',
                error=f'Only {len(rows)} calibration points; need at least {MIN_TRAIN_POINTS}. Earliest event was not skipped.'))
            continue
        cutoff = max(r['mapped_dt'] for r in rows)
        testing = []
        test_rows, removed = {}, 0
        seen = set()
        for event in ordered[train_count:]:
            retained = []
            for row in sources[event['id']]:
                t = row['mapped_dt']
                if t <= cutoff or t in seen:
                    removed += 1
                    continue
                seen.add(t)
                retained.append(row)
            if len(retained) >= 3:
                testing.append(event)
                test_rows[event['id']] = retained
            else:
                exclusions.append(dict(subject_id=subject, event_id=event['id'],
                    error='Fewer than three later, non-overlapping held-out CGM samples'))
        if not testing:
            exclusions.append(dict(subject_id=subject, event_id='', error='No later non-overlapping validation event'))
            continue
        included[subject] = dict(train=training, validation=testing, validation_rows=test_rows,
                                 overlap_removed=removed)
    return included, exclusions


def parse_time(value):
    return datetime.fromisoformat(value)


def validated_series(series, name):
    if len(series) < 3:
        raise ValueError(f'{name}: at least three observations required')
    points = [(parse_time(p['timestamp']), float(p['value'])) for p in series]
    if any(not math.isfinite(v) for _,v in points):
        raise ValueError(f'{name}: nonfinite observation')
    if any(b[0] <= a[0] for a,b in zip(points,points[1:])):
        raise ValueError(f'{name}: timestamps must be strictly increasing and unique')
    return points


def prepare_event(event, smoothing=0.04):
    """Warp native SEG limbs and average expanded features at observed CGM bins."""
    seg = validated_series(event['seg'], 'SEG')
    cgm = validated_series(event['cgm'], 'CGM')
    anchors = {signal:{k:parse_time(event['landmarks'][signal][k]) for k in ['trough1','peak','trough2']} for signal in ['seg','cgm']}
    for signal,points in [('seg',seg),('cgm',cgm)]:
        a=anchors[signal]
        if not a['trough1'] < a['peak'] < a['trough2']:
            raise ValueError(f'{signal}: landmarks must be ordered trough1 < peak < trough2')
        if not points[0][0] <= a['trough1'] or not a['trough2'] <= points[-1][0]:
            raise ValueError(f'{signal}: observations must cover the labeled event')
    if any(value <= 0 for _,value in cgm):
        raise ValueError('CGM must be positive mmol/L values')
    seg=[p for p in seg if anchors['seg']['trough1'] <= p[0] <= anchors['seg']['trough2']]
    norm=minmax(smooth_values([v for _,v in seg],smoothing))
    output=[]
    for limb,left,right,direction in [('trough1_to_peak','trough1','peak',1.),('peak_to_trough2','peak','trough2',-1.)]:
        seg_start,seg_end=anchors['seg'][left],anchors['seg'][right]
        cgm_start,cgm_end=anchors['cgm'][left],anchors['cgm'][right]
        dense=[];previous=None;before=None
        for (when,_),x in zip(seg,norm):
            if not seg_start <= when <= seg_end:continue
            progress=(when-seg_start).total_seconds()/(seg_end-seg_start).total_seconds()
            mapped=cgm_start+(cgm_end-cgm_start)*progress
            seconds=(when-before).total_seconds() if before is not None else 0
            slope=(x-previous)*60/seconds if previous is not None and .5 <= seconds <= 60 else 0.
            phase=direction*(2*phase_sigmoid(progress)-1)
            dense.append(dict(dt=when,mapped_dt=mapped,features=[1.,x,x*x,x*x*x,slope,phase],
                              progress=progress,seg_norm=x,signed_sigmoid_phase=phase,interval=limb))
            previous,before=x,when
        sampled=average_rows_to_cgm_samples(dense,cgm,cgm_start,cgm_end)
        # The original bin helper returns dense rows if a limb has no CGM samples.
        # Only actual observed CGM targets may enter this workflow.
        output.extend(r for r in sampled if 'target' in r)
    truth=dict(cgm)
    unique={r['mapped_dt']:dict(r,target=truth[r['mapped_dt']],event_id=event['event_id'],subject_id=event['subject_id'])
            for r in output if r['mapped_dt'] in truth}
    if not unique:raise ValueError('No observed CGM samples inside labeled limbs')
    return sorted(unique.values(),key=lambda r:r['mapped_dt'])


def metrics(points):
    if not points:return dict(n=0,mard_percent=None,rmse=None,r_squared=None)
    y=[p['actual_cgm'] for p in points];pred=[p['predicted_cgm'] for p in points]
    errors=[a-b for a,b in zip(pred,y)];mean=sum(y)/len(y)
    total=sum((v-mean)**2 for v in y);residual=sum(e*e for e in errors)
    return dict(n=len(y),mard_percent=100*sum(abs(e)/abs(v) for e,v in zip(errors,y))/len(y),
                rmse=math.sqrt(residual/len(y)),r_squared=1-residual/total if total>0 else None)


def prepare_subjects(events, smoothing=0.04):
    """Do not substitute a later calibration event if the earliest event is invalid."""
    groups=defaultdict(list);keys=set()
    for event in events:
        key=(event['subject_id'],event['event_id'])
        if key in keys:raise ValueError('Duplicate subject/event identifier')
        keys.add(key);groups[event['subject_id']].append(event)
    eligible=[];sources={};exclusions=[]
    for subject,items in sorted(groups.items()):
        # A missing start prevents reliable chronological selection; exclude subject.
        try:ordered=sorted(items,key=lambda e:(parse_time(e['landmarks']['cgm']['trough1']),str(e['event_id'])))
        except (KeyError,ValueError,TypeError) as error:
            exclusions.append(dict(subject_id=subject,event_id='',error=str(error)));continue
        if len(ordered)<2:
            exclusions.append(dict(subject_id=subject,event_id='',error='Only one event; no later validation'));continue
        valid=[]
        for index,event in enumerate(ordered):
            # Internal keys remain unique even when local event IDs repeat between subjects.
            internal=json.dumps([subject,event['event_id']],separators=(',',':'))
            try:
                sources[internal]=prepare_event(event,smoothing)
                valid.append(dict(id=internal,subject_id=subject,event=event))
            except (ValueError,KeyError,TypeError) as error:
                exclusions.append(dict(subject_id=subject,event_id=event['event_id'],error=str(error)))
                if index==0:valid=[];break
        eligible.extend(valid)
    subjects,split_errors=chronological_split(eligible,sources,1)
    return subjects,sources,exclusions+split_errors


def fit_subjects(subjects, sources, exclusions=None, smoothing=0.04, alpha=1e-6):
    points=[];coefficients={};event_metrics=[];subject_metrics=[]
    for subject,spec in subjects.items():
        training=[r for e in spec['train'] for r in sources[e['id']]]
        fit=ridge_fit(training,alpha)
        if not all(math.isfinite(v) for v in fit):raise ValueError('Nonfinite coefficients')
        coefficients[subject]=dict(zip(FEATURE_NAMES,fit))
        by_split={'train':[],'validation':[]}
        for split in by_split:
            for item in spec[split]:
                event=item['event']
                rows=sources[item['id']] if split=='train' else spec['validation_rows'][item['id']]
                start=parse_time(event['landmarks']['cgm']['trough1']);current=[]
                for row in rows:
                    p=dict(subject_id=subject,event_id=event['event_id'],split=split,interval=row['interval'],
                           timestamp=row['mapped_dt'].isoformat(' '),event_start_timestamp=start.isoformat(' '),
                           elapsed_seconds=(row['mapped_dt']-start).total_seconds(),actual_cgm=row['target'],
                           predicted_cgm=predict(fit,row['features']),**{f'feature_{i}':v for i,v in enumerate(row['features'])})
                    current.append(p)
                event_metrics.append(dict(subject_id=subject,event_id=event['event_id'],split=split,**metrics(current)))
                points.extend(current);by_split[split].extend(current)
        subject_metrics.append(dict(subject_id=subject,overlap_points_removed=spec['overlap_removed'],
                                    **{f'{split}_{key}':value for split,rows in by_split.items() for key,value in metrics(rows).items()}))
    return dict(model='SEG-only warped minmax, native-time slope, training-only standardized ridge',
                smoothing_ratio=smoothing,ridge_alpha=alpha,feature_names=FEATURE_NAMES,
                interpretation='Retrospective reconstruction using labeled CGM landmarks; not prospective prediction.',
                points=points,coefficients=coefficients,event_metrics=event_metrics,subject_metrics=subject_metrics,
                exclusions=exclusions or [])


def write_csv(path, records):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not records:return
    fields=list(dict.fromkeys(key for r in records for key in r))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(records)


def export_results(result, output_dir):
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    (output/'model_results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for key in ['points','subject_metrics','event_metrics','exclusions']:
        write_csv(output/(key+'.csv'),result[key])
    write_csv(output/'coefficients.csv',[dict(subject_id=s,**c) for s,c in result['coefficients'].items()])


def load_events(path):
    payload=json.loads(Path(path).read_text())
    events=payload['events'] if isinstance(payload,dict) else payload
    if not isinstance(events,list):raise ValueError('Expected an events list')
    return events


def load_dashboard_events(root):
    """Read the companion dashboard's three-folder dataset; no dashboard imports."""
    root=Path(root).resolve()
    payload=json.loads((root/'annotated_data/alignment_annotations.json').read_text())
    cases=payload.get('peak_pair_features',payload.get('paired_features',{})).get('cases',{})
    events=[]
    for case,state in cases.items():
        path=(root/case).resolve()
        if not path.is_relative_to(root):raise ValueError('Case path escapes dataset root')
        with path.open() as stream:rows=list(csv.DictReader(stream))
        match=re.search(r'(?:^|/)([PH]\d+)(?:/|$)',case)
        if not match:raise ValueError(f'Participant folder missing from case path: {case}')
        subject=match.group(1);cgm_file=root/'raw_data/cgm'/f'{subject[0]}{int(subject[1:]):03d}_cleaned.csv'
        with cgm_file.open() as stream:
            cgm=[dict(timestamp=r['timestamp'],value=float(r['glucose_mmol_L'])) for r in csv.DictReader(stream)
                 if r.get('timestamp') and r.get('glucose_mmol_L')]
        for index,pair in enumerate(state.get('paired_features',[])):
            selected=pair.get('selected_peak_metadata',{})
            if not all(signal in selected for signal in ['seg','cgm']):continue
            landmarks={signal:{point:selected[signal][point+'_timestamp'] for point in ['trough1','peak','trough2']} for signal in ['seg','cgm']}
            indices=re.search(r'_p(\d+)_l(\d+)_r(\d+)$',selected['seg'].get('aggregate_id',''))
            if indices:
                for point,i in zip(['peak','trough1','trough2'],map(int,indices.groups())):
                    if 0<=i<len(rows):landmarks['seg'][point]=rows[i]['Timestamp']
            seg=[dict(timestamp=r['Timestamp'],value=float(r['Delta_Channel2_minus_Channel1']) if r.get('Delta_Channel2_minus_Channel1') else float(r['Channel2'])-float(r['Channel1'])) for r in rows]
            events.append(dict(subject_id=subject,event_id=pair.get('instance_id') or f'{case}::{index}',seg=seg,cgm=cgm,landmarks=landmarks))
    return events
