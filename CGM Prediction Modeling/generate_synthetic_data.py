"""Standalone artificial paired events; no real participant data are read."""
import argparse
from datetime import datetime, timedelta
import json
import math
from pathlib import Path
import random


def generate(seed=20261023):
    rng=random.Random(seed);events=[]
    for subject in range(1,4):
        for event in range(3):
            start=datetime(2035,1,subject*4)+timedelta(days=event,hours=8)
            seg_start=start+timedelta(minutes=7)
            seg_peak=seg_start+timedelta(minutes=105)
            seg_end=seg_start+timedelta(minutes=210)
            cgm_peak=start+timedelta(minutes=80)
            cgm_end=start+timedelta(minutes=180)
            stamp=lambda t:t.isoformat(' ')
            seg=[];cgm=[]
            for j in range(421):
                fraction=j/420
                value=.04+.012*math.sin(math.pi*fraction)**2+rng.gauss(0,.00002)
                seg.append(dict(timestamp=stamp(seg_start+timedelta(seconds=j*30)),value=value))
            for j in range(37):
                minutes=j*5
                phase=minutes/80 if minutes<=80 else (minutes-80)/100
                x=math.sin(math.pi*phase/2)**2 if minutes<=80 else math.cos(math.pi*phase/2)**2
                value=4.8+.4*subject+(2+.25*subject)*x+.4*x*x+.12*event+rng.gauss(0,.04)
                cgm.append(dict(timestamp=stamp(start+timedelta(minutes=minutes)),value=value))
            events.append(dict(subject_id=f'SYNTHETIC_{subject:03d}',event_id=f'event_{event+1}',seg=seg,cgm=cgm,
                               landmarks={'seg':dict(trough1=stamp(seg_start),peak=stamp(seg_peak),trough2=stamp(seg_end)),
                                          'cgm':dict(trough1=stamp(start),peak=stamp(cgm_peak),trough2=stamp(cgm_end))}))
    return dict(synthetic=True,seed=seed,description='Fictional analytic curves; no private observations, dates, or calibrated coefficients.',events=events)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--seed',type=int,default=20261023)
    args=parser.parse_args()
    if args.output.exists():parser.error('Choose a new output file; existing data are not overwritten.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(generate(args.seed),indent=2,allow_nan=False)+'\n')
