"""Run the notebook workflow non-interactively on synthetic or user-supplied data."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from modeling import load_events,load_dashboard_events,prepare_subjects,fit_subjects,export_results
from plotting import plot_subject_timeseries,plot_clarke


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--input',type=Path,default=None)
    group.add_argument('--dashboard-root',type=Path)
    parser.add_argument('--output',type=Path,default=Path('outputs/run'))
    args=parser.parse_args()
    if args.output.exists():parser.error('Choose a new output directory to preserve previous results.')
    source=args.input or Path(__file__).resolve().parent/'data/synthetic_events.json'
    events=load_dashboard_events(args.dashboard_root) if args.dashboard_root else load_events(source)
    subjects,rows,errors=prepare_subjects(events)
    result=fit_subjects(subjects,rows,errors)
    result['input_sha256']=hashlib.sha256(source.read_bytes()).hexdigest() if not args.dashboard_root else None
    export_results(result,args.output/'data')
    if not result['points']:raise SystemExit('No eligible subjects; see exported exclusions.')
    for subject in result['coefficients']:plt.close(plot_subject_timeseries(result,subject,args.output/'figures'))
    summaries={}
    for split in ['train','validation']:
        points=[p for p in result['points'] if p['split']==split]
        if points:
            fig,summaries[split]=plot_clarke(points,'Calibration' if split=='train' else 'Held out',args.output/'figures',f'clarke_{split}');plt.close(fig)
    (args.output/'data/clarke_summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    print(f'Completed {len(subjects)} subjects; {len(result["points"])} paired samples. Results: {args.output.resolve()}')


if __name__=='__main__':main()
