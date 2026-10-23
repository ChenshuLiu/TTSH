"""Time-series and Clarke-grid figures; all retained samples are plotted."""
from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from modeling import metrics, write_csv

CONVERSION = 18.0

ZONE_COLORS = dict(A='#21835C', B='#327CB2', C='#A88213', D='#D26D28', E='#B63C55')

REGIONS = [
    ('A', [(0,0),(70,0),(70,56),(400,320),(400,400),(400/1.2,400),(70/1.2,70),(0,70)]),
    ('B', [(70,84),(400/1.2,400),(290,400),(70,180)]),
    ('B', [(70,0),(130,0),(180,70),(240,70),(240,180),(400,180),(400,320),(70,56)]),
    ('C', [(70,180),(290,400),(70,400)]),
    ('C', [(130,0),(180,0),(180,70)]),
    ('D', [(0,70),(70/1.2,70),(70,84),(70,180),(0,180)]),
    ('D', [(240,70),(400,70),(400,180),(240,180)]),
    ('E', [(0,180),(70,180),(70,400),(0,400)]),
    ('E', [(180,0),(400,0),(400,70),(180,70)]),
]

def classify(actual, predicted):
    """mg/dL; prioritize inclusive A boundaries, then E, C, D, and B."""
    a, p = np.broadcast_arrays(np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float))
    if not (np.isfinite(a).all() and np.isfinite(p).all()):
        raise ValueError('Nonfinite glucose pair')
    accurate = ((a <= 70) & (p <= 70)) | ((p >= .8*a) & (p <= 1.2*a))
    opposite = ((a >= 180) & (p <= 70)) | ((a <= 70) & (p >= 180))
    overcorrect = ((a >= 70) & (a <= 290) & (p >= a+110)) | ((a >= 130) & (a <= 180) & (p <= 1.4*a-182))
    missed = ((a >= 240) & (p >= 70) & (p <= 180)) | ((a <= 70) & (p >= 70) & (p <= 180))
    return np.select([accurate, opposite, overcorrect, missed], ['A', 'E', 'C', 'D'], default='B')

def summary(rows):
    counts = {z:sum(p['clarke_zone'] == z for p in rows) for z in 'ABCDE'}
    assert sum(counts.values()) == len(rows)
    return dict(n=len(rows), counts=counts,
                percent={z:100*n/len(rows) for z,n in counts.items()},
                A_plus_B_percent=100*(counts['A']+counts['B'])/len(rows))


def save_figure(fig, output_dir, stem):
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    for ext in ['png','svg']:
        fig.savefig(output/f'{stem}.{ext}',dpi=220,bbox_inches='tight')


def plot_subject_timeseries(result, subject, output_dir=None):
    rows=[p for p in result['points'] if p['subject_id']==subject]
    ids=list(dict.fromkeys(p['event_id'] for p in rows))
    if not ids:raise ValueError('No points for subject')
    fig,axes=plt.subplots(1,len(ids),figsize=(4*len(ids),3.4),sharey=True,squeeze=False)
    for index,(ax,eid) in enumerate(zip(axes[0],ids)):
        points=sorted((p for p in rows if p['event_id']==eid),key=lambda p:p['timestamp'])
        split=points[0]['split'];color='#007C91' if split=='train' else '#C34D32'
        x=[p['elapsed_seconds']/60 for p in points]
        ax.plot(x,[p['actual_cgm'] for p in points],'o-',ms=3,color='#292929',label='Observed CGM')
        ax.plot(x,[p['predicted_cgm'] for p in points],'o--',ms=2,color=color,label='Reconstructed CGM')
        label='Calibration' if split=='train' else 'Held out'
        ax.set_title(f'Event {index+1}: {label}\nMARD {metrics(points)["mard_percent"]:.2f}%')
        ax.set_xlabel('Minutes from labeled CGM start');ax.grid(axis='y',alpha=.25)
    axes[0,0].set_ylabel('Glucose (mmol/L)');axes[0,0].legend(frameon=False,fontsize=8)
    fig.suptitle(f'{subject} — retrospective reconstruction');fig.tight_layout()
    if output_dir is not None:
        # IDs may contain path delimiters in external data; never use them as paths.
        import hashlib
        stem='subject_'+hashlib.sha256(subject.encode()).hexdigest()[:10]+'_time_series'
        save_figure(fig,output_dir,stem)
    return fig


def clarke_rows(points):
    if not points:raise ValueError('No paired values for Clarke error grid')
    zones=classify([p['actual_cgm']*CONVERSION for p in points],[p['predicted_cgm']*CONVERSION for p in points])
    return [dict(p,clarke_zone=str(z)) for p,z in zip(points,zones)]


def plot_clarke(points, title='Held-out reconstruction', output_dir=None, stem='clarke_held_out'):
    rows=clarke_rows(points);stats=summary(rows)
    fig,ax=plt.subplots(figsize=(6.5,6))
    for zone,vertices in REGIONS:
        ax.add_patch(Polygon(np.array(vertices)/CONVERSION,facecolor=ZONE_COLORS[zone],alpha=.12,edgecolor='none'))
    boundaries=[([0,70/1.2,400/1.2],[70,70,400]),([70,70],[84,400]),([0,70,290],[180,180,400]),
                ([70,70,400],[0,56,320]),([180,180,400],[0,70,70]),([240,240,400],[70,180,180]),([130,180],[0,70])]
    for x,y in boundaries:ax.plot(np.array(x)/CONVERSION,np.array(y)/CONVERSION,color='#777777',lw=.7)
    ax.plot([0,400/CONVERSION],[0,400/CONVERSION],'--',color='#888888',lw=.7)
    for x,y,z in [(30,15,'A'),(370,260,'B'),(280,370,'B'),(160,370,'C'),(160,15,'C'),(30,140,'D'),(370,120,'D'),(30,370,'E'),(370,15,'E')]:
        ax.text(x/CONVERSION,y/CONVERSION,z,color=ZONE_COLORS[z],ha='center')
    for zone in 'ABCDE':
        group=[p for p in rows if p['clarke_zone']==zone]
        ax.scatter([p['actual_cgm'] for p in group],[p['predicted_cgm'] for p in group],s=14,color=ZONE_COLORS[zone],alpha=.8,label=f'{zone}: {stats["percent"][zone]:.1f}%')
    values=[p[k] for p in rows for k in ['actual_cgm','predicted_cgm']]
    low=min(0,min(values)-.5);high=max(400/CONVERSION,max(values)+.5)
    ax.set(xlim=(low,high),ylim=(low,high),aspect='equal',xlabel='Observed CGM (mmol/L)',ylabel='Reconstructed CGM (mmol/L)',title=f'{title}, n={len(rows)}')
    ax.legend(fontsize=8,frameon=False,loc='center left',bbox_to_anchor=(1.02,.5))
    fig.text(.5,.01,'Comparator: CGM, not laboratory blood glucose. No MARD-based sample filtering.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.03,1,1))
    if output_dir is not None:
        save_figure(fig,output_dir,stem);write_csv(Path(output_dir)/f'{stem}_points.csv',rows)
    return fig,stats
