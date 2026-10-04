"""Step 05: GLMY persistent path homology and H0-H3 barcodes.
Adapted from 0317glmy.py; signed Effect (not scaled visualization weight)
is shifted by +100 for GLMY input, then restored for plotting.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import tempfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def barcode(homology, path, title, effect_min, effect_max, shift):
    # Birth 0 is the initial vertex filtration level, mapped to -shift.
    restored = {str(d): [[float(b)-shift, None if e == -1 else float(e)-shift]
                         for b,e in homology.get(str(d), [])] for d in range(4)}
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    left = min(effect_min, min([b for bars in restored.values() for b,e in bars if b > -shift] or [effect_min]))
    right = max(effect_max, max([e for bars in restored.values() for b,e in bars if e is not None] or [effect_max]))
    pad = max((right-left)*0.03, 0.05)
    for ax, d, color in zip(axes, (3,2,1,0), ('#8054a3','#38934a','#cc4b4b','#3478b7')):
        bars = sorted(restored[str(d)], key=lambda v: (v[0], float('inf') if v[1] is None else v[1]))
        for j,(birth,death) in enumerate(bars):
            end = right+pad if death is None else death
            start = max(birth, left-pad)
            ax.plot([start,end], [j,j], color=color, lw=1.5)
            if death is not None and abs(end-start) < pad*0.01:
                ax.plot(end,j,'|',color=color,markersize=5)
            if birth < left-pad:
                ax.annotate('',xy=(left-pad,j),xytext=(left,j),arrowprops={'arrowstyle':'->','color':color,'lw':1})
            if death is None:
                ax.annotate('', xy=(end,j), xytext=(end-pad,j), arrowprops={'arrowstyle':'->','color':color,'lw':1.5})
        if not bars: ax.text(.5,.5,'No intervals',transform=ax.transAxes,ha='center',color='grey')
        ax.set_ylabel(f'H{d}\n({len(bars)} bars)')
        ax.set_ylim(-1, max(len(bars),3))
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='x', alpha=.15)
    axes[-1].set_xlim(left-pad,right+2*pad)
    axes[-1].set_xlabel('Filtration threshold (signed Effect; GLMY threshold minus shift)')
    fig.suptitle(title + '\nLeft arrows: classes born at the initial vertex level; ticks: zero-length bars', fontsize=11)
    fig.tight_layout(rect=(0,0,1,.96))
    fig.savefig(path/'barcode.png',dpi=200)
    fig.savefig(path/'barcode.pdf')
    plt.close(fig)
    return restored


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weight-shift',type=float,default=100)
    parser.add_argument('--timeout',type=float,default=120)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    exe=root/'tools/GLMY.exe'
    source=root/'results/04_network_analysis'
    out=root/'results/05_glmy_homology'
    out.mkdir(parents=True,exist_ok=True)
    scores=pd.read_csv(root/'results/02_clustering/summary.csv')
    best=scores.sort_values(['BIC','cluster_number','run']).iloc[0]
    model=json.loads((root/f'results/02_clustering/k_{int(best.cluster_number):02d}/run_{int(best.run):02d}/result.json').read_text())
    modules=np.asarray(model['max_omega_logi'])+1
    proteins=np.asarray(model['protein_ids'])
    inputs=sorted(source.rglob('*_links.csv'))
    assert len(inputs)==40
    minimum = min(float(pd.read_csv(f).Effect.min()) for f in inputs if len(pd.read_csv(f)))
    args.weight_shift = max(args.weight_shift, 1.0 - minimum)
    print(f"Global weight shift: {args.weight_shift:.6f}", flush=True)
    records=[]
    for csv in inputs:
        relative=csv.parent.relative_to(source)
        scale,group,module=relative.parts
        folder=out/relative
        folder.mkdir(parents=True,exist_ok=True)
        edges=pd.read_csv(csv)
        assert edges.columns.tolist()==['From','To','size','Effect','edge_type','weight']
        vertices=sorted([f'M{i}' for i in range(1,int(model['cluster_number'])+1)] if scale=='coarse_grained'
                        else proteins[modules==int(module[1:])].tolist())
        ids={v:i+1 for i,v in enumerate(vertices)}
        assert set(edges.From).union(edges.To)<=set(ids)
        weights=edges.Effect.to_numpy(dtype=float)+args.weight_shift
        assert np.isfinite(weights).all() and (weights>0).all(), 'Increase --weight-shift'
        lines=[','.join(str(ids[v]) for v in vertices)]
        lines += [f'({ids[r.From]},{ids[r.To]},{w:.17g})' for r,w in zip(edges.itertuples(),weights)]
        text='\n'.join(lines)+'\n#\n4\ny\n\n\n'
        (folder/'glmy_input.txt').write_text(text,encoding='ascii')
        pd.DataFrame({'Node':vertices,'GLMY_ID':range(1,len(vertices)+1)}).to_csv(folder/'vertex_mapping.csv',index=False)
        print(f'Starting {relative}: {len(vertices)} vertices, {len(edges)} edges',flush=True)
        # Temporary isolated cwd prevents one network from reusing another's JSON.
        with tempfile.TemporaryDirectory(prefix='glmy_') as work:
            try:
                proc=subprocess.run([str(exe)],input=text.encode('ascii'),stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,cwd=work,timeout=args.timeout)
            except subprocess.TimeoutExpired as e:
                (folder/'stdout.log').write_bytes(e.stdout or b'')
                records.append({'Scale':scale,'Group':group,'Module':module,'Status':'timeout'})
                pd.DataFrame(records).to_csv(out/'summary.csv',index=False)
                print(f'TIMEOUT {relative}',flush=True)
                continue
            (folder/'stdout.log').write_bytes(proc.stdout)
            (folder/'stderr.log').write_bytes(proc.stderr)
            raw=Path(work)/'homology.json'
            if proc.returncode!=0 or not raw.exists():
                records.append({'Scale':scale,'Group':group,'Module':module,'Status':f'failed:{proc.returncode}'})
                pd.DataFrame(records).to_csv(out/'summary.csv',index=False)
                print(f'FAILED {relative}',flush=True)
                continue
            homology=json.loads(raw.read_text(encoding='utf-8-sig'))
            for dim,bars in homology.items():
                for pair in bars:
                    assert len(pair)==2 and np.isfinite(pair).all()
                    assert pair[1]==-1 or pair[1]>=pair[0], pair
            (folder/'homology_raw.json').write_text(json.dumps(homology,indent=2),encoding='utf-8')
        restored=barcode(homology,folder,f'{group} | {scale} | {module}',
                         float(edges.Effect.min()) if len(edges) else 0,
                         float(edges.Effect.max()) if len(edges) else 0,args.weight_shift)
        (folder/'homology_effect_scale.json').write_text(json.dumps(restored,indent=2,allow_nan=False),encoding='utf-8')
        records.append({'Scale':scale,'Group':group,'Module':module,'Status':'complete','Vertices':len(vertices),'Edges':len(edges),
                        **{f'H{d}_bars':len(homology.get(str(d),[])) for d in range(4)}})
        pd.DataFrame(records).to_csv(out/'summary.csv',index=False)
        print(f'Finished {relative}: '+str({f'H{d}':len(homology.get(str(d),[])) for d in range(4)}),flush=True)
    (out/'manifest.json').write_text(json.dumps({'shift':args.weight_shift,'timeout':args.timeout,
        'exe_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'filtration':'signed Effect + shift',
        'isolated_vertices':'included from selected clustering model'},indent=2),encoding='utf-8')
    failures=[r for r in records if r['Status']!='complete']
    print(f'Completed {len(records)-len(failures)}/{len(inputs)} networks; failures={len(failures)}',flush=True)
    if failures: raise SystemExit(1)

if __name__=='__main__': main()
