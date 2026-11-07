"""Export a portable HTML metrics report, optionally refresh it while jobs run."""
import argparse
import datetime
import json
from pathlib import Path
import time
from summarize_benchmarks import summarize


def read(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError,ValueError):
        return None


def build(root,training_report,output):
    training=read(training_report) if training_report else None
    reports=[v for p in root.rglob('benchmark.json') if (v:=read(p))]
    comparison=summarize(reports)
    data=dict(generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        training=training,comparison=comparison,convergence_claim=False)
    if not training:
        paths=list(root.rglob('metrics.jsonl'))
        if paths:
            metrics=[]
            for line in paths[-1].read_text(encoding='utf-8').splitlines():
                try:
                    metrics.append(json.loads(line))
                except ValueError:
                    continue
            data['training']=dict(metrics=metrics,method='live metrics',convergence_claim=False)
    # Paths, server identities and tracebacks are unnecessary in the browser.
    if data['training']:
        data['training']={k:data['training'][k] for k in ['method','samples','updates','initial_updates',
            'new_updates','device','elapsed_seconds','metrics'] if k in data['training']}
    for group in comparison['groups']:
        for episode in group['episodes']:
            episode.pop('traceback',None)
            episode.pop('shutdowns',None)
    output.mkdir(parents=True,exist_ok=True)
    raw=json.dumps(data,ensure_ascii=False,allow_nan=False).replace('</','<\\/')
    template=Path(__file__).with_name('monitor.html').read_text(encoding='utf-8')
    for name,text in [('monitor-data.json',raw),('index.html',template.replace('__MONITOR_DATA__',raw))]:
        path=output/name
        tmp=path.with_suffix('.tmp')
        tmp.write_text(text,encoding='utf-8')
        tmp.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--training-report',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--watch-seconds',type=float,default=0)
    args=p.parse_args()
    while True:
        build(args.root,args.training_report,args.output)
        if args.watch_seconds<=0:
            break
        time.sleep(max(2,args.watch_seconds))


if __name__=='__main__':
    main()
