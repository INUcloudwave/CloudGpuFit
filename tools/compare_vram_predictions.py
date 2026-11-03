#!/usr/bin/env python3
"""Compare like-for-like: allocated, reserved, calibrated NVML; retain failures."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vram_policy import process_estimate


def compare(predictions, archives, profiles):
    predicted={}
    for path in predictions.glob('*.prediction.json'):
        data=json.loads(path.read_text())
        cid=data.get('case_id') or path.name.removesuffix('.prediction.json')
        if cid in predicted:raise ValueError('Duplicate prediction ID')
        predicted[cid]=data
    records=[]
    for archive_path in archives:
        with tarfile.open(archive_path) as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith('__memory__r0.json'):continue
                actual=json.load(archive.extractfile(member));case=actual['case'];cid=case['case_id']
                row={'archive':archive_path.name,'case_id':cid,'gpu':actual.get('environment',{}).get('gpu'),
                     'model':case['model'],'dtype':case['dtype'],'actual_status':actual['status']}
                pred=predicted.get(cid)
                if actual['status']!='ok':row['status']='actual_'+actual['status']
                elif pred is None:row['status']='missing_prediction'
                elif pred['status']!='ok':row['status']='prediction_error';row['error']=pred.get('error')
                else:
                    if pred.get('revision')!=actual['environment']['revision']:
                        raise ValueError('Model revision mismatch: '+cid)
                    if pred['actual_input_shapes']!=actual['actual_input_shapes']:
                        raise ValueError('Input shape mismatch: '+cid)
                    proc=process_estimate(pred['predicted_allocated_bytes'],pred['predicted_reserved_bytes'],row['gpu'],case['dtype'],profiles)
                    for key,value in profiles['gpus'].get(proc['gpu'],{}).get('runtime_scope',{}).items():
                        if actual['environment'].get(key)!=value:raise ValueError('Overhead calibration runtime mismatch: '+key)
                    row['status']='ok' if proc['status']=='ok' else proc['status']
                    for name,pred_value,actual_value in [
                        ('allocated',pred['predicted_allocated_bytes'],actual['inference_peak_allocated_bytes']),
                        ('reserved',pred['predicted_reserved_bytes'],actual['inference_peak_reserved_bytes']),
                        ('nvml',proc.get('estimated_nvml_bytes'),actual['nvml_process_sampled_peak_bytes'])]:
                        if pred_value is None:continue
                        if actual_value<=0:raise ValueError('Actual memory is not positive')
                        row[name+'_predicted_bytes']=pred_value
                        row[name+'_actual_bytes']=actual_value
                        row[name+'_signed_error_pct']=(pred_value/actual_value-1)*100
                    row['nvml_samples']=len(actual['memory_samples'])
                records.append(row)
    groups={}
    for row in records:
        if row['status']!='ok':continue
        for grouping in ['all',row['gpu'],row['gpu']+'/'+row['dtype'],row['gpu']+'/'+row['model']]:
            for metric in ['allocated','reserved','nvml']:
                groups.setdefault((grouping,metric),[]).append(row[metric+'_signed_error_pct'])
    summary=[]
    for (group,metric),errors in sorted(groups.items()):
        summary.append({'group':group,'metric':metric,'n':len(errors),
            'mape_pct':statistics.mean(abs(e) for e in errors),
            'median_absolute_error_pct':statistics.median(abs(e) for e in errors),
            'max_absolute_error_pct':max(abs(e) for e in errors),
            'mean_signed_error_pct':statistics.mean(errors),
            'underestimate_fraction':sum(e<0 for e in errors)/len(errors)})
    return records,summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--actual',type=Path,nargs='+',required=True)
    p.add_argument('--profiles',type=Path,default=ROOT/'gpu_overheads.json')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rows,summary=compare(a.predictions,a.actual,json.loads(a.profiles.read_text()))
    a.output.mkdir(parents=True,exist_ok=True)
    fields=sorted({k for r in rows for k in r})
    with (a.output/'paired_errors.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    (a.output/'summary.json').write_text(json.dumps({'groups':summary},ensure_ascii=False,indent=2))
    counts={status:sum(r['status']==status for r in rows) for status in sorted({r['status'] for r in rows})}
    print('Status counts:',counts)
    for row in summary:
        if row['group']=='all':print(row)
    return 1 if any(r['status'] in ['missing_prediction','prediction_error'] for r in rows) else 0


if __name__=='__main__':raise SystemExit(main())
