#!/usr/bin/env python3
"""Pinned CPU-only native xMem predictions for the GPU benchmark case IDs."""
from __future__ import annotations
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CONTAINER = "xmem-cpu"
CONFIG = "from huggingface_hub import hf_hub_download; import json,sys; print(open(hf_hub_download(sys.argv[1], 'config.json', revision=sys.argv[2])).read())"


def save(path, value):
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n")
    temp.replace(path)


def prediction_json(text):
    decoder=json.JSONDecoder()
    for line in reversed([i for i,c in enumerate(text) if c=='{' and (i==0 or text[i-1]=='\n')]):
        try:
            result,_=decoder.raw_decode(text[line:])
            if result.get("status") in ("ok","error"):return result
        except (ValueError,AttributeError):pass
    raise ValueError("Prediction JSON missing; inspect the case log")


def request(case, config, threads):
    return {"model":case["model"], "workload_type":case["workload_type"],
            "model_config":config,"dtype":case["dtype"], **case["params"],
            "mode":"prefill" if case["workload_type"]=="vision_cls" else "generate",
            "gpu_memory_gb":80,"threads":threads}


def select_cases(plan, case_set):
    cases=[c for c in plan["cases"] if c["suite"]=="core"]
    if len(cases)!=342 or len({c['case_id'] for c in cases})!=342:
        raise ValueError("Expected the exact 342 unique core conditions")
    if case_set=="baseline9":
        cases=[c for c in cases if c['dtype']=='fp16' and c['case_id'].endswith('__01')]
        if len(cases)!=9:raise ValueError("Expected nine baseline conditions")
    return cases


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root',required=True,type=Path)
    p.add_argument('--plan',type=Path,default=ROOT/'benchmarks/aws342/selected-plan.json')
    p.add_argument('--revisions',type=Path,default=ROOT/'benchmarks/aws342/revisions.json')
    p.add_argument('--case-set',choices=['baseline9','all'],default='baseline9')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--prepare',action='store_true',help='Update existing CPU container with validated patches; keep image and dependencies')
    p.add_argument('--preflight-only',action='store_true')
    a=p.parse_args()
    if a.threads<1:p.error('threads must be positive')
    out=a.output_root.resolve();out.mkdir(parents=True,exist_ok=True)
    lock=(out/'prediction.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads(a.plan.read_text());pins=json.loads(a.revisions.read_text());cases=select_cases(plan,a.case_set)
    sources=['predict_vram.py','cpu_tensor_trace.py','vram_policy.py','gpu_overheads.json']
    identity={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources}
    identity.update(plan=hashlib.sha256(a.plan.read_bytes()).hexdigest(),revisions=hashlib.sha256(a.revisions.read_bytes()).hexdigest(),threads=a.threads)
    manifest_path=out/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())['identity']!=identity:
        raise SystemExit('Existing output belongs to different code/inputs. Use a new output directory.')
    def run(argv,**kwargs):
        return subprocess.run(argv,text=True,encoding='utf-8',errors='replace',check=True,**kwargs)
    state=run(['docker','inspect','-f','{{.State.Running}}',CONTAINER],capture_output=True)
    if state.stdout.strip()!='true':raise SystemExit('Start the existing xmem-cpu container first')
    if a.prepare:
        sys.path.insert(0,str(ROOT/'orchestrator'))
        from setup_manager import _apply_patch_if_needed,_sync_predictor,_xmem_dir
        _apply_patch_if_needed(_xmem_dir());_sync_predictor()
    running=run(['docker','top',CONTAINER,'-eo','pid,args'],capture_output=True).stdout
    if 'predict_vram.py' in running:raise SystemExit('Another CPU prediction is running; it was left untouched')
    envcode='import json,torch,transformers,sys; print(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,transformers=transformers.__version__,python=sys.version.split()[0],gpu_available=torch.cuda.is_available())))'
    environment=json.loads(run(['docker','exec',CONTAINER,'python','-c',envcode],capture_output=True).stdout)
    if environment['torch']!='2.6.0+cu124' or environment['cuda']!='12.4' or environment['transformers']!='4.52.4' or environment['python']!='3.11.11' or environment['gpu_available']:
        raise SystemExit('Environment does not match the original CPU container: '+str(environment))
    for name in sources:
        digest=run(['docker','exec',CONTAINER,'sha256sum','/work/xMem/'+name],capture_output=True).stdout.split()[0]
        if digest!=identity[name]:raise SystemExit('Stale container file: '+name+'; use --prepare')
    image_id=run(['docker','inspect','-f','{{.Image}}',CONTAINER],capture_output=True).stdout.strip()
    allocator_lock=json.loads((ROOT/'benchmarks/aws342/xmem-allocator-lock.json').read_text())
    actual_commit=run(['docker','exec','-w','/work/xMem',CONTAINER,'git','rev-parse','HEAD'],capture_output=True).stdout.strip()
    if actual_commit!=allocator_lock['commit']:raise SystemExit('xMem upstream commit differs from the pinned environment')
    check='import sys,json,hashlib,pathlib; files=json.load(sys.stdin); print(json.dumps({n:hashlib.sha256((pathlib.Path("/work/xMem")/n).read_bytes()).hexdigest() for n in files}))'
    actual_hashes=json.loads(run(['docker','exec','-i',CONTAINER,'python','-c',check],input=json.dumps(list(allocator_lock['sha256'])),capture_output=True).stdout)
    if actual_hashes!=allocator_lock['sha256']:raise SystemExit('xMem allocator source differs from the pinned source')
    print('Environment:',environment,'Image:',image_id,flush=True)
    print('Cases:',len(cases),'Results:',out,flush=True)
    if a.preflight_only:return 0
    git_commit=run(['git','-C',str(ROOT),'rev-parse','HEAD'],capture_output=True).stdout.strip()
    manifest={'identity':identity,'status':'running','case_set':a.case_set,'total_cases':len(cases),'environment':environment,'image_id':image_id,'allocator_lock':allocator_lock,'git_commit':git_commit}
    with (out/'invocations.jsonl').open('a') as file:
        file.write(json.dumps({'case_set':a.case_set,'started_unix':time.time(),'git_commit':git_commit})+'\n')
    save(manifest_path,manifest);rows=[];start=time.monotonic();configs={}
    for number,case in enumerate(cases,1):
        cid=case['case_id'];target=out/(cid+'.prediction.json');inputfile=out/(cid+'.input.json')
        if target.exists() and json.loads(target.read_text()).get('status')=='ok':
            result=json.loads(target.read_text());print(f'[{number}/{len(cases)}] REUSE {cid}',flush=True)
        else:
            print(f'[{number}/{len(cases)}] START {cid}',flush=True)
            try:
                model=case['model'];revision=pins[model]['revision']
                if model not in configs:
                    configs[model]=json.loads(run(['docker','exec',CONTAINER,'python','-c',CONFIG,model,revision],capture_output=True).stdout)
                payload=request(case,configs[model],a.threads);save(inputfile,{'case_id':cid,'revision':revision,'request':payload})
                log=out/(cid+'.log')
                with log.open('w') as output:
                    proc=subprocess.run(['docker','exec','-i','-w','/work/xMem',CONTAINER,'python','-u','predict_vram.py','--json','-'],input=json.dumps(payload),text=True,stdout=output,stderr=subprocess.STDOUT)
                result=prediction_json(log.read_text())
                if proc.returncode or result['status']!='ok':raise RuntimeError(result.get('error','Predictor exited unsuccessfully'))
                result.update(case_id=cid,revision=revision)
                save(target,result)
            except Exception as exc:
                result={'status':'error','case_id':cid,'error':str(exc)};save(target,result)
        rows.append({'case_id':cid,'status':result['status'],'predicted_allocated_bytes':result.get('predicted_allocated_bytes'),
                     'predicted_reserved_bytes':result.get('predicted_reserved_bytes'),'error':result.get('error')})
        manifest.update(completed_cases=len(rows),success_count=sum(x['status']=='ok' for x in rows),elapsed_seconds=time.monotonic()-start)
        save(manifest_path,manifest)
        print(f'[{number}/{len(cases)}] {result["status"]} {cid}',flush=True)
    manifest['status']='complete' if all(x['status']=='ok' for x in rows) else 'complete_with_errors'
    save(manifest_path,manifest)
    with (out/'summary.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('Status:',manifest['status'],'Results:',out,flush=True)
    return 0 if manifest['status']=='complete' else 1


if __name__=='__main__':raise SystemExit(main())
