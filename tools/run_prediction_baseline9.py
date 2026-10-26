#!/usr/bin/env python3
"""Run the nine pinned FP16 CPU predictions and package logs/results."""
import argparse
from datetime import datetime, timezone
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import time

CASES = [
 ("decoder_llm","small","openai-community/gpt2","607a30d783dfa663caf39e06633721c8d4cfcd7e"),
 ("decoder_llm","medium","openai-community/gpt2-medium","6dcaa7a952f72f9298047fd5137cd6e4f05f41da"),
 ("decoder_llm","large","openai-community/gpt2-xl","15ea56dee5df4983c59b2538573817e1667135e2"),
 ("vision_cls","small","google/mobilenet_v2_1.0_224","75e607b00aeae1297cc89d026a118bce012f5c5a"),
 ("vision_cls","medium","microsoft/resnet-50","34c2154c194f829b11125337b98c8f5f9965ff19"),
 ("vision_cls","large","google/vit-large-patch16-224","9e2727f4250d3973839eecfa5c4b42e41b709a50"),
 ("text_encdec","small","google-t5/t5-small","df1b051c49625cf57a3d0d8d3863ed4d13564fe4"),
 ("text_encdec","medium","google-t5/t5-base","a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1"),
 ("text_encdec","large","google-t5/t5-3b","bed96aab9ee46012a5046386105ee5fd0ac572f0"),
]
CONFIG_SCRIPT = (
 "import json,sys; from huggingface_hub import hf_hub_download; "
 "p=hf_hub_download(repo_id=sys.argv[1],filename='config.json',revision=sys.argv[2]); "
 "print(json.dumps(json.load(open(p))))"
)

def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def command(argv, **kwargs):
    return subprocess.run(argv, text=True, encoding="utf-8", errors="replace", **kwargs)

def parse_result(text):
    decoder = json.JSONDecoder()
    for match in reversed(list(re.finditer(r"(?m)^\{", text))):
        try:
            result, _ = decoder.raw_decode(text[match.start():])
            if isinstance(result, dict) and result.get("status") in ("ok", "error"):
                return result
        except json.JSONDecodeError:
            pass
    raise ValueError("No prediction JSON found; inspect the case log")

def request(kind, model, config):
    args = dict(model=model, workload_type=kind, model_config=config,
                dtype="fp16", batch_size=1, gpu_memory_gb=24)
    if kind == "decoder_llm":
        args.update(seq_len=256, max_new_tokens=64, mode="generate")
    elif kind == "text_encdec":
        args.update(encoder_seq_len=256, decoder_seq_len=1,
                    max_new_tokens=64, mode="generate")
    else:
        args.update(resolution=224, mode="prefill")
    return args

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", required=True)
    a = p.parse_args()
    os.umask(0o022)
    root = Path(a.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "prediction.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another baseline9 runner is already running")
    repo = Path(__file__).resolve().parents[1]
    run = root / ("run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + str(os.getpid()))
    run.mkdir()
    (root / "LATEST.txt").write_text(str(run) + "\n", encoding="utf-8")
    start = time.monotonic()
    state = dict(status="running", total_cases=9, completed_cases=0, pid=os.getpid(),
                 started_utc=datetime.now(timezone.utc).isoformat(), cases=CASES)
    records = []
    print("Results directory:", run, flush=True)
    try:
        source = repo / "predict_vram.py"
        if "MARGIN_BYTES" in source.read_text(encoding="utf-8"):
            raise RuntimeError("Predictor still contains the old fixed margin")
        state["git_commit"] = command(["git","-C",str(repo),"rev-parse","HEAD"],
                                      capture_output=True, check=True).stdout.strip()
        state["predictor_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        write(run / "manifest.json", state)
        running = command(["docker","top","xmem-cpu","-eo","pid,args"],
                          capture_output=True, check=True).stdout
        if "predict_vram.py" in running:
            raise RuntimeError("An existing prediction is running; it was left untouched")
        patch = repo / "xMem/patches/core_inference.patch"
        command(["docker","exec","-i","-w","/work/xMem","xmem-cpu","git",
                 "-c","safe.directory=/work/xMem","apply","--reverse","--check","-"],
                input=patch.read_text(encoding="utf-8"), capture_output=True, check=True)
        command(["docker","cp",str(source),"xmem-cpu:/work/xMem/predict_vram.py"],
                capture_output=True, check=True)
        digest = command(["docker","exec","xmem-cpu","sha256sum","/work/xMem/predict_vram.py"],
                         capture_output=True, check=True).stdout.split()[0]
        if digest != state["predictor_sha256"]:
            raise RuntimeError("Container predictor differs from repository source")
        shutil.copyfile(source, run / "predict_vram.used.py")
        shutil.copyfile(patch, run / "core_inference.used.patch")
        shutil.copyfile(Path(__file__), run / "runner.used.py")
        for number, (kind, size, model, revision) in enumerate(CASES, 1):
            case_id = f"core__{kind}__{size}__fp16__01"
            print(f"[{number}/9] START {case_id} {model}", flush=True)
            case_start = time.monotonic()
            record = dict(case_id=case_id, model=model, revision=revision, status="error")
            try:
                config = command(["docker","exec","xmem-cpu","python","-c",CONFIG_SCRIPT,model,revision],
                                 capture_output=True, check=True)
                args = request(kind, model, json.loads(config.stdout))
                write(run / (case_id + ".input.json"), args)
                raw = run / (case_id + ".log")
                with raw.open("w", encoding="utf-8") as output:
                    proc = command(["docker","exec","-i","-w","/work/xMem","xmem-cpu",
                                    "python","-u","predict_vram.py","--json","-"],
                                   input=json.dumps(args), stdout=output, stderr=subprocess.STDOUT)
                result = parse_result(raw.read_text(encoding="utf-8"))
                write(run / (case_id + ".prediction.json"), result)
                if proc.returncode != 0 or result["status"] != "ok":
                    raise RuntimeError(result.get("error", f"Exit code {proc.returncode}"))
                b = result["breakdown_bytes"]
                if b.get("fixed_margin") != 0:
                    raise RuntimeError("Nonzero fixed margin found in prediction")
                record.update(status="ok", **{k: b.get(k) for k in [
                    "predicted_allocated_scaled", "after_underestimate_correction",
                    "kv_cache_estimate", "fixed_margin"]},
                    required_vram_bytes=result["required_vram_bytes"],
                    required_vram_gb=result["required_vram_gb"],
                    elapsed_seconds=result.get("elapsed_seconds"),
                    warnings=result.get("warnings", []))
            except Exception as exc:
                record["error"] = str(exc)
                if isinstance(exc, subprocess.CalledProcessError):
                    record["error"] += "\n" + (exc.stderr or "")
            record["wall_seconds"] = time.monotonic() - case_start
            records.append(record)
            with (run / "predictions.jsonl").open("a", encoding="utf-8") as output:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
            state["completed_cases"] = len(records)
            write(run / "manifest.json", state)
            print(f"[{number}/9] {record['status']} {case_id} {record['wall_seconds']:.1f}s", flush=True)
        state["status"] = "complete" if all(x["status"] == "ok" for x in records) else "complete_with_errors"
    except Exception as exc:
        state.update(status="failed", error=str(exc))
        if isinstance(exc, subprocess.CalledProcessError):
            state["error"] += "\n" + (exc.stderr or "")
        print("FAILED:", state["error"], flush=True)
    finally:
        state.update(total_wall_seconds=time.monotonic()-start,
                     success_count=sum(x["status"]=="ok" for x in records))
        write(run / "manifest.json", state)
        fields = ["case_id","model","revision","status","predicted_allocated_scaled",
                  "after_underestimate_correction","kv_cache_estimate","fixed_margin",
                  "required_vram_bytes","required_vram_gb","elapsed_seconds","wall_seconds","warnings","error"]
        with (run / "summary.csv").open("w", encoding="utf-8-sig", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in records:
                writer.writerow({**row, "warnings":json.dumps(row.get("warnings",[]),ensure_ascii=False)})
        archive = run / "prediction-results.tar.gz"
        with tarfile.open(archive, "w:gz") as bundle:
            for file in sorted(run.iterdir()):
                if file.is_file() and file != archive:
                    bundle.add(file, arcname=run.name + "/" + file.name)
        temporary = root / ".latest-predictions.tar.gz.tmp"
        shutil.copyfile(archive, temporary)
        os.replace(temporary, root / "latest-predictions.tar.gz")
        print("STATUS:", state["status"], flush=True)
        print("SEND THIS FILE:", root / "latest-predictions.tar.gz", flush=True)
    return 0 if state["status"] == "complete" else 1

if __name__ == "__main__":
    raise SystemExit(main())
