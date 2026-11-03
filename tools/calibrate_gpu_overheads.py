#!/usr/bin/env python3
"""Fit only FP16/condition01 (9 models); report every other case as held out.

No prediction error is fitted here. Samples use same-time NVML minus reserved;
subtracting two separately observed peaks would be invalid.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import tarfile

SCOPE_KEYS = ("torch", "transformers", "cuda_runtime", "driver", "image_id")


def memory_cases(path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with tarfile.open(path) as archive:
        for member in archive:
            if member.isfile() and member.name.endswith("__memory__r0.json"):
                obj = json.load(archive.extractfile(member))
                if obj["status"] == "ok":
                    yield obj, digest


def observations(obj):
    values = []
    for sample in obj["memory_samples"]:
        value = sample["nvml_process_bytes"] - sample["reserved_bytes"]
        if value < 0:
            raise ValueError("NVML < reserved: incompatible/misaligned sample")
        values.append(value)
    if not values:
        raise ValueError("Missing NVML samples")
    return values


def build(fit_files, validation_files):
    training = {}; scopes = {}; sources = {}; heldout = []
    for path in fit_files:
        for obj, digest in memory_cases(path):
            gpu = obj["environment"]["gpu"].replace("Tesla T4", "NVIDIA T4")
            case = obj["case"]
            if case["dtype"] != "fp16" or not case["case_id"].endswith("__01"):
                continue
            if case["case_id"] in training.setdefault(gpu, {}):
                raise ValueError(f"Duplicate calibration case for {gpu}; select one fit run per GPU")
            training[gpu][case["case_id"]] = observations(obj)
            scope = {key: obj["environment"][key] for key in SCOPE_KEYS}
            if gpu in scopes and scopes[gpu] != scope:
                raise ValueError("A GPU profile cannot mix runtime scopes")
            scopes[gpu] = scope
            sources.setdefault(gpu, {})[path.name] = digest
    profiles = {}
    for gpu, cases in training.items():
        if len(cases) != 9:
            raise ValueError(f"{gpu}: expected exactly nine calibration cases, got {len(cases)}")
        values = [v for vals in cases.values() for v in vals]
        profiles[gpu] = {
            "outside_median_bytes": int(statistics.median(statistics.median(v) for v in cases.values())),
            "outside_min_training_sample_bytes": min(values),
            "outside_max_training_sample_bytes": max(values),
            "supported_dtypes": ["fp16", "fp32"] + ([] if gpu == "NVIDIA T4" else ["bf16"]),
            "runtime_scope": scopes[gpu], "training_case_ids": sorted(cases),
            "source_archives_sha256": sources[gpu],
        }
    for path in fit_files + validation_files:
        for obj, digest in memory_cases(path):
            gpu = obj["environment"]["gpu"].replace("Tesla T4", "NVIDIA T4")
            profile = profiles.get(gpu)
            if not profile:
                raise ValueError(f"No fit run for {gpu}")
            if {k:obj["environment"][k] for k in SCOPE_KEYS} != profile["runtime_scope"]:
                raise ValueError("Validation scope differs from fitted runtime scope")
            if path in fit_files and obj["case"]["case_id"] in training[gpu]:
                continue
            vals = observations(obj); estimate = profile["outside_median_bytes"]
            heldout.append({"archive":path.name,"gpu":gpu,"case_id":obj["case"]["case_id"],
                "observed_case_median_bytes":int(statistics.median(vals)),
                "median_error_bytes":estimate-int(statistics.median(vals)),
                "min_sample_bytes":min(vals),"max_sample_bytes":max(vals),
                "outside_training_envelope":min(vals)<profile["outside_min_training_sample_bytes"] or max(vals)>profile["outside_max_training_sample_bytes"]})
    identity = hashlib.sha256(json.dumps(profiles,sort_keys=True).encode()).hexdigest()
    return {"schema_version":1,"calibration_id":"outside-"+identity[:16],
        "definition":"same-sample NVML process bytes minus PyTorch reserved bytes",
        "fit_policy":"median of nine FP16 condition01 case medians; upper bound from training samples only",
        "gpus":profiles}, {"calibration_id":"outside-"+identity[:16],"heldout":heldout}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fit",nargs="+",type=Path,required=True)
    p.add_argument("--validate",nargs="*",type=Path,default=[])
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--validation-output",type=Path,required=True)
    a=p.parse_args();profiles,report=build(a.fit,a.validate)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.validation_output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(profiles,ensure_ascii=False,indent=2)+"\n")
    a.validation_output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print("Calibration:",profiles["calibration_id"])
    for gpu,item in profiles["gpus"].items():
        errors=[abs(x["median_error_bytes"]) for x in report["heldout"] if x["gpu"]==gpu]
        print(gpu,"outside MiB",item["outside_median_bytes"]/2**20,
              "heldout cases",len(errors),"max median error MiB",max(errors,default=0)/2**20)


if __name__=="__main__":main()
