#!/usr/bin/env python3
"""Compatibility entry point: nine pinned FP16 conditions using the v3 predictor."""
import sys
from run_prediction_suite import main
if __name__ == "__main__":
    if "--case-set" in sys.argv:
        raise SystemExit("This entry point is baseline9 only; use run_prediction_suite.py for all conditions")
    sys.argv.extend(["--case-set", "baseline9"])
    raise SystemExit(main())
