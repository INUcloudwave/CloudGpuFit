import importlib.util
import io
import json
from pathlib import Path
import tempfile
import tarfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('comparison',ROOT/'tools/compare_vram_predictions.py')
comparison=importlib.util.module_from_spec(spec);spec.loader.exec_module(comparison)


class ComparisonTests(unittest.TestCase):
    def fixture(self,root):
        pred=root/'predictions';pred.mkdir()
        obj={'case':{'case_id':'case','dtype':'fp32','model':'model'},'status':'ok',
            'environment':{'gpu':'NVIDIA L4','revision':'rev','torch':'test'},
            'actual_input_shapes':{'input_ids':[1,3]},'inference_peak_allocated_bytes':100,
            'inference_peak_reserved_bytes':200,'nvml_process_sampled_peak_bytes':230,
            'memory_samples':[{'nvml_process_bytes':230,'reserved_bytes':200}]}
        archive=root/'actual.tar.gz';raw=json.dumps(obj).encode()
        with tarfile.open(archive,'w:gz') as t:
            member=tarfile.TarInfo('results/case__memory__r0.json');member.size=len(raw);t.addfile(member,io.BytesIO(raw))
        profile={'calibration_id':'test','gpus':{'NVIDIA L4':{'outside_median_bytes':30,
            'outside_max_training_sample_bytes':35,'supported_dtypes':['fp32'],'runtime_scope':{'torch':'test'}}}}
        return pred,archive,profile

    def test_each_metric_is_compared_to_matching_actual(self):
        with tempfile.TemporaryDirectory() as temp:
            pred,archive,profile=self.fixture(Path(temp))
            (pred/'case.prediction.json').write_text(json.dumps({'status':'ok','case_id':'case','revision':'rev',
                'actual_input_shapes':{'input_ids':[1,3]},'predicted_allocated_bytes':100,'predicted_reserved_bytes':200}))
            rows,summary=comparison.compare(pred,[archive],profile)
            self.assertEqual(rows[0]['status'],'ok')
            self.assertTrue(all(x['mape_pct']==0 for x in summary))

    def test_missing_prediction_is_retained(self):
        with tempfile.TemporaryDirectory() as temp:
            pred,archive,profile=self.fixture(Path(temp))
            rows,summary=comparison.compare(pred,[archive],profile)
            self.assertEqual(rows[0]['status'],'missing_prediction')
            self.assertEqual(summary,[])


if __name__=='__main__':unittest.main()
