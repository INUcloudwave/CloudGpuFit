import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]


class CandidateMemoryTests(unittest.TestCase):
    def module(self):
        fake=types.ModuleType('eager_runtime')
        fake.available_targets=lambda:[]
        fake.target_provenance=lambda _: {'status':'calibrated'}
        spec=importlib.util.spec_from_file_location('memory_filter_test',ROOT/'orchestrator/instance_filter.py')
        module=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{'eager_runtime':fake,'memory_filter_test':module}):
            spec.loader.exec_module(module)
        return module

    def test_each_gpu_requirement_used_instead_of_global_maximum(self):
        module=self.module()
        specs={'g4dn.test':{'gpu_name':'NVIDIA T4','gpu_vram_gb':100,'gpu_count':1,'vcpu':8},
               'g6.test':{'gpu_name':'NVIDIA L4','gpu_vram_gb':100,'gpu_count':1,'vcpu':8}}
        catalog=[{'instance_type':name,'aws_region':'test','price_usd_per_hour':1} for name in specs]
        required={'NVIDIA T4':{'status':'ok','required_vram_bytes':120*1024**3},
                  'NVIDIA L4':{'status':'ok','required_vram_bytes':80*1024**3}}
        values=module.filter_instances(150,'test',None,catalog,supported=specs,gpu_requirements=required)
        self.assertEqual([r['instance_type'] for r in values],['g6.test'])

    def test_unknown_and_unsupported_gpu_not_assigned_zero(self):
        module=self.module()
        specs={'test':{'gpu_name':'NVIDIA L40S','gpu_vram_gb':100,'gpu_count':1,'vcpu':8}}
        catalog=[{'instance_type':'test','aws_region':'test','price_usd_per_hour':1}]
        for requirements in [{},{'NVIDIA L40S':{'status':'unsupported_dtype'}}]:
            self.assertEqual(module.filter_instances(1,'test',None,catalog,supported=specs,gpu_requirements=requirements),[])


if __name__=='__main__':unittest.main()
