#!/usr/bin/env python3
"""Offline tiny-model integrations through actual torch, HF and xMem.

Run inside the prepared /work/xMem directory with the same Python environment.
No Hugging Face weights/config downloads and no GPU are used.
"""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
CONFIGS={
 'decoder_llm':{'model_type':'gpt2','n_layer':1,'n_head':2,'n_embd':16,'vocab_size':32,'n_positions':64,'eos_token_id':2,'pad_token_id':2},
 'text_encdec':{'model_type':'t5','d_model':16,'d_kv':8,'d_ff':32,'num_layers':1,'num_decoder_layers':1,'num_heads':2,'vocab_size':32,'decoder_start_token_id':0,'eos_token_id':1,'pad_token_id':0},
 'vision_cls':{'model_type':'vit','hidden_size':16,'num_hidden_layers':1,'num_attention_heads':2,'intermediate_size':32,'image_size':16,'patch_size':8,'num_channels':3,'num_labels':2},
 'vision_cls:resnet':{'model_type':'resnet','embedding_size':8,'hidden_sizes':[8,16,32,64],'depths':[1,1,1,1],'layer_type':'basic','num_channels':3,'num_labels':2},
 'vision_cls:mobilenet_v2':{'model_type':'mobilenet_v2','depth_multiplier':0.1,'min_depth':8,'output_stride':16,'num_channels':3,'num_labels':2},
}


def main():
    # Run from the xMem checkout so its absolute package imports resolve.
    sys.path.insert(0,str(ROOT/'tools'))
    from run_prediction_suite import prediction_json
    results=[]
    for kind,config in CONFIGS.items():
        workload=kind.split(':')[0]
        for dtype,width in [('fp32',4),('fp16',2),('bf16',2)]:
            payload={'model':'offline-tiny-'+kind,'model_config':config,'workload_type':workload,
                'dtype':dtype,'batch_size':1,'seq_len':8,'encoder_seq_len':8,'decoder_seq_len':3,
                'resolution':16 if kind=='vision_cls' else 32,'mode':'prefill' if workload=='vision_cls' else 'generate',
                'max_new_tokens':3,'threads':2}
            run=subprocess.run([sys.executable,'predict_vram.py','--json','-'],input=json.dumps(payload),
                text=True,capture_output=True)
            result=prediction_json(run.stdout)
            if run.returncode or result['status']!='ok':
                raise RuntimeError(kind+'/'+dtype+' failed: '+run.stdout+'\n'+run.stderr)
            trace=result['trace_summary']
            assert trace['measured_invocations']==1
            assert trace['warmup_invocations']==0
            assert trace['request_policy']=='cold_single_request'
            assert result['predicted_allocated_bytes']>=result['model_storage_bytes']
            assert result['predicted_reserved_bytes']>=result['predicted_allocated_bytes']
            assert result['dtype_policy']=='native_cpu_no_rescale'
            if kind=='text_encdec':
                assert trace['self_kv_logical_bytes']==2*1*2*8*(3+3-1)*width,trace
                assert trace['cross_kv_logical_bytes']==2*1*2*8*8*width,trace
                assert trace['cross_kv_storage_bytes']>=trace['cross_kv_logical_bytes']
            elif kind=='decoder_llm':
                assert trace['self_kv_logical_bytes']==2*1*2*8*(8+3-1)*width,trace
                # GPT-2 caches may be views into a larger shared QKV storage.
                assert trace['self_kv_storage_bytes']>=trace['self_kv_logical_bytes']
                assert trace['cross_kv_storage_bytes']==0
            else:
                assert trace['self_kv_storage_bytes']==trace['cross_kv_storage_bytes']==0
            results.append({'workload':kind,'dtype':dtype,'allocated':result['predicted_allocated_bytes'],
                'reserved':result['predicted_reserved_bytes'],'self_kv':trace['self_kv_storage_bytes'],
                'cross_kv':trace['cross_kv_storage_bytes']})
            print('PASS',kind,dtype,flush=True)
    print(json.dumps({'status':'complete','cases':results},indent=2))


if __name__=='__main__':main()
