"""Native inference tensor lifetimes -> xMem allocator simulation.

Trace only the forward/generate region, not CPU RSS or data loading. Training
estimator heuristics (gradient residency, optimiser state, copied iterations)
are deliberately not used for inference.
"""
from __future__ import annotations
import json
from pathlib import Path
import tempfile


REGION = "cloudgpufit_inference"


def selected_lifetimes(events):
    regions=[(e['ts'],e['ts']+e['dur']) for e in events if e.get('name')==REGION and e.get('cat')=='user_annotation']
    if len(regions)!=3:
        raise ValueError(f"Expected exactly three measured inference regions, got {len(regions)}")
    selected=[];live={}
    for event in sorted(events,key=lambda e:e.get('ts',0)):
        if event.get('cat')!='cpu_instant_event' or event.get('name')!='[memory]':continue
        args=event['args'];size=args.get('Bytes',0)
        if args.get('Device Type')!=0 or size==0:continue
        address=args['Addr']
        if size>0:
            if any(begin<=event['ts']<=end for begin,end in regions):
                if address in live:raise ValueError('Allocation address reused without a matching free')
                live[address]=event
        elif address in live:
            alloc=live.pop(address)
            if alloc['args']['Bytes']!=-size:raise ValueError('Allocation/free size mismatch')
            selected.append((alloc,event))
    selected.extend((event,None) for event in live.values())
    if not selected:raise ValueError('No tensor allocations observed in the inference regions')
    return selected,regions


def trace_and_estimate(model, inputs, mode, new_tokens, storage_sizes, gpu_memory_gb):
    import torch
    from perf_estimator.allocator import AllocatorSim
    from perf_estimator.profiler import MemoryBlock, CpuInstantNode
    from perf_estimator.config import Config

    kv_max = {"self_kv_storage_bytes":0,"cross_kv_storage_bytes":0,
              "self_kv_logical_bytes":0,"cross_kv_logical_bytes":0}
    def observe_cache(module, args, output):
        cache=getattr(output,'past_key_values',None)
        if cache is None:return
        if hasattr(cache,'to_legacy_cache'):cache=cache.to_legacy_cache()
        self_tensors=[];cross_tensors=[]
        for layer in cache:
            self_tensors.extend(layer[:2])
            cross_tensors.extend(layer[2:4])
        for name,tensors in [('self_kv_storage_bytes',self_tensors),('cross_kv_storage_bytes',cross_tensors)]:
            kv_max[name]=max(kv_max[name],sum(storage_sizes(tensors)))
            logical=name.replace('_storage_', '_logical_')
            kv_max[logical]=max(kv_max[logical],sum(t.numel()*t.element_size() for t in tensors))

    def invoke():
        if mode=='generate':
            output=model.generate(**inputs,do_sample=False,num_beams=1,use_cache=True,
                min_new_tokens=new_tokens,max_new_tokens=new_tokens,
                pad_token_id=(model.config.pad_token_id if model.config.pad_token_id is not None else model.config.eos_token_id),
                return_dict_in_generate=False,output_scores=False,output_attentions=False,output_hidden_states=False)
            prefix=inputs.get('decoder_input_ids') if model.config.is_encoder_decoder else inputs['input_ids']
            prefix_len=prefix.shape[1] if prefix is not None else 1
            if output.shape[1]!=prefix_len+new_tokens:
                raise ValueError('Generated output length differs from the benchmark condition')
        elif 'pixel_values' in inputs:
            output=model(**inputs)
        else:
            output=model(**inputs,use_cache=False)
        del output

    model.eval()
    hook=model.register_forward_hook(observe_cache) if mode=='generate' else None
    try:
        with torch.inference_mode():
            invoke()  # The same one-request warmup as the shortened GPU benchmark.
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                    profile_memory=True,record_shapes=False,with_stack=False) as profiler:
                for _ in range(3):
                    with torch.profiler.record_function(REGION):invoke()
    finally:
        if hook is not None:hook.remove()
    with tempfile.TemporaryDirectory(prefix='xmem-inference-') as directory:
        trace=Path(directory)/'trace.json';profiler.export_chrome_trace(str(trace))
        events=json.loads(trace.read_text())['traceEvents']
    lifetimes,regions=selected_lifetimes(events)
    blocks=[]
    for alloc,free in lifetimes:
        block=MemoryBlock(CpuInstantNode(alloc))
        if free is not None:block.set_free_node(CpuInstantNode(free))
        blocks.append(block)
    def resident(size,address):
        # Start residency before any inference allocation; each tensor storage
        # keeps its own allocation granularity. No integer/bool dtype rescaling.
        event={'ph':'i','cat':'cpu_instant_event','s':'t','name':'[memory]',
               'pid':0,'tid':0,'ts':regions[0][0]-1,
               'args':{'Bytes':size,'Addr':address,'Device Id':-1,'Device Type':0,'Ev Idx':0}}
        return MemoryBlock(CpuInstantNode(event))
    weights=storage_sizes(list(model.parameters())+list(model.buffers()))
    initial_inputs=storage_sizes(list(inputs.values()))
    for index,size in enumerate(weights+initial_inputs):
        if size:blocks.append(resident(size,-index-1))
    blocks.sort(key=lambda b:b.alloc_time)
    result=AllocatorSim(max_allocated_memory_gb=gpu_memory_gb,config=Config(save2tmp=True)).simulate(blocks)
    if result.oom:raise RuntimeError('xMem simulation reached its configured memory ceiling')
    allocated=max(result._trace.max_usage_changes)
    reserved=max(result._trace.max_segment_changes)
    if allocated<sum(weights)+sum(initial_inputs):
        raise ValueError('Allocator trace lost resident model/input memory')
    return {'memory':{'tensor':allocated,'segment':reserved},'OOM':False,
        'trace_summary':{'measured_invocations':3,'warmup_invocations':1,
            **kv_max,
            'tensor_lifetime_count':len(lifetimes),'resident_storage_bytes':sum(weights),
            'input_storage_bytes':sum(initial_inputs),'cpu_device_only':True,
            'method':'native_inference_lifetimes_into_xmem_allocator',
            'cross_kv_policy':'traced_with_full_generate_no_analytic_addition'}}
