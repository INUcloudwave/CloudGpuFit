import torch
import torch.nn as nn
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
import operator
import pandas as pd
import lightgbm as lgb
import numpy as np
import math
import os
import json
from pathlib import Path
import torchvision.models as models
import gc
from transformers import BertModel, BertConfig, OPTModel, OPTConfig, T5Model, T5Config, Wav2Vec2Model, Wav2Vec2Config
from transformers import RobertaModel, RobertaConfig, Qwen2Model, Qwen2Config

pd.set_option('display.max_rows', None)


HARDWARE_PROFILE_FIELDS = (
    "peak_flops",
    "sm_count",
    "peak_bandwidth",
    "precision_bytes",
    "l2_cache_bytes",
    "t_h2d_base",
    "t_d2h_base",
    "bw_h2d_GBs",
    "bw_d2h_GBs",
    "t_dispatch_light",
    "t_dispatch_heavy",
    "t_sync",
    "t_python_loop",
    "t_pytorch_overhead",
)


def apply_hardware_profile(router):
    path_value = os.environ.get("EAGER_HARDWARE_PROFILE")
    if not path_value:
        raise RuntimeError(
            "EAGER_HARDWARE_PROFILE is not configured. "
            "Validate a target runtime manifest before creating the router."
        )
    path = Path(path_value)
    payload = json.loads(path.read_text(encoding="utf-8"))
    missing = [name for name in HARDWARE_PROFILE_FIELDS if name not in payload]
    if missing:
        raise ValueError(
            f"Hardware profile is missing fields in {path}: "
            + ", ".join(missing)
        )
    for name in HARDWARE_PROFILE_FIELDS:
        setattr(router, name, payload[name])
    router.bytes_per_ms_h2d = router.bw_h2d_GBs * 1e6
    router.bytes_per_ms_d2h = router.bw_d2h_GBs * 1e6
    router.host_schedule_scale = float(payload.get("host_schedule_scale", 1.0))
    router.host_schedule_policy = str(payload.get("host_schedule_policy", "none"))
    router.hardware_profile_path = path


def tensor_meta_elements(node):
    if not hasattr(node, "meta") or "tensor_meta" not in node.meta:
        return 0
    meta = node.meta["tensor_meta"]
    return math.prod(meta.shape) if hasattr(meta, "shape") else 0


def input_elements(value):
    if isinstance(value, torch.fx.Node):
        return tensor_meta_elements(value)
    if isinstance(value, (tuple, list)):
        return sum(input_elements(item) for item in value)
    if isinstance(value, dict):
        return sum(input_elements(item) for item in value.values())
    return 0


def tensor_input_count(value):
    if isinstance(value, torch.fx.Node):
        return int(tensor_meta_elements(value) > 0)
    if isinstance(value, (tuple, list)):
        return sum(tensor_input_count(item) for item in value)
    if isinstance(value, dict):
        return sum(tensor_input_count(item) for item in value.values())
    return 0


def make_light_node(op_type, node, output_elements, out_shape, module=None):
    inputs = input_elements(node.args)
    parameters = 0
    reduction_size = 1

    if module is not None and "Norm" in module.__class__.__name__:
        parameters = sum(value.numel() for value in module.parameters())
        parameters += sum(value.numel() for value in module.buffers())
        normalized_shape = getattr(module, "normalized_shape", None)
        if normalized_shape is not None:
            reduction_size = math.prod(normalized_shape)

    if op_type == "Embedding":
        parameters = output_elements
    elif op_type == "Softmax" and out_shape is not None:
        reduction_size = out_shape[-1]
    elif op_type in {"AdaptiveAvgPool2d", "MaxPool2d", "Reduction"}:
        reduction_size = max(1, inputs // max(output_elements, 1))

    return {
        "type": op_type,
        "elements": output_elements,
        "input_elements": inputs,
        "output_elements": output_elements,
        "parameter_elements": parameters,
        "reduction_size": reduction_size,
        "is_binary": int(tensor_input_count(node.args) >= 2),
        "is_adaptive_pool": int(op_type == "AdaptiveAvgPool2d"),
        "index_input": int(op_type == "Embedding"),
        "Is_Fused": 0,
    }

class FXGraphParser:
    def __init__(self):
        print("🔍 FX Graph Parser 초기화 중 (Analytical Fusion Group ⚡ 활성화)...")
        
    def trace_model(self, model, dummy_input, is_hf=False):
        model.eval()
        if is_hf:
            from transformers.utils.fx import symbolic_trace as hf_symbolic_trace
            input_names = list(dummy_input.keys()) if isinstance(dummy_input, dict) else ["input_ids"]
            gm = hf_symbolic_trace(model, input_names=input_names)
        else:
            from torch.fx import symbolic_trace
            gm = symbolic_trace(model)
        return gm

    def extract_nodes(self, gm, dummy_input, engine_mode="eager"):
        with torch.no_grad():
            if isinstance(dummy_input, dict):
                ShapeProp(gm).propagate(*dummy_input.values())
            else:
                ShapeProp(gm).propagate(dummy_input)
        return self._parse_nodes(gm, engine_mode)

    def apply_analytical_fusion_pass(self, fx_nodes):
        fused_graph = []
        # 💡 [업데이트] Qwen, RoBERTa의 커스텀 Norm 클래스도 Fusion 꼬리로 인식하도록 추가
        fusible_tails = ['ReLU', 'ReLU6', 'GELU', 'ElementWise', 'Reduction', 'Dropout', 'Softmax', 'Sum', 'MemoryFormat', 'Embedding']
        
        i = 0
        n = len(fx_nodes)
        while i < n:
            node = fx_nodes[i].copy()
            if node.get('type') == 'Ignore':
                i += 1; continue
                
            is_fused_base = 0
            base_ops_count = 1
            
            if node['type'] in ['Conv1d', 'Conv2d']:
                if i + 2 < n and fx_nodes[i+1]['type'] in ['BatchNorm1d', 'BatchNorm2d'] and fx_nodes[i+2]['type'] in ['ReLU', 'GELU']:
                    is_fused_base = 1; base_ops_count = 3
                elif i + 1 < n and fx_nodes[i+1]['type'] in ['BatchNorm1d', 'BatchNorm2d', 'ReLU', 'ReLU6', 'GELU']:
                    is_fused_base = 1; base_ops_count = 2
            elif node['type'] == 'Linear':
                if i + 1 < n and fx_nodes[i+1]['type'] in ['GELU', 'ReLU', 'ReLU6']:
                    is_fused_base = 1; base_ops_count = 2
            elif node['type'] == 'ElementWise':
                # 💡 [업데이트] Norm 계열을 범용적으로 매칭
                if i + 1 < n and "Norm" in fx_nodes[i+1]['type']:
                    node['type'] = 'ElementWise_LayerNorm'
                    is_fused_base = 1; base_ops_count = 2
                elif i + 1 < n and fx_nodes[i+1]['type'] in ['ReLU', 'ReLU6', 'GELU']:
                    is_fused_base = 1; base_ops_count = 2
            
            node['Is_Fused'] = is_fused_base
            
            fusion_group = {
                'type': 'FusionGroup',
                'base_op': node,
                'absorbed_base_ops': [fx_nodes[k]['type'] for k in range(i+1, i+base_ops_count)],
                'tail_ops': []
            }
            
            j = i + base_ops_count
            while j < n:
                tail_candidate = fx_nodes[j].copy()
                
                # 💡 [수정] "Norm"이라는 글자가 이름에 있으면 무조건 꼬리 융합(Tail)으로 흡수!
                is_tail = tail_candidate['type'] in fusible_tails or "Norm" in tail_candidate['type']
                
                if is_tail:
                    fusion_group['tail_ops'].append(tail_candidate)
                    j += 1
                else:
                    break
                    
            fused_graph.append(fusion_group)
            i = j
            
        return fused_graph

    def _parse_nodes(self, gm, engine_mode):
        fx_nodes = []
        modules = dict(gm.named_modules())

        for node in gm.graph.nodes:
            out_shape = None; elements = 1
            
            if 'tensor_meta' in node.meta:
                meta = node.meta['tensor_meta']
                if hasattr(meta, 'shape'): out_shape = meta.shape
                elif isinstance(meta, (tuple, list)) and len(meta) > 0 and hasattr(meta[0], 'shape'): out_shape = meta[0].shape
                elif hasattr(meta, 'values'):
                    for val in meta.values():
                        if hasattr(val, 'shape'): out_shape = val.shape; break
                if out_shape is not None:
                    for s in out_shape: elements *= s

            current_op = None

            if node.op == 'call_module':
                target_module = modules[node.target]
                if isinstance(target_module, nn.Conv2d):
                    B, C_out, H_out, W_out = out_shape
                    groups = target_module.groups
                    K = (target_module.in_channels // groups) * target_module.kernel_size[0] * target_module.kernel_size[1]
                    current_op = {'type': 'Conv2d', 'M': B * H_out * W_out, 'N': C_out, 'K': K, 'kernel_size': target_module.kernel_size[0], 'Is_Conv1D': 0, 'Is_Conv2D': 1, 'Is_BMM': 0, 'Is_Fused': 0}
                elif isinstance(target_module, nn.Conv1d):
                    B, C_out, L_out = out_shape
                    groups = target_module.groups
                    K = (target_module.in_channels // groups) * target_module.kernel_size[0]
                    current_op = {'type': 'Conv1d', 'M': B * L_out, 'N': C_out, 'K': K, 'kernel_size': target_module.kernel_size[0], 'Is_Conv1D': 1, 'Is_Conv2D': 0, 'Is_BMM': 0, 'Is_Fused': 0}
                elif isinstance(target_module, nn.Linear):
                    N = target_module.out_features; K = target_module.in_features; M = elements // N 
                    current_op = {'type': 'Linear', 'M': M, 'N': N, 'K': K, 'kernel_size': 1, 'Is_Conv1D': 0, 'Is_Conv2D': 0, 'Is_BMM': 0, 'Is_Fused': 0}
                # 💡 [업데이트] 이름에 "Norm"이 들어간 클래스(Qwen2RMSNorm 등)를 모두 동적으로 캡처
                elif isinstance(target_module, (nn.ReLU, nn.ReLU6, nn.GELU, nn.BatchNorm2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d)) or "Norm" in target_module.__class__.__name__:
                    current_op = make_light_node(
                        target_module.__class__.__name__, node, elements, out_shape, target_module
                    )
                elif isinstance(target_module, (nn.Dropout, nn.Embedding)):
                    current_op = make_light_node(
                        target_module.__class__.__name__, node, elements, out_shape, target_module
                    )

            elif node.op == 'call_function':
                if node.target is torch.addmm:
                    matrix_a = node.args[1] if len(node.args) > 1 else None
                    matrix_b = node.args[2] if len(node.args) > 2 else None
                    meta_a = getattr(matrix_a, 'meta', {}).get('tensor_meta')
                    meta_b = getattr(matrix_b, 'meta', {}).get('tensor_meta')
                    if meta_a is not None and meta_b is not None:
                        shape_a = meta_a.shape
                        shape_b = meta_b.shape
                        if len(shape_a) == 2 and len(shape_b) == 2:
                            M, K = int(shape_a[0]), int(shape_a[1])
                            N = int(shape_b[1])
                            if int(shape_b[0]) == K:
                                current_op = {
                                    'type': 'Linear', 'M': M, 'N': N, 'K': K,
                                    'kernel_size': 1, 'Is_Conv1D': 0,
                                    'Is_Conv2D': 0, 'Is_BMM': 0, 'Is_Fused': 0,
                                    'RHS_Is_Transposed': 0,
                                }
                elif node.target in [torch.bmm, torch.matmul, operator.matmul]:
                    if hasattr(node.args[0], 'meta') and 'tensor_meta' in node.args[0].meta:
                        shape_A = node.args[0].meta['tensor_meta'].shape
                        K = shape_A[-1]; N = out_shape[-1]; M_true = shape_A[-2] if len(shape_A) >= 2 else 1
                        B_dim = 1
                        for s in shape_A[:-2]: B_dim *= s
                        is_bmm_flag = 1 if len(shape_A) >= 3 else 0
                        current_op = {'type': 'BMM', 'M': B_dim * M_true, 'N': N, 'K': K, 'M_true': M_true, 'B_dim': B_dim, 'kernel_size': 1, 'Is_Conv1D': 0, 'Is_Conv2D': 0, 'Is_BMM': is_bmm_flag, 'Is_Fused': 0}
                elif node.target in [torch.nn.functional.softmax]:
                    current_op = make_light_node('Softmax', node, elements, out_shape)
                elif node.target in [operator.add, torch.add, operator.sub, torch.sub, operator.mul, torch.mul, operator.truediv, torch.div]:
                    current_op = make_light_node('ElementWise', node, elements, out_shape)
                elif node.target in [torch.mean, torch.sum]:
                    current_op = make_light_node('Reduction', node, elements, out_shape)
                elif node.target in [torch.sqrt, torch.rsqrt, torch.pow, torch.exp]:
                    current_op = make_light_node('ElementWise', node, elements, out_shape)
                elif node.target is torch.nn.functional.scaled_dot_product_attention:
                    q_meta, k_meta, v_meta = node.args[0].meta.get('tensor_meta'), node.args[1].meta.get('tensor_meta'), node.args[2].meta.get('tensor_meta')
                    if q_meta and k_meta and v_meta:
                        shape_q, shape_k = q_meta.shape, k_meta.shape
                        B_dim = shape_q[0] * shape_q[1]; M_true_1, K1, N1 = shape_q[2], shape_q[3], shape_k[2]
                        current_op = {'type': 'BMM_SDPA', 'M': B_dim * M_true_1, 'N': N1, 'K': K1, 'M_true': M_true_1, 'B_dim': B_dim, 'kernel_size': 1, 'Is_Conv1D': 0, 'Is_Conv2D': 0, 'Is_BMM': 1, 'Is_Fused': 1}
                elif node.target in [torch.nn.functional.gelu, torch.nn.functional.relu, torch.nn.functional.relu6]:
                    if node.target is torch.nn.functional.gelu:
                        op_name = 'GELU'
                    elif node.target is torch.nn.functional.relu6:
                        op_name = 'ReLU6'
                    else:
                        op_name = 'ReLU'
                    current_op = make_light_node(op_name, node, elements, out_shape)
            
            elif node.op == 'call_method':
                if str(node.target) in ['view', 'reshape', 'transpose', 'permute', 'contiguous', 'clone']:
                    current_op = make_light_node('MemoryFormat', node, elements, out_shape)
                elif str(node.target) in ['mean', 'sum']:
                    current_op = make_light_node('Reduction', node, elements, out_shape)
                elif str(node.target) in ['pow', 'rsqrt', 'mul', 'add', 'div', 'sqrt', 'unsqueeze', 'squeeze', 'expand']:
                    current_op = make_light_node('ElementWise', node, elements, out_shape)

            if current_op is None: continue
            fx_nodes.append(current_op)

        if engine_mode == "compiled": return self.apply_analytical_fusion_pass(fx_nodes)
        return [{'type': 'FusionGroup', 'base_op': node, 'absorbed_base_ops': [], 'tail_ops': []} for node in fx_nodes]


class SimulatorRouter:
    def __init__(self, engine_mode="eager"):
        print(f"🚀 CloudSim Router 초기화 중 (모드: {engine_mode.upper()})...")
        self.engine_mode = engine_mode
        try:
            clean_mode = "eager"
            model_dir = Path(os.environ.get("EAGER_BASE_MODEL_DIR", "."))
            self.gemm_median = lgb.Booster(
                model_file=str(model_dir / f"gemm_{clean_mode}_median.pkl")
            )
            self.gemm_90th = lgb.Booster(
                model_file=str(model_dir / f"gemm_{clean_mode}_90th.pkl")
            )
            self.red_median = lgb.Booster(
                model_file=str(model_dir / f"reduction_{clean_mode}_median.pkl")
            )
            self.red_90th = lgb.Booster(
                model_file=str(model_dir / f"reduction_{clean_mode}_90th.pkl")
            )
        except Exception as e:
            raise e
            
        apply_hardware_profile(self)
        if self.engine_mode == "compiled":
            self.t_python_loop = 0.0
            self.t_pytorch_overhead = 0.0

    def simulate_e2e_pipeline(self, log_df, input_bytes, output_bytes):
        h2d_total_ms = self.t_h2d_base + (input_bytes / self.bytes_per_ms_h2d)
        d2h_total_ms = self.t_d2h_base + (output_bytes / self.bytes_per_ms_d2h)
        cpu_time, gpu_time = self.t_pytorch_overhead, h2d_total_ms
        
        if self.engine_mode == "compiled":
            cpu_time += self.t_dispatch_heavy
            gpu_time_pure = log_df["Predicted_Actual_ms"].sum()

            # H2D와 compiled graph dispatch가 끝난 뒤 kernel 실행
            gpu_time = max(h2d_total_ms, cpu_time) + gpu_time_pure
        else:
            for _, row in log_df.iterrows():
                is_heavy = row['Layer_Type'] in ['Conv2d', 'Conv1d', 'Linear', 'BMM', 'BMM_SDPA']
                cpu_time += self.t_python_loop + (self.t_dispatch_heavy if is_heavy else self.t_dispatch_light)
                gpu_time = max(gpu_time, cpu_time) + row['Predicted_Actual_ms']

        gpu_time += d2h_total_ms
        return gpu_time + self.t_sync, []

    def _predict_node(self, node, batch_size, prediction_mode, force_memory_zero=False):
        op_type = node.get('type')
        actual_time_ms = 0.0; theoretical_time_ms = 0.0
        
        if op_type in ['Conv2d', 'Conv1d', 'Linear', 'BMM', 'BMM_SDPA']:
            M, N, K = node['M'], node['N'], node['K']
            is_bmm, is_fused = node.get('Is_BMM', 0), node.get('Is_Fused', 0)  
            
            total_flops = 2 * M * N * K
            if is_bmm and is_fused: total_flops *= 2
            
            B_dim, M_true = node.get('B_dim', 1), node.get('M_true', M)
            ideal_memory_bytes = 0 if force_memory_zero else (B_dim * (M_true * K + K * N + M_true * N) * self.precision_bytes if is_bmm else (M * K + K * N + M * N) * self.precision_bytes)
            
            arithmetic_intensity = total_flops / ideal_memory_bytes if ideal_memory_bytes > 0 else 0
            estimated_l2_fit = 1 if ideal_memory_bytes <= self.l2_cache_bytes else 0
            estimated_wave_count = (math.ceil(M / 128.0) * math.ceil(N / 128.0)) / self.sm_count

            theoretical_time_ms = max((total_flops / self.peak_flops) * 1000, (ideal_memory_bytes / self.peak_bandwidth) * 1000, 1e-6)

            features = pd.DataFrame([{
                'Batch_Size': batch_size, 'Dim_M': M, 'Dim_N': N, 'Dim_K': K, 'Total_FLOPs': total_flops,
                'Ideal_Memory_Bytes': ideal_memory_bytes, 'Arithmetic_Intensity': arithmetic_intensity,
                'Estimated_L2_Fit': estimated_l2_fit, 'Orig_Kernel_Size': node.get('kernel_size', 1),
                'Precision_Bytes': self.precision_bytes, 'Is_Conv1D': node.get('Is_Conv1D', 0),
                'Is_Conv2D': node.get('Is_Conv2D', 0), 'Is_BMM': is_bmm, 'Is_Fused': is_fused,
                'Estimated_Wave_Count': estimated_wave_count
            }])
            y_ratio_log = self.gemm_median.predict(features)[0] if prediction_mode == "median" else self.gemm_90th.predict(features)[0]
            actual_time_ms = theoretical_time_ms * np.expm1(y_ratio_log)

        else:
            output_elements = node.get('output_elements', node.get('elements', 0))
            input_elements = node.get('input_elements', output_elements)
            parameter_elements = node.get('parameter_elements', 0)
            reduction_size = node.get('reduction_size', 1)
            is_binary = node.get('is_binary', 0)
            is_adaptive_pool = node.get('is_adaptive_pool', 0)
            
            is_pool = 1 if "Pool" in op_type else 0
            is_norm = 1 if "Norm" in op_type else 0  # 💡 [업데이트] Qwen, RoBERTa의 Norm 포함
            is_embedding = 1 if "Embedding" in op_type else 0
            is_softmax = 1 if "Softmax" in op_type else 0
            is_elementwise = 1 if not (is_pool or is_norm or is_embedding or is_softmax) else 0

            if node.get('index_input', 0):
                read_bytes = input_elements * 8 + parameter_elements * self.precision_bytes
            else:
                read_bytes = (input_elements + parameter_elements) * self.precision_bytes
            write_bytes = output_elements * self.precision_bytes
            ideal_memory_bytes = read_bytes + write_bytes

            if force_memory_zero:
                read_bytes = 0
                write_bytes = 0
                ideal_memory_bytes = 0
            
            estimated_l2_fit = 1 if ideal_memory_bytes <= self.l2_cache_bytes else 0
            estimated_wave_count = math.ceil(output_elements / 1024.0) / self.sm_count
            theoretical_time_ms = max((ideal_memory_bytes / self.peak_bandwidth) * 1000, 1e-6)

            features = pd.DataFrame([{
                'Batch_Size': batch_size, 'Total_Elements': output_elements,
                'Input_Elements': input_elements, 'Output_Elements': output_elements,
                'Parameter_Elements': parameter_elements, 'Read_Bytes': read_bytes,
                'Write_Bytes': write_bytes, 'Reduction_Size': reduction_size,
                'Ideal_Memory_Bytes': ideal_memory_bytes,
                'Estimated_L2_Fit': estimated_l2_fit, 'Precision_Bytes': self.precision_bytes,
                'Is_Fused': node.get('Is_Fused', 0), 'Estimated_Wave_Count': estimated_wave_count,
                'Is_ElementWise': is_elementwise, 'Is_Pool': is_pool, 
                'Is_Norm': is_norm, 'Is_Softmax': is_softmax, 'Is_Embedding': is_embedding,
                'Is_Binary': is_binary, 'Is_AdaptivePool': is_adaptive_pool
            }])
            y_ratio_log = self.red_median.predict(features)[0] if prediction_mode == "median" else self.red_90th.predict(features)[0]
            actual_time_ms = theoretical_time_ms * np.expm1(y_ratio_log)
            
        return actual_time_ms, theoretical_time_ms
        
    def simulate_fx_graph(self, fused_graph, batch_size, prediction_mode="median"):
        total_time_ms, simulation_log = 0.0, []
        for group in fused_graph:
            if group['type'] != 'FusionGroup': continue
            
            base_node = group['base_op']
            base_actual_ms, base_theo_ms = self._predict_node(base_node, batch_size, prediction_mode, force_memory_zero=False)
            group_total_actual_ms = base_actual_ms
            group_total_theo_ms = base_theo_ms
            
            for tail_node in group['tail_ops']:
                tail_actual_ms, tail_theo_ms = self._predict_node(tail_node, batch_size, prediction_mode, force_memory_zero=True)
                group_total_actual_ms += tail_actual_ms
                group_total_theo_ms += tail_theo_ms
                
            total_time_ms += group_total_actual_ms
            
            log_name = base_node['type']
            if base_node.get('Is_Fused', 0) == 1: log_name += f" + {'+'.join(group['absorbed_base_ops'])}"
            if len(group['tail_ops']) > 0: log_name += f" (+ Tail: {len(group['tail_ops'])} ops)"
                
            simulation_log.append({
                'Layer_Type': base_node['type'], 
                'Log_Name': log_name,            
                'Theoretical_ms': round(group_total_theo_ms, 4), 
                'Predicted_Actual_ms': round(group_total_actual_ms, 4)
            })
        return total_time_ms, pd.DataFrame(simulation_log)

HEAVY_TYPES = {
    "Conv2d",
    "Conv1d",
    "Linear",
    "BMM",
    "BMM_SDPA",
}

BREAKDOWN_FAMILIES = [
    "conv1d",
    "conv2d",
    "linear",
    "attention",
    "reduction",
    "embedding",
    "memory",
    "pointwise",
]


def prediction_family(layer_type):
    if layer_type == "Conv1d":
        return "conv1d"

    if layer_type == "Conv2d":
        return "conv2d"

    if layer_type == "Linear":
        return "linear"

    if layer_type in {"BMM", "BMM_SDPA"}:
        return "attention"

    if (
        "Norm" in layer_type
        or "Pool" in layer_type
        or layer_type in {"Softmax", "Sum"}
    ):
        return "reduction"

    if "Embedding" in layer_type:
        return "embedding"

    if layer_type == "MemoryFormat":
        return "memory"

    return "pointwise"


def build_prediction_breakdown(
    router,
    model_name,
    batch_size,
    mode,
    log_df,
    input_bytes,
    output_bytes,
    predicted_e2e_ms,
):
    h2d_ms = (
        router.t_h2d_base
        + input_bytes / router.bytes_per_ms_h2d
    )
    d2h_ms = (
        router.t_d2h_base
        + output_bytes / router.bytes_per_ms_d2h
    )
    kernel_sum_ms = float(
        log_df["Predicted_Actual_ms"].sum()
    )

    heavy_mask = log_df["Layer_Type"].isin(HEAVY_TYPES)
    heavy_count = int(heavy_mask.sum())
    light_count = int((~heavy_mask).sum())

    if mode == "compiled":
        host_enqueue_raw_ms = (
            router.t_pytorch_overhead
            + router.t_dispatch_heavy
        )
    else:
        host_enqueue_raw_ms = router.t_pytorch_overhead
        host_enqueue_raw_ms += (
            heavy_count
            * (
                router.t_python_loop
                + router.t_dispatch_heavy
            )
        )
        host_enqueue_raw_ms += (
            light_count
            * (
                router.t_python_loop
                + router.t_dispatch_light
            )
        )

    fixed_pipeline_ms = (
        h2d_ms
        + kernel_sum_ms
        + d2h_ms
        + router.t_sync
    )
    exposed_schedule_ms = (
        predicted_e2e_ms - fixed_pipeline_ms
    )

    record = {
        "model": model_name,
        "batch_size": batch_size,
        "mode": mode,
        "node_count": len(log_df),
        "heavy_count": heavy_count,
        "light_count": light_count,
        "kernel_sum_ms": kernel_sum_ms,
        "h2d_ms": h2d_ms,
        "d2h_ms": d2h_ms,
        "sync_ms": router.t_sync,
        "host_enqueue_raw_ms": host_enqueue_raw_ms,
        "exposed_schedule_ms": exposed_schedule_ms,
        "predicted_non_kernel_ms": (
            predicted_e2e_ms - kernel_sum_ms
        ),
        "predicted_e2e_ms": predicted_e2e_ms,
    }

    families = log_df["Layer_Type"].map(
        prediction_family
    )

    for family in BREAKDOWN_FAMILIES:
        mask = families == family
        record[f"{family}_count"] = int(mask.sum())
        record[f"{family}_ms"] = float(
            log_df.loc[
                mask,
                "Predicted_Actual_ms",
            ].sum()
        )

    return record
        
def run_all_simulations(target_mode="eager"):
    parser = FXGraphParser()
    router = SimulatorRouter(engine_mode=target_mode)
    batch_sizes = [1, 8, 16, 32, 64, 128]
    breakdown_rows = []

    models_config = {
        "MobileNetV2": {
            "is_hf": False,
            "load_fn": lambda: models.mobilenet_v2().half(),
            "dummy_fn": lambda bs: torch.randn(bs, 3, 224, 224, dtype=torch.float16),
            "bytes_fn": lambda bs, precision: (bs * 3 * 224 * 224 * precision, bs * 1000 * precision)
        },
        # 💡 [업데이트] RoBERTa-Base 추가
        "RoBERTa-Base": {
            "is_hf": True,
            "load_fn": lambda: RobertaModel(RobertaConfig(vocab_size=50265, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)).half(),
            "dummy_fn": lambda bs: torch.randint(0, 50265, (bs, 128)),
            "bytes_fn": lambda bs, precision: (bs * 128 * 4, bs * 128 * 768 * precision)
        },
        # 💡 [업데이트] Qwen2-1.5B 추가
        "Qwen2-1.5B": {
            "is_hf": True,
            "load_fn": lambda: Qwen2Model(Qwen2Config(vocab_size=151936, hidden_size=1536, intermediate_size=8960, num_hidden_layers=28, num_attention_heads=12, num_key_value_heads=2, use_cache=False)).half(),
            "dummy_fn": lambda bs: torch.randint(0, 151936, (bs, 128)),
            "bytes_fn": lambda bs, precision: (bs * 128 * 4, bs * 128 * 1536 * precision)
        },
        "ResNet-18": {
            "is_hf": False,
            "load_fn": lambda: models.resnet18().half(),
            "dummy_fn": lambda bs: torch.randn(bs, 3, 224, 224, dtype=torch.float16),
            "bytes_fn": lambda bs, precision: (bs * 3 * 224 * 224 * precision, bs * 1000 * precision)
        },
        "BERT-Base": {
            "is_hf": True,
            "load_fn": lambda: BertModel(BertConfig(vocab_size=30522, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)).half(),
            "dummy_fn": lambda bs: torch.randint(0, 30522, (bs, 128)),
            "bytes_fn": lambda bs, precision: (bs * 128 * 4, bs * 128 * 768 * precision)
        },
        "OPT-125m": {
            "is_hf": True,
            "load_fn": lambda: OPTModel(OPTConfig(vocab_size=50272, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, ffn_dim=3072)).half(),
            "dummy_fn": lambda bs: torch.randint(0, 50272, (bs, 128)),
            "bytes_fn": lambda bs, precision: (bs * 128 * 4, bs * 128 * 768 * precision)
        },
        "T5-Small": {
            "is_hf": True,
            "load_fn": lambda: T5Model(T5Config(vocab_size=32128, d_model=512, d_kv=64, d_ff=2048, num_layers=6, num_decoder_layers=6, num_heads=8, use_cache=False)).half(),
            "dummy_fn": lambda bs: {"input_ids": torch.randint(0, 32128, (bs, 128)), "decoder_input_ids": torch.randint(0, 32128, (bs, 128))},
            "bytes_fn": lambda bs, precision: ((bs * 128 * 4) * 2, bs * 128 * 512 * precision)
        },
        "Wav2Vec2": {
            "is_hf": True,
            "load_fn": lambda: Wav2Vec2Model(Wav2Vec2Config(vocab_size=32, hidden_size=768, num_hidden_layers=6, num_attention_heads=8, conv_dim=(512, 512, 512, 512, 512, 512, 512))),
            "dummy_fn": lambda bs: {"input_values": torch.randn(bs, 32000)},
            "bytes_fn": lambda bs, precision: (bs * 32000 * precision, bs * (32000 // 320) * 768 * precision)
        }
    }

    print("=" * 70); print(f"🚀 통합 시뮬레이션 시작 (모드: {target_mode.upper()})"); print("=" * 70)

    for model_name, config in models_config.items():
        print(f"\n📦 [{model_name}] 모델 로드 및 예측 시작...")
        model = config["load_fn"]().eval()

        print(f"{'Batch Size':<10} | {'예상 성능 (Median)':<20} | {'안전 보장 (90th)':<20}"); print("-" * 65)

        dummy_for_trace = config["dummy_fn"](1)
        gm = parser.trace_model(model, dummy_input=dummy_for_trace, is_hf=config["is_hf"])
        model, gm = model.half().cuda(), gm.half().cuda()
        
        for bs in batch_sizes:
            dummy_input = config["dummy_fn"](bs)
            if isinstance(dummy_input, dict): dummy_input = {k: v.half().cuda() if v.is_floating_point() else v.cuda() for k, v in dummy_input.items()}
            else: dummy_input = dummy_input.half().cuda() if dummy_input.is_floating_point() else dummy_input.cuda()
            
            input_bytes, output_bytes = config["bytes_fn"](bs, router.precision_bytes)
            extracted_nodes = parser.extract_nodes(gm, dummy_input, engine_mode=target_mode)
            
            _, median_log_df = router.simulate_fx_graph(extracted_nodes, batch_size=bs, prediction_mode="median")
            _, worst_log_df = router.simulate_fx_graph(extracted_nodes, batch_size=bs, prediction_mode="90th")
            
            median_e2e_ms, _ = router.simulate_e2e_pipeline(median_log_df, input_bytes, output_bytes)
            breakdown_record = build_prediction_breakdown(
                router=router,
                model_name=model_name,
                batch_size=bs,
                mode=target_mode,
                log_df=median_log_df,
                input_bytes=input_bytes,
                output_bytes=output_bytes,
                predicted_e2e_ms=median_e2e_ms,
            )
            raw_90th_e2e_ms, _ = router.simulate_e2e_pipeline(worst_log_df, input_bytes, output_bytes)

            existing_p90_ms = max(median_e2e_ms * 1.10, raw_90th_e2e_ms)
            finalize_p90 = getattr(router, "finalize_p90_prediction", None)
            if finalize_p90 is None:
                conservative_upper_bound_ms = existing_p90_ms
            else:
                conservative_upper_bound_ms = finalize_p90(
                    median_e2e_ms=median_e2e_ms,
                    raw_model_p90_ms=raw_90th_e2e_ms,
                )
            breakdown_record.update(
                {
                    "raw_model_p90_ms": raw_90th_e2e_ms,
                    "existing_p90_ms": existing_p90_ms,
                    "predicted_p90_ms": conservative_upper_bound_ms,
                }
            )
            breakdown_rows.append(breakdown_record)
            print(f"{bs:<10} | {median_e2e_ms:>10.3f} ms         | {conservative_upper_bound_ms:>16.3f} ms")

        del model, gm, dummy_for_trace
        gc.collect(); torch.cuda.empty_cache()
            
    print(f"\n✅ {target_mode.upper()} 모드 시뮬레이션 종료.\n")

    breakdown_path = globals().get(
        "breakdown_output_path",
        f"fx_breakdown_{target_mode}.csv",
    )

    pd.DataFrame(breakdown_rows).to_csv(
        breakdown_path,
        index=False,
    )

    print(f"Saved breakdown: {breakdown_path}")

if __name__ == "__main__":
    run_all_simulations(target_mode="eager")
