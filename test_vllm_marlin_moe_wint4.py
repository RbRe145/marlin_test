# SPDX-License-Identifier: Apache-2.0
"""Tests for the MOE layers.

Run `pytest tests/kernels/test_moe.py`.
"""

import os
import numpy as np
import pytest
import torch
from torch.nn import Parameter
from torch.nn import functional as F
from transformers import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

import vllm.model_executor.layers.fused_moe  # noqa
from tests.kernels.utils import opcheck, stack_and_dev, torch_moe
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import fused_moe
from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.moe_torch_iterative import (
    fused_moe as iterative_moe)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    rand_marlin_weight_fp4_like)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    marlin_quant_fp8_torch)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    awq_marlin_quantize, marlin_quantize)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    quantize_weights)
from vllm.model_executor.models.mixtral import MixtralMoE
from vllm.platforms import current_platform
from vllm.scalar_type import ScalarType, scalar_types

NUM_EXPERTS = [8, 64]
EP_SIZE = [1, 4]
TOP_KS = [2, 6]

vllm_config = VllmConfig()
vllm_config.scheduler_config.max_num_seqs = 128
vllm_config.scheduler_config.max_model_len = 8192


def dump(x, name):
    if x is None:
        print(f"-- {name}: None")

    dump_dir = os.path.dirname(os.path.abspath(__file__)) + "/dump"
    if isinstance(x, torch.Tensor):
        if x.dtype in [torch.float32, torch.float16, torch.int, torch.int64]:
            y = x.cpu().numpy()
        elif x.dtype == torch.bfloat16:
            y = x.view(torch.uint16).cpu().numpy()
        else:
            assert False, f'{name}: {x.dtype} {x}'
        print(f"-- {name}: shape={x.size()}, dtype={x.dtype}, dump to {dump_dir}/{name}.npy")
        np.save(f"{dump_dir}/{name}.npy",y)


def marlin_moe_generate_valid_test_cases():
    import itertools
    m_list = [95]
    n_list = [256]
    k_list = [7168]
    e_list = [256]
    topk_list = [8]
    ep_size_list = [1]
    dtype_list = [torch.float16]
    group_size_list = [-1]# -1, 16, 32, 64, 128
    act_order_list = [False]
    quant_type_list = [
        # scalar_types.float4_e2m1f,
        # scalar_types.float8_e4m3fn,
        scalar_types.uint4,
        # scalar_types.uint4b8,
        # scalar_types.uint8b128,
    ]
    is_k_full_list = [True]

    all_combinations = itertools.product(m_list, n_list, k_list, e_list,
                                         topk_list, ep_size_list, dtype_list,
                                         group_size_list, act_order_list,
                                         quant_type_list, is_k_full_list)

    def is_invalid(m, n, k, e, topk, ep_size, dtype, group_size, act_order,
                   quant_type, is_k_full):

        if quant_type == scalar_types.float8_e4m3fn and \
                group_size not in [-1, 128]:
            return False
        if quant_type == scalar_types.float4_e2m1f and group_size != 16:
            return False
        if quant_type != scalar_types.float4_e2m1f and group_size == 16:
            return False

        # Filter act_order
        if act_order:
            if group_size in (-1, k, n):
                return False
            if quant_type not in [scalar_types.uint4b8]:
                return False
        elif not is_k_full:
            return False

        return True

    cases = []
    for case in all_combinations:
        if is_invalid(*case):
            cases.append(case)
    return cases


def test_fused_marlin_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    ep_size: int,
    dtype: torch.dtype,
    group_size: int,
    act_order: bool,
    quant_type: ScalarType,
    is_k_full: bool,
):
    torch.cuda.manual_seed(0)
    has_zp = quant_type in [scalar_types.uint4, scalar_types.uint8]

    print("=============  MoE Arguments =============")
    print(f"-- m                    : {m}")
    print(f"-- n (intermedia_size)  : {n}")
    print(f"-- k (hidden_size)      : {k}")
    print(f"-- e                    : {e}")
    print(f"-- topk                 : {topk}")
    print(f"-- ep_size              : {ep_size}")
    print(f"-- dtype                : {dtype}")
    print(f"-- group_size           : {group_size}")
    print(f"-- act_order            : {act_order}")
    print(f"-- quant_type           : {quant_type}")
    print(f"-- is_k_full            : {is_k_full}")
    print(f"-- has_zp               : {has_zp}")
    print("===========================================")

    # if quant_type == scalar_types.float8_e4m3fn:
    #     if group_size not in [-1, 128]:
    #         return
    #     if act_order:
    #         return

    # # Filter act_order
    # if act_order:
    #     if quant_type == scalar_types.float8_e4m3fn:
    #         return
    #     if group_size == -1:
    #         return
    #     if group_size in (k, n):
    #         return
    #     if has_zp:
    #         return
    # else:
    #     if not is_k_full:
    #         return

    # if quant_type == scalar_types.float4_e2m1f and group_size != 16:
    #     return
    # if quant_type != scalar_types.float4_e2m1f and group_size == 16:
    #     return

    a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 20
    w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 20

    dump(a, "a") 
    dump(w1, "w1")
    dump(w2, "w2")

    if ep_size > 1:
        local_e = e // ep_size
        e_ids = torch.randperm(e, device="cuda", dtype=torch.int32)[:local_e]
        e_map = torch.full((e, ), -1, device="cuda", dtype=torch.int32)
        e_map[e_ids] = torch.arange(local_e, device="cuda", dtype=torch.int32)
        w1 = w1[e_ids]
        w2 = w2[e_ids]
    else:
        e_map = None

    w_ref1_l = []
    qweight1_l = []
    scales1_l = []
    global_scale1_l = []
    zeros1_l = []
    g_idx1_l = []
    sort_indices1_l = []

    for i in range(w1.shape[0]):
        if quant_type == scalar_types.float4_e2m1f:
            w_ref1, qweight1, scales1, global_scale1 = \
                rand_marlin_weight_fp4_like(w1[i], group_size)

            w_ref1_l.append(w_ref1.T)
            qweight1_l.append(qweight1)
            scales1_l.append(scales1)
            global_scale1_l.append(global_scale1)
        elif quant_type == scalar_types.float8_e4m3fn:
            w_ref1, qweight1, scales1 = marlin_quant_fp8_torch(
                w1[i], group_size)
            w_ref1_l.append(w_ref1.T)
            qweight1_l.append(qweight1)
            scales1_l.append(scales1)
        elif has_zp:
            w_ref1, qweight1, scales1, zeros1 = awq_marlin_quantize(
                w1[i].transpose(1, 0), quant_type, group_size)
            if i == 0:
                print(f"-- [awq_marlin_quantize] w_ref1: shape={w_ref1.size()}, dtype={w_ref1.dtype}")
                print(f"-- [awq_marlin_quantize] qweight1: shape={qweight1.size()}, dtype={qweight1.dtype}")
                print(f"-- [awq_marlin_quantize] scales1: shape={scales1.size()}, dtype={scales1.dtype}")
                print(f"-- [awq_marlin_quantize] zeros1: shape={zeros1.size()}, dtype={zeros1.dtype}")

            w_ref1_l.append(w_ref1.T)
            qweight1_l.append(qweight1)
            scales1_l.append(scales1)
            zeros1_l.append(zeros1)
        else:
            test_perm = torch.randperm(k)
            w_ref1, qweight1, scales1, g_idx1, sort_indices1, _ = \
                marlin_quantize(w1[i].transpose(1, 0), quant_type,
                                group_size, act_order, test_perm)

            w_ref1_l.append(w_ref1.T)
            qweight1_l.append(qweight1)
            scales1_l.append(scales1)
            g_idx1_l.append(g_idx1)
            sort_indices1_l.append(sort_indices1)

    w_ref1 = stack_and_dev(w_ref1_l)
    qweight1 = stack_and_dev(qweight1_l).contiguous()
    scales1 = stack_and_dev(scales1_l)
    global_scale1 = stack_and_dev(global_scale1_l) if global_scale1_l else None
    g_idx1 = stack_and_dev(g_idx1_l) if g_idx1_l else None
    zeros1 = stack_and_dev(zeros1_l) if zeros1_l else None
    sort_indices1 = stack_and_dev(sort_indices1_l) if sort_indices1_l else None

    dump(w_ref1, "w_ref1")
    dump(qweight1, "qweight1")
    dump(scales1, "scales1")
    dump(global_scale1, "global_scale1")
    dump(g_idx1, "g_idx1")
    dump(zeros1, "zeros1")
    dump(sort_indices1, "sort_indices1")

    w_ref2_l = []
    qweight2_l = []
    scales2_l = []
    global_scale2_l = []
    zeros2_l = []
    g_idx2_l = []
    sort_indices2_l = []

    for i in range(w2.shape[0]):
        if quant_type == scalar_types.float4_e2m1f:
            w_ref2, qweight2, scales2, global_scale2 = \
                rand_marlin_weight_fp4_like(w2[i], group_size)

            w_ref2_l.append(w_ref2.T)
            qweight2_l.append(qweight2)
            scales2_l.append(scales2)
            global_scale2_l.append(global_scale2)
        elif quant_type == scalar_types.float8_e4m3fn:
            w_ref2, qweight2, scales2 = marlin_quant_fp8_torch(
                w2[i], group_size)
            w_ref2_l.append(w_ref2.T)
            qweight2_l.append(qweight2)
            scales2_l.append(scales2)
        elif has_zp:
            w_ref2, qweight2, scales2, zeros2 = awq_marlin_quantize(
                w2[i].transpose(1, 0), quant_type, group_size)
            if i == 0:
                print(f"-- [awq_marlin_quantize] w_ref2: shape={w_ref2.size()}, dtype={w_ref2.dtype}")
                print(f"-- [awq_marlin_quantize] qweight2: shape={qweight2.size()}, dtype={qweight2.dtype}")
                print(f"-- [awq_marlin_quantize] scales2: shape={scales2.size()}, dtype={scales2.dtype}")
                print(f"-- [awq_marlin_quantize] zeros2: shape={zeros2.size()}, dtype={zeros2.dtype}")

            w_ref2_l.append(w_ref2.T)
            qweight2_l.append(qweight2)
            scales2_l.append(scales2)
            zeros2_l.append(zeros2)
        else:
            test_perm = torch.randperm(n)
            w_ref2, qweight2, scales2, g_idx2, sort_indices2, _ = \
                marlin_quantize(w2[i].transpose(1, 0), quant_type,
                                group_size, act_order, test_perm)

            w_ref2_l.append(w_ref2.T)
            qweight2_l.append(qweight2)
            scales2_l.append(scales2)
            g_idx2_l.append(g_idx2)
            sort_indices2_l.append(sort_indices2)

    w_ref2 = stack_and_dev(w_ref2_l)
    qweight2 = stack_and_dev(qweight2_l).contiguous()
    scales2 = stack_and_dev(scales2_l)
    global_scale2 = stack_and_dev(global_scale2_l) if global_scale2_l else None
    g_idx2 = stack_and_dev(g_idx2_l) if g_idx2_l else None
    zeros2 = stack_and_dev(zeros2_l) if zeros2_l else None
    sort_indices2 = stack_and_dev(sort_indices2_l) if sort_indices2_l else None

    dump(w_ref2, "w_ref2")
    dump(qweight2, "qweight2")
    dump(scales2, "scales2")
    dump(global_scale2, "global_scale2")
    dump(g_idx2, "g_idx2")
    dump(zeros2, "zeros2")
    dump(sort_indices2, "sort_indices2")

    score = torch.randn((m, e), device="cuda", dtype=dtype)
    dump(score, "score")

    topk_weights, topk_ids, _ = fused_topk(a, score, topk, False)
    dump(topk_ids, "topk_ids")
    dump(topk_weights, "topk_weights")

    #with set_current_vllm_config(vllm_config):
    #    torch_output = torch_moe(a, w_ref1, w_ref2, score, topk, e_map)

    marlin_output = torch.ops.vllm.fused_marlin_moe(
        a,
        qweight1,
        qweight2,
        scales1,
        scales2,
        score,
        topk_weights,
        topk_ids,
        global_num_experts=e,
        expert_map=e_map,
        global_scale1=global_scale1,
        global_scale2=global_scale2,
        g_idx1=g_idx1,
        g_idx2=g_idx2,
        sort_indices1=sort_indices1,
        sort_indices2=sort_indices2,
        w1_zeros=zeros1,
        w2_zeros=zeros2,
        quant_type_id=quant_type.id,
        is_k_full=is_k_full)

    dump(marlin_output, "moe_out")

    #torch.testing.assert_close(marlin_output, torch_output, atol=5e-2, rtol=0)


if __name__ == "__main__":
    all_test_cases = marlin_moe_generate_valid_test_cases()
    for case in all_test_cases:
        print(case)
        test_fused_marlin_moe(
            m=case[0],
            n=case[1],
            k=case[2],
            e=case[3],
            topk=case[4],
            ep_size=case[5],
            dtype=case[6],
            group_size=case[7],
            act_order=case[8],
            quant_type=case[9],
            is_k_full=case[10],
        )
