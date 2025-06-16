# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fused MoE utilities for GPTQ."""
import functools
import numpy as np
from typing import Optional

import paddle
from paddle.base.core import moe_wna16_marlin_gemm

#from vllm.model_executor.layers.fused_moe.fused_moe import (
#    moe_align_block_size, try_get_optimal_moe_config)
#from vllm.model_executor.layers.quantization.utils.marlin_utils import (
#    marlin_make_workspace_new, maybe_warn_marlin_atomic_add)


def print_tensor_info(t, name, func_name):
    if t is None:
        print(f"-- [{func_name}] {name}: None")
    elif isinstance(t, paddle.Tensor):
        print(f"-- [{func_name}] {name}: shape={t.shape}, dtype={t.dtype}")
    else:
        print(f"-- [{func_name}] {name}: {t}")


def fused_marlin_moe(hidden_states: paddle.Tensor,
                     w1: paddle.Tensor,
                     w2: paddle.Tensor,
                     w1_scale: paddle.Tensor,
                     w2_scale: paddle.Tensor,
                     gating_output: paddle.Tensor,
                     topk_weights: paddle.Tensor,
                     topk_ids: paddle.Tensor,
                     quant_type_str: str,
                     global_num_experts: int = -1,
                     w1_zeros: Optional[paddle.Tensor] = None,
                     w2_zeros: Optional[paddle.Tensor] = None,
                     workspace: Optional[paddle.Tensor] = None,
                     is_k_full: bool = True) -> paddle.Tensor:
    """
    This function computes a Mixture of Experts (MoE) layer using two sets of
    weights, w1 and w2, and top-k gating mechanism.

    Parameters:
    - hidden_states (paddle.Tensor): The input tensor to the MoE layer.
    - w1 (paddle.Tensor): The first set of expert weights.
    - w2 (paddle.Tensor): The second set of expert weights.
    - w1_scale (paddle.Tensor): Scale to be used for w1.
    - w2_scale (paddle.Tensor): Scale to be used for w2.
    - gating_output (paddle.Tensor): The output of the gating operation
        (before softmax).
    - topk_weights (paddle.Tensor): Top-k weights.
    - topk_ids (paddle.Tensor): Indices of topk-k elements.
    - quant_type_str (str): the weights quantization used, the optional value is "uint4".
    - w1_zeros (Optional[paddle.Tensor]): Optional zero points to be used for w1.
    - w2_zeros (Optional[paddle.Tensor]): Optional zero points to be used for w2.

    Returns:
    - paddle.Tensor: The output tensor after applying the MoE layer.
    """
    assert quant_type_str in ["uint4"]

    num_bits = 4

    # Check constraints.
    assert hidden_states.shape[0] == gating_output.shape[
        0], "Number of tokens mismatch"
    assert hidden_states.shape[
        1] == w1.shape[1] * 16, "Hidden size mismatch w1"
    assert hidden_states.shape[1] == w2.shape[2] // (
        num_bits // 2), "Hidden size mismatch w2"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [paddle.float16, paddle.bfloat16]
    assert num_bits in [4]

    M, K = hidden_states.shape
    E = w1.shape[0]
    N = w2.shape[1] * 16
    topk = topk_ids.shape[1]

    # get_config_func = functools.partial(
    #     try_get_optimal_moe_config,
    #     w1.shape,
    #     w2.shape,
    #     topk_ids.shape[1],
    #     None,
    #     is_marlin=True,
    # )
    # config = get_config_func(M)

    block_size_m = 8 # config["BLOCK_SIZE_M"]

    if global_num_experts == -1:
        global_num_experts = E

    from paddlenlp_ops import preprocess_for_moe
    if topk_ids.dtype != paddle.int64:
        topk_ids = topk_ids.cast("int64")
    sorted_token_ids, expert_ids, num_tokens_post_padded = preprocess_for_moe(topk_ids, global_num_experts, block_size_m)

    if workspace is None:
        #workspace = marlin_make_workspace_new(hidden_states.device, 4)
        workspace = paddle.empty([1024 * 1024], dtype="int32")

    intermediate_cache1 = paddle.zeros(
        [M * topk, 2 * N],
        dtype=hidden_states.dtype,
    )
    intermediate_cache2 = paddle.zeros(
        (M * topk, N),
        dtype=hidden_states.dtype,
    )
    intermediate_cache3 = paddle.zeros(
        (M * topk, K),
        dtype=hidden_states.dtype,
    )

    # maybe_warn_marlin_atomic_add(hidden_states.device, hidden_states.dtype)
    use_atomic_add = hidden_states.dtype == paddle.float16 or \
        paddle.device.cuda.get_device_capability()[0] >= 9
    print(f"-- [fused_marlin_moe] use_atomic_add={use_atomic_add}")

    print_tensor_info(t=hidden_states, name="a", func_name="moe_wna16_marlin_gemm-1")
    print_tensor_info(t=intermediate_cache1, name="c_or_none", func_name="moe_wna16_marlin_gemm-1")
    print_tensor_info(t=w1, name="b_q_weight", func_name="moe_wna16_marlin_gemm-1")
    print_tensor_info(t=w1_scale, name="b_scales", func_name="moe_wna16_marlin_gemm-1")
    print_tensor_info(t=w1_zeros, name="b_zeros_or_none", func_name="moe_wna16_marlin_gemm-1")
    print_tensor_info(t=workspace, name="workspace", func_name="moe_wna16_marlin_gemm-1")
    print_tensor_info(t=sorted_token_ids, name="sorted_token_ids", func_name="moe_wna16_marlin_gemm-1")
    print_tensor_info(t=expert_ids, name="expert_ids", func_name="moe_wna16_marlin_gemm-1")
    print_tensor_info(t=num_tokens_post_padded, name="num_tokens_post_padded", func_name="moe_wna16_marlin_gemm-1")
    print_tensor_info(t=topk_weights, name="topk_weights", func_name="moe_wna16_marlin_gemm-1")
    print_tensor_info(t=block_size_m, name="moe_block_size", func_name="moe_wna16_marlin_gemm-1")
    print_tensor_info(t=topk, name="top_k", func_name="moe_wna16_marlin_gemm-1")
    print_tensor_info(t=False, name="mul_topk_weights", func_name="moe_wna16_marlin_gemm-1")
    print_tensor_info(t=False, name="is_ep", func_name="moe_wna16_marlin_gemm-1")
    print_tensor_info(t=quant_type_str, name="b_q_type_id", func_name="moe_wna16_marlin_gemm-1")
    print_tensor_info(t=M, name="size_m", func_name="moe_wna16_marlin_gemm-1")
    print_tensor_info(t=2 * N, name="size_n", func_name="moe_wna16_marlin_gemm-1")
    print_tensor_info(t=K, name="size_k", func_name="moe_wna16_marlin_gemm-1")
    print_tensor_info(t=is_k_full, name="is_k_full", func_name="moe_wna16_marlin_gemm-1")
    print_tensor_info(t=use_atomic_add, name="use_atomic_add", func_name="moe_wna16_marlin_gemm-1")
    print_tensor_info(t=True, name="use_fp32_reduce", func_name="moe_wna16_marlin_gemm-1")
    print_tensor_info(t=False, name="is_zp_float", func_name="moe_wna16_marlin_gemm-1")
    intermediate_cache1 = moe_wna16_marlin_gemm(
        a=hidden_states,
        c_or_none=intermediate_cache1,
        b_q_weight=w1,
        b_scales=w1_scale,
        global_scale_or_none=None,
        b_zeros_or_none=w1_zeros,
        g_idx_or_none=None,
        perm_or_none=None,
        workspace=workspace,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        topk_weights=topk_weights,
        moe_block_size=block_size_m,
        top_k=topk,
        mul_topk_weights=False,
        is_ep=False,
        b_q_type_str=quant_type_str,
        size_m=M,
        size_n=2 * N,
        size_k=K,
        is_k_full=is_k_full,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=True,
        is_zp_float=False)

    intermediate_cache2 = paddle.incubate.nn.functional.swiglu(intermediate_cache1.reshape([-1, 2 * N]))

    print_tensor_info(t=intermediate_cache2, name="a", func_name="moe_wna16_marlin_gemm-2")
    print_tensor_info(t=intermediate_cache3, name="c_or_none", func_name="moe_wna16_marlin_gemm-2")
    print_tensor_info(t=w2, name="b_q_weight", func_name="moe_wna16_marlin_gemm-2")
    print_tensor_info(t=w2_scale, name="b_scales", func_name="moe_wna16_marlin_gemm-2")
    print_tensor_info(t=w2_zeros, name="b_zeros_or_none", func_name="moe_wna16_marlin_gemm-2")
    print_tensor_info(t=workspace, name="workspace", func_name="moe_wna16_marlin_gemm-2")
    print_tensor_info(t=sorted_token_ids, name="sorted_token_ids", func_name="moe_wna16_marlin_gemm-2")
    print_tensor_info(t=expert_ids, name="expert_ids", func_name="moe_wna16_marlin_gemm-2")
    print_tensor_info(t=num_tokens_post_padded, name="num_tokens_post_padded", func_name="moe_wna16_marlin_gemm-2")
    print_tensor_info(t=topk_weights, name="topk_weights", func_name="moe_wna16_marlin_gemm-2")
    print_tensor_info(t=block_size_m, name="moe_block_size", func_name="moe_wna16_marlin_gemm-2")
    print_tensor_info(t=1, name="top_k", func_name="moe_wna16_marlin_gemm-2")
    print_tensor_info(t=True, name="mul_topk_weights", func_name="moe_wna16_marlin_gemm-2")
    print_tensor_info(t=False, name="is_ep", func_name="moe_wna16_marlin_gemm-2")
    print_tensor_info(t=quant_type_str, name="b_q_type_id", func_name="moe_wna16_marlin_gemm-2")
    print_tensor_info(t=M * topk, name="size_m", func_name="moe_wna16_marlin_gemm-2")
    print_tensor_info(t=K, name="size_n", func_name="moe_wna16_marlin_gemm-2")
    print_tensor_info(t=N, name="size_k", func_name="moe_wna16_marlin_gemm-2")
    print_tensor_info(t=is_k_full, name="is_k_full", func_name="moe_wna16_marlin_gemm-2")
    print_tensor_info(t=use_atomic_add, name="use_atomic_add", func_name="moe_wna16_marlin_gemm-2")
    print_tensor_info(t=True, name="use_fp32_reduce", func_name="moe_wna16_marlin_gemm-2")
    print_tensor_info(t=False, name="is_zp_float", func_name="moe_wna16_marlin_gemm-2")
    intermediate_cache3 = moe_wna16_marlin_gemm(
        a=intermediate_cache2,
        c_or_none=intermediate_cache3,
        b_q_weight=w2,
        b_scales=w2_scale,
        global_scale_or_none=None,
        b_zeros_or_none=w2_zeros,
        g_idx_or_none=None,
        perm_or_none=None,
        workspace=workspace,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        topk_weights=topk_weights,
        moe_block_size=block_size_m,
        top_k=1,
        mul_topk_weights=True,
        is_ep=False,
        b_q_type_str=quant_type_str,
        size_m=M * topk,
        size_n=K,
        size_k=N,
        is_k_full=is_k_full,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=True,
        is_zp_float=False).reshape([-1, topk, K])

    out_hidden_states = paddle.sum(intermediate_cache3, axis=1)
    del intermediate_cache1, intermediate_cache2, intermediate_cache3
    del sorted_token_ids, expert_ids, num_tokens_post_padded
    return out_hidden_states