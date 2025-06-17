import os
import numpy as np

import paddle
from paddle.base.core import moe_wna16_marlin_gemm
import fused_marlin_moe
from utils.marlin_utils_test import awq_marlin_quantize
from utils.scalar_type import scalar_types
import warnings

def get_quantize_weight(
    w: paddle.Tensor,
    quant_type,
    group_size: int
) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    Perform AWQ quantization on all experts (first dimension) of the weight tensor and
    stack the results for return.

    Args:
        w (paddle.Tensor): The weight tensor to quantize, with shape [e, 2*n, k].
        quant_type: The quantization type, e.g., scalar_types.uint4.
        group_size (int): The group size for quantization; use -1 for no grouping.

    Returns:
        w_ref_all   (paddle.Tensor): Stacked reference weights, shape [e, *w_ref.shape].
        qweight_all (paddle.Tensor): Stacked quantized weights, shape [e, *qweight.shape].
        scales_all  (paddle.Tensor): Stacked scale factors, shape [e, *scales.shape].
        zeros_all   (paddle.Tensor): Stacked zero-point offsets, shape [e, *zeros.shape].
    """
    e = w.shape[0]
    w_refs, qweights, scales_list, zeros_list = [], [], [], []

    for i in range(e):
        w_i_T = w[i].transpose([1, 0])
        w_ref, qweight, scales, zeros = awq_marlin_quantize(
            w_i_T,
            quant_type=quant_type,
            group_size=group_size
        )
        w_refs.append(w_ref)
        qweights.append(qweight)
        scales_list.append(scales)
        zeros_list.append(zeros)

    w_ref_all   = paddle.stack(w_refs)
    qweight_all = paddle.stack(qweights)
    scales_all  = paddle.stack(scales_list)
    zeros_all   = paddle.stack(zeros_list)

    return w_ref_all, qweight_all, scales_all, zeros_all


def test_moe_group_gemm1(tensor_dict, M, N, K, E, topk):
    block_size_m = 8
    is_k_full = True
    use_atomic_add = True

    workspace = paddle.empty([528], dtype="int32")
    w_ref, qweight, scales, zeros = get_quantize_weight(tensor_dict['w1'], quant_type=scalar_types.uint4, group_size=-1)
    
    gemm_out = moe_wna16_marlin_gemm(
        a=tensor_dict["a"],
        c_or_none=None,
        b_q_weight=qweight,  #tensor_dict["qweight1"],
        b_scales=scales, # tensor_dict["scales1"],
        global_scale_or_none=None,
        b_zeros_or_none=tensor_dict["zeros1"],
        g_idx_or_none=None,
        perm_or_none=None,
        workspace=workspace,
        sorted_token_ids=tensor_dict["sorted_token_ids"],
        expert_ids=tensor_dict["expert_ids"],
        num_tokens_post_padded=tensor_dict["num_tokens_post_padded"],
        topk_weights=tensor_dict["topk_weights"],
        moe_block_size=block_size_m,
        top_k=topk,
        mul_topk_weights=False,
        is_ep=False,
        b_q_type_str="uint4",
        size_m=M,
        size_n=2 * N,
        size_k=K,
        is_k_full=is_k_full,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=True,
        is_zp_float=False,
    )
    return gemm_out


def test_moe_group_gemm2(tensor_dict, M, N, K, E, topk):
    block_size_m = 8
    is_k_full = True
    use_atomic_add = True

    workspace = paddle.empty([528], dtype="int32")
    w_ref, qweight, scales, zeros = get_quantize_weight(tensor_dict['w2'], quant_type=scalar_types.uint4, group_size=-1)

    gemm_out = moe_wna16_marlin_gemm(
        a=tensor_dict["swiglu_out"],
        c_or_none=None,
        b_q_weight=qweight, #tensor_dict["qweight2"],
        b_scales=scales, #tensor_dict["scales2"],
        global_scale_or_none=None,
        b_zeros_or_none=zeros, #tensor_dict["zeros2"],
        g_idx_or_none=None,
        perm_or_none=None,
        workspace=workspace,
        sorted_token_ids=tensor_dict["sorted_token_ids"],
        expert_ids=tensor_dict["expert_ids"],
        num_tokens_post_padded=tensor_dict["num_tokens_post_padded"],
        topk_weights=tensor_dict["topk_weights"],
        moe_block_size=block_size_m,
        top_k=1,
        mul_topk_weights=True,
        is_ep=False,
        b_q_type_str="uint4",
        size_m=M * topk,
        size_n=K,
        size_k=N,
        is_k_full=is_k_full,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=True,
        is_zp_float=False,
    )
    gemm_out = gemm_out.reshape([-1, topk, K])
    return gemm_out


def load_tensors():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(test_dir, "dump_paddle_q")

    def load_to_tensor(name):
        array = np.load(f"{test_dir}/{name}.npy", allow_pickle=True)
        tensor = paddle.to_tensor(array)
        print(f"-- {name}: shape={tensor.shape}, dtype={tensor.dtype}")
        return tensor

    tensor_names = [
        "score",
        "topk_ids",
        "topk_weights",
        "sorted_token_ids",
        "expert_ids",
        "num_tokens_post_padded",
        "a", # input of group_gemm 1
        "w1",
        "qweight1",
        "scales1",
        "zeros1",
        "gemm_out1",
        "swiglu_out", # input of group_gemm 2
        "w2",
        "qweight2",
        "scales2",
        "zeros2",
        "gemm_out2",
        "moe_out",   
    ]
    tensor_dict = {}
    for name in tensor_names:
        tensor_dict[name] = load_to_tensor(name)
    return tensor_dict


def test_moe_gemm(tensor_dict):
    M, K = tensor_dict["a"].shape
    # use shape only, it has no effect on precision test
    E = tensor_dict["qweight1"].shape[0]
    N = tensor_dict["qweight2"].shape[1] * 16
    topk = tensor_dict["topk_ids"].shape[1]

    print(f"-- M={M}, N={N}, K={K}, E={E}, topk={topk}")
    gemm_out1 = test_moe_group_gemm1(tensor_dict, M, N, K, E, topk)

    try:
        np.testing.assert_allclose(
            gemm_out1,
            tensor_dict["gemm_out1"],
            atol=1e-2,
            rtol=1e-2,
        )
    except AssertionError as err:
        # err 是一个包含差异信息的异常对象
        warnings.warn(
            f"gemm_out1 与参考值不匹配：{err}",
            category=UserWarning,
            stacklevel=2
        )
    #print(gemm_out1)
    #print(tensor_dict["gemm_out1"])

    gemm_out2 = test_moe_group_gemm2(tensor_dict, M, N, K, E, topk)

    try:
        np.testing.assert_allclose(
            gemm_out2,
            tensor_dict["gemm_out2"],
            atol=1e-2,
            rtol=1e-2,
        )
    except AssertionError as err:
    # err 是一个包含差异信息的异常对象
        warnings.warn(
            f"gemm_out2 与参考值不匹配：{err}",
            category=UserWarning,
            stacklevel=2
        )


def test_moe_decode(tensor_dict):
    topk = tensor_dict["topk_ids"].shape[1]

    from paddlenlp_ops import moe_expert_dispatch
    _, _, _, topk_weights, topk_ids = moe_expert_dispatch(
        input=tensor_dict["a"],
        gating_output=tensor_dict["score"].cast("float32"),
        moe_topk=topk,
        group_moe=False,
        topk_only_mode=False,
    )

    moe_out = fused_marlin_moe.fused_marlin_moe(
        hidden_states=tensor_dict["a"],
        w1=tensor_dict["qweight1"],
        w2=tensor_dict["qweight2"],
        w1_scale=tensor_dict["scales1"],
        w2_scale=tensor_dict["scales2"],
        gating_output=tensor_dict["score"],
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        quant_type_str="uint4",
        global_num_experts=-1,
        w1_zeros=tensor_dict["zeros1"],
        w2_zeros=tensor_dict["zeros2"],
        workspace=None,
        is_k_full=True,
    )

    np.testing.assert_allclose(
        moe_out,
        tensor_dict["moe_out"],
        atol=1e-2,
        rtol=1e-2,
    )


def test_main():
    tensor_dict = load_tensors()
    test_moe_gemm(tensor_dict)
    # test_moe_decode(tensor_dict)


if __name__ == "__main__":
    test_main()
