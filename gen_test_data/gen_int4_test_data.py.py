import os
import numpy as np
import torch
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import awq_marlin_quantize
from vllm.scalar_type import scalar_types
from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk

def generate_awq_int4_data_gpu(
    m=95,           # token 数（size_m）
    n=256,          # 每个专家输出列数（size_n before packing）
    k=7168,         # 隐层维度（size_k）
    e=8,            # 专家总数
    group_size=64,  # 分组量化大小
    topk=8,
    save_dir="moe_int4_awq_data_gpu"
):
    device = torch.device("cuda")
    os.makedirs(save_dir, exist_ok=True)

    # 1) 在 GPU 上生成激活 A:[m, k]
    A = torch.randn((m, k), dtype=torch.half, device=device) / 10
    np.save(os.path.join(save_dir, "A.npy"), A.cpu().numpy())

    # 2) 在 GPU 上生成专家权重 W:[e, k, n]
    W = torch.randn((e, k, n), dtype=torch.half, device=device) / 20
    np.save(os.path.join(save_dir, "W.npy"), W.cpu().numpy())

    w_ref_list   = []
    qw_list      = []
    scales_list  = []
    zeros_list   = []

    # 3) 在 GPU 上调用 awq_marlin_quantize
    for i in range(e):
        w_i = W[i].transpose(1, 0)   # [n, k] on CUDA
        w_ref, qweight, scales, zeros = awq_marlin_quantize(
            w_i,
            scalar_types.uint4,
            group_size
        )
        # 保存前搬到 CPU
        w_ref_list  .append(w_ref.transpose(1, 0).cpu().numpy())
        qw_list     .append(qweight.cpu().numpy())
        scales_list .append(scales.cpu().numpy())
        zeros_list  .append(zeros.cpu().numpy())
        
    score = torch.randn((m, e), device="cuda", dtype=torch.half)
    topk_weights, topk_ids, _ = fused_topk(A, score, topk, False)
    # 4) 堆叠并保存
    np.save(os.path.join(save_dir, "w_ref.npy"),   np.stack(w_ref_list,   axis=0))
    np.save(os.path.join(save_dir, "qweight.npy"), np.stack(qw_list,      axis=0))
    np.save(os.path.join(save_dir, "scales.npy"),  np.stack(scales_list,  axis=0))
    np.save(os.path.join(save_dir, "zeros.npy"),   np.stack(zeros_list,   axis=0))

    # 5) 其余占位
    none_arr = np.array(None, dtype=object)
    for name in ("global_scale", "g_idx", "sort_idx"):
        np.save(os.path.join(save_dir, f"{name}.npy"), none_arr)

    print(f"✅ GPU 量化测试数据已保存到目录：{save_dir}/")

if __name__ == "__main__":
    generate_awq_int4_data_gpu()
