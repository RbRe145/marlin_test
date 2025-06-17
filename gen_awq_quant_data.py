import os
import numpy as np
import paddle
from utils.marlin_utils_test import awq_marlin_quantize
from utils.scalar_type import scalar_types
from paddlenlp_ops import moe_expert_dispatch

# —— 配置 —— #
DUMP_DIR   = "dump"         # 原始 .npy 文件所在目录
OUT_DIR    = "dump_paddle_q"# 量化后结果存放目录
os.makedirs(OUT_DIR, exist_ok=True)

# 设置 Paddle 使用 GPU
paddle.set_device("gpu")

# 量化参数
# 请根据你自己的 ScalarType 和 group_size 填写
quant_type = scalar_types.uint4  # e.g. scalar_types.uint4
group_size = -1  # e.g. 64

# —— 1. 从磁盘加载原始张量 —— #
# NumPy 加载
w1_np = np.load(f"{DUMP_DIR}/w1.npy")   # shape = [e, 2*n, k]
w2_np = np.load(f"{DUMP_DIR}/w2.npy")   # shape = [e, k, n]

# 转 Paddle Tensor（自动放到 GPU）
w1 = paddle.to_tensor(w1_np, dtype="float16")
w2 = paddle.to_tensor(w2_np, dtype="float16")

# 专家数
e = w1.shape[0]

# —— 2. 对 w1 的所有专家循环量化 —— #
w1_refs, qweights1, scales1, zeros1 = [], [], [], []
for i in range(e):
    # 转置 (2*n, k) -> (k, 2*n)
    w1_i_T = w1[i].transpose([1, 0])
    w_ref, qweight, scales, zeros = awq_marlin_quantize(
        w1_i_T, quant_type=quant_type, group_size=group_size
    )
    w1_refs.append(w_ref)
    qweights1.append(qweight)
    scales1.append(scales)
    zeros1.append(zeros)

# 堆叠成形状 [e, ...]
w1_ref_all   = paddle.stack(w1_refs)
qweight1_all = paddle.stack(qweights1)
scales1_all  = paddle.stack(scales1)
zeros1_all   = paddle.stack(zeros1)

# 保存到磁盘（先转回 NumPy）
np.save(f"{OUT_DIR}/w_ref1.npy",   w1_ref_all.cpu().numpy())
np.save(f"{OUT_DIR}/qweight1.npy", qweight1_all.cpu().numpy())
np.save(f"{OUT_DIR}/scales1.npy",  scales1_all.cpu().numpy())
np.save(f"{OUT_DIR}/zeros1.npy",   zeros1_all.cpu().numpy())

# —— 3. 对 w2 的所有专家循环量化 —— #
w2_refs, qweights2, scales2, zeros2 = [], [], [], []
for i in range(e):
    # 转置 (k, n) -> (n, k)
    w2_i_T = w2[i].transpose([1, 0])
    w_ref, qweight, scales, zeros = awq_marlin_quantize(
        w2_i_T, quant_type=quant_type, group_size=group_size
    )
    w2_refs.append(w_ref)
    qweights2.append(qweight)
    scales2.append(scales)
    zeros2.append(zeros)

w2_ref_all   = paddle.stack(w2_refs)
qweight2_all = paddle.stack(qweights2)
scales2_all  = paddle.stack(scales2)
zeros2_all   = paddle.stack(zeros2)

np.save(f"{OUT_DIR}/w_ref2.npy",   w2_ref_all.cpu().numpy())
np.save(f"{OUT_DIR}/qweight2.npy", qweight2_all.cpu().numpy())
np.save(f"{OUT_DIR}/scales2.npy",  scales2_all.cpu().numpy())
np.save(f"{OUT_DIR}/zeros2.npy",   zeros2_all.cpu().numpy())

print(f"All {e} experts quantized. Results saved under `{OUT_DIR}/`.")
