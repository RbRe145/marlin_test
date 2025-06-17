# marlin\_test Paddle 移植（wint4）

## 进度

1. **AWQ wint4 量化移植**：基于 vLLM 的 Python 实现完成；量化精度与 vLLM 保持一致。
2. **MOE\_WNA16 Marlin GEMM 移植**：集成并验证，通过与 vLLM 中的 `fused_marlin_moe` 对比，精度一致。


## 使用须知

vLLM 环境与 PaddleNLP 环境存在兼容性差异，实验流程如下：

1. 在 **vLLM 环境** 下运行，生成并导出测试数据（`.npy` 文件）。
2. 切换到 **Paddle 环境**，加载测试数据进行精度对比。

## 在 PaddleNLP 中替换 fused\_topk

原始 Marlin MOE GEMM 使用 `fused_topk`，在 PaddleNLP 中可使用 `moe_expert_dispatch` 实现同等功能。请在代码中插入以下片段：

```python
from paddlenlp_ops import moe_expert_dispatch

# Top-K 门控分发
_, _, _, topk_weights, topk_ids = moe_expert_dispatch(
    input=A,
    gating_output=score.cast("float32"),
    moe_topk=topk,
    group_moe=False,
    topk_only_mode=False,
)
```
说明：

* `input`：待分发的激活张量 `A`。
* `gating_output`：门控概率 `score`，需 cast 为 `float32`。
* `moe_topk`：Top-K 值。
* `group_moe`：是否使用分组 MOE，设为 `False`。
* `topk_only_mode`：只输出 Top-K 权重和索引，设为 `False`。

## 发布包

* **Paddle 安装包**：

  ```
  /root/paddlejob/workspace/env_run/output/zhenghuaijin/marlin_test/paddlepaddle_gpu-0.0.0-cp310-cp310-linux_x86_64.whl
  ```

