# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility functions used for tests and benchmarks"""

from typing import Optional

import numpy as np
import paddle

from .scalar_type import ScalarType

from .marlin_utils import (GPTQ_MARLIN_TILE, marlin_permute_scales,
                           marlin_zero_points)
from .quant_utils import (get_pack_factor,
                          quantize_weights, sort_weights)


class MarlinWorkspace:

    def __init__(self, out_features, min_thread_n, max_parallel):
        assert (out_features % min_thread_n == 0), (
            "out_features = {} is undivisible by min_thread_n = {}".format(
                out_features, min_thread_n))

        max_workspace_size = ((out_features // min_thread_n) * max_parallel)

        self.scratch = paddle.zeros(max_workspace_size,
                                   dtype="int64",
                                   device="cuda")


def marlin_permute_weights(q_w, size_k, size_n, perm, tile=GPTQ_MARLIN_TILE):
    assert q_w.shape == [size_k, size_n], f"{q_w.shape}   {size_k}   {size_n}"
    assert size_k % tile == 0, f"size_k = {size_k}, tile = {tile}"
    assert size_n % tile == 0, f"size_k = {size_n}, tile = {tile}"

    # Permute weights to 16x64 marlin tiles
    q_w = q_w.reshape((size_k // tile, tile, size_n // tile, tile))
    q_w = paddle.transpose(q_w, perm=[0, 2, 1, 3])
    q_w = q_w.reshape((size_k // tile, size_n * tile))

    q_w = q_w.reshape((-1, perm.numel()))[:, perm].reshape(q_w.shape)

    return q_w


def marlin_weights(q_w, size_k, size_n, num_bits, perm):
    # Permute
    q_w = marlin_permute_weights(q_w, size_k, size_n, perm)

    # Pack
    pack_factor = get_pack_factor(num_bits)
    orig_device = q_w.place

    q_w = q_w.cpu().numpy().astype(np.uint32)

    q_packed = np.zeros((q_w.shape[0], q_w.shape[1] // pack_factor),
                        dtype=np.uint32)
    for i in range(pack_factor):
        q_packed |= q_w[:, i::pack_factor] << num_bits * i

    q_packed = paddle.to_tensor(q_packed.astype(np.int32), place=orig_device)

    return q_packed


def get_weight_perm(num_bits: int):
    perm_list: list[int] = []
    for i in range(32):
        perm1: list[int] = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                    2 * (i % 4),
                    2 * (i % 4) + 1,
                    2 * (i % 4 + 4),
                    2 * (i % 4 + 4) + 1,
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm_list.extend([p + 256 * j for p in perm1])

    perm = np.array(perm_list)

    if num_bits == 4:
        interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    elif num_bits == 8:
        interleave = np.array([0, 2, 1, 3])
    else:
        raise Exception("num_bits must be 4 or 8, got {}".format(num_bits))

    perm = perm.reshape((-1, len(interleave)))[:, interleave].ravel()
    perm = paddle.to_tensor(perm)
    return perm


def awq_marlin_quantize(w: paddle.Tensor, quant_type: ScalarType,
                        group_size: int):
    size_k, size_n = w.shape

    # Normalize group_size
    if group_size == -1:
        group_size = size_k
    assert group_size <= size_k

    # Detect num groups
    assert size_k % group_size == 0
    num_groups = size_k // group_size

    # Quantize with zp
    w_ref, q_w, s, zp = quantize_weights(w,
                                         quant_type,
                                         group_size,
                                         zero_points=True)

    # Reformat to marlin
    weight_perm = get_weight_perm(quant_type.size_bits)
    marlin_q_w = marlin_weights(q_w, size_k, size_n, quant_type.size_bits,
                                weight_perm)
    marlin_s = marlin_permute_scales(s, size_k, size_n, group_size)
    marlin_zp = marlin_zero_points(zp, num_groups, size_n,
                                   quant_type.size_bits)

    # Create result
    res_list = [w_ref, marlin_q_w, marlin_s, marlin_zp]
    for i in range(len(res_list)):
        res_list[i] = res_list[i].to(w.place)

    return res_list
