# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""This file is used for /tests and /benchmarks"""
from collections.abc import Mapping
from types import MappingProxyType
from typing import Optional

import numpy
import paddle

from .scalar_type import ScalarType, scalar_types

def get_pack_factor(num_bits):
    assert 32 % num_bits == 0, f"Unsupported num_bits = {num_bits}"
    return 32 // num_bits


def permute_rows(q_w: paddle.Tensor,
                 w_ref: paddle.Tensor,
                 group_size: int,
                 test_perm: Optional[paddle.Tensor] = None):
    assert q_w.shape == w_ref.shape

    orig_device = q_w.device
    k_size, _ = q_w.shape

    g_idx = paddle.zeros((k_size, ), dtype=paddle.int32)
    for i in range(k_size):
        g_idx[i] = i // group_size

    # Simulate act_order by doing a random permutation on K
    rand_perm = test_perm if test_perm is not None else paddle.randperm(k_size)

    g_idx = g_idx[rand_perm].contiguous()
    q_w = q_w[rand_perm, :].contiguous()
    w_ref = w_ref[rand_perm, :].contiguous()

    return (
        w_ref.to(device=orig_device),
        q_w.to(device=orig_device),
        g_idx.to(device=orig_device),
        rand_perm.to(device=orig_device),
    )


def quantize_weights(w: paddle.Tensor,
                     quant_type: ScalarType,
                     group_size: Optional[int],
                     zero_points: bool = False,
                     ref_zero_points_after_scales: bool = False):
    assert quant_type.is_integer(), \
        "Floating point quantization may work but has not been tested"
    assert not zero_points or group_size is not None, \
        "to have group zero points, group_size must be provided "\
        "(-1 group_size is channelwise)"

    orig_device = w.place
    orig_type = w.dtype
    size_k, size_n = w.shape

    assert w.is_floating_point(), "w must be float"

    if group_size == -1:
        group_size = size_k

    # Reshape to [groupsize, -1]
    if group_size is not None and group_size < size_k:
        w = w.reshape((-1, group_size, size_n))
        w = paddle.transpose(w, perm=[1, 0, 2])
        w = w.reshape((group_size, -1))

    # Compute scale for each group
    max_val = paddle.max(w, 0, keepdim=True)
    min_val = paddle.min(w, 0, keepdim=True)

    max_q_val = quant_type.max()
    min_q_val = quant_type.min()

    w_s = paddle.to_tensor([1.0], dtype=w.dtype, place=w.place)
    maybe_w_zp = None
    if group_size is not None:
        if zero_points:
            assert not quant_type.is_signed() and quant_type.max() > 0
            delta = max_val - min_val
            delta_clamped = paddle.clip(delta, min=1e-5) 
            w_s = delta_clamped / quant_type.max()
            
            delta2    = paddle.abs(min_val / w_s)
            rounded  = paddle.round(delta2)
            clamped  = paddle.clip(rounded, min=min_q_val, max=max_q_val)
            maybe_w_zp = clamped.astype('int32')
        else:
            # If the bias is such that there are no possible negative/positive
            #  values, set the max value to inf to avoid divide by 0
            w_s = paddle.max(
                abs(max_val / (max_q_val if max_q_val != 0 else paddle.inf)),
                abs(min_val / (min_q_val if min_q_val != 0 else paddle.inf)))

    # Quantize
    w_q = paddle.round(w / w_s).astype('int32') + (maybe_w_zp if zero_points else 0)
    w_q = w_q.clip(min=min_q_val, max=max_q_val)

    # Compute ref (dequantized)
    # For some kernels (namely Machete) the zero-points are applied after the
    # scales are applied, for this case computing the reference in similar way
    # allows us to use tighter error tolerances in our unit tests.
    if ref_zero_points_after_scales and maybe_w_zp is not None:
        w_ref = w_q.to(orig_type) * w_s - maybe_w_zp.to(orig_type) * w_s
    else:
        w_ref = (w_q - (maybe_w_zp if zero_points else 0)).to(orig_type) * w_s

    if quant_type.has_bias():
        w_q += quant_type.bias

    # Restore original shapes
    if group_size is not None and group_size < size_k:

        def reshape_w(w):
            w = w.reshape((group_size, -1, size_n))
            w = paddle.transpose(w, perm=[1, 0, 2])
            w = w.reshape((size_k, size_n)).contiguous()
            return w

        w_q = reshape_w(w_q)
        w_ref = reshape_w(w_ref)
        w_s = w_s.reshape((-1, size_n)).contiguous()

    if maybe_w_zp is not None:
        maybe_w_zp = maybe_w_zp.reshape((-1, size_n)).contiguous()
        maybe_w_zp = maybe_w_zp.to(device=orig_device)

    return (
        w_ref.to(device=orig_device),
        w_q.to(device=orig_device),
        w_s if group_size is not None else None,
        maybe_w_zp,
    )


def sort_weights(q_w: paddle.Tensor, g_idx: paddle.Tensor):
    orig_device = q_w.device

    sort_indices = paddle.argsort(g_idx).to(
        dtype=paddle.int32)  # Sort based on g_idx

    g_idx = g_idx[sort_indices].contiguous()
    q_w = q_w[sort_indices, :].contiguous()

    return (
        q_w.to(device=orig_device),
        g_idx.to(device=orig_device),
        sort_indices.to(device=orig_device),
    )


def pack_rows(
    q_w: paddle.Tensor,
    num_bits: int,
    size_k: int,
    size_n: int,
):
    assert q_w.shape == (size_k, size_n)

    pack_factor = get_pack_factor(num_bits)
    assert size_k % pack_factor == 0

    orig_device = q_w.device

    q_w = q_w.cpu().numpy().astype(numpy.uint32)

    q_res = numpy.zeros((size_k // pack_factor, size_n), dtype=numpy.uint32)

    for i in range(pack_factor):
        q_res |= q_w[i::pack_factor, :] << num_bits * i

    q_res = paddle.from_numpy(q_res.astype(numpy.int32)).to(orig_device)
    return q_res


def pack_cols(
    q_w: paddle.Tensor,
    num_bits: int,
    size_k: int,
    size_n: int,
):
    assert q_w.shape == [size_k, size_n]

    pack_factor = get_pack_factor(num_bits)
    assert size_n % pack_factor == 0

    orig_device = q_w.place

    q_w = q_w.cpu().numpy().astype(numpy.uint32)

    q_res = numpy.zeros((size_k, size_n // pack_factor), dtype=numpy.uint32)

    for i in range(pack_factor):
        q_res |= q_w[:, i::pack_factor] << num_bits * i

    q_res = paddle.to_tensor(q_res.astype(numpy.int32), place=orig_device)
    # q_res = q_res.contiguous()

    return q_res


def unpack_cols(
    packed_q_w: paddle.Tensor,
    num_bits: int,
    size_k: int,
    size_n: int,
):
    pack_factor = get_pack_factor(num_bits)
    assert size_n % pack_factor == 0
    assert packed_q_w.shape == (
        size_k, size_n // pack_factor
    ), "packed_q_w.shape = {} size_k = {}, size_n = {} pack_Factor = {}".format(
        packed_q_w.shape, size_k, size_n, pack_factor)

    orig_device = packed_q_w.device

    packed_q_w_cpu = packed_q_w.cpu().numpy().astype(numpy.uint32)
    q_res = numpy.zeros((size_k, size_n), dtype=numpy.uint32)

    mask = (1 << num_bits) - 1
    for i in range(pack_factor):
        vals = packed_q_w_cpu & mask
        packed_q_w_cpu >>= num_bits
        q_res[:, i::pack_factor] = vals

    q_res = paddle.from_numpy(q_res.astype(numpy.int32)).to(orig_device)
    q_res = q_res.contiguous()

    return q_res

def awq_pack(
    q_w: paddle.Tensor,
    num_bits: int,
    size_k: int,
    size_n: int,
):
    assert q_w.shape == (size_k, size_n)

    # Interleave column dim (for the dequantize code) and pack it to int32
    if num_bits == 4:
        interleave = numpy.array([0, 2, 4, 6, 1, 3, 5, 7])
    elif num_bits == 8:
        interleave = numpy.array([0, 2, 1, 3])
    else:
        raise Exception("num_bits must be 4 or 8, got {}".format(num_bits))

    q_w = q_w.reshape((-1, len(interleave)))[:, interleave].ravel()
    q_w = q_w.reshape((-1, size_n)).contiguous()

    return pack_cols(q_w, num_bits, size_k, size_n)
