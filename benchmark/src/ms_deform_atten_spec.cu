#include "ms_deform_atten_spec.h"
#include <cuda/std/limits>
#include <cstdio>

namespace
{
constexpr int kWarpSize = 32;
constexpr int kVecBytes = sizeof(int4);
constexpr int kVecHalf = kVecBytes / sizeof(__half);
constexpr int kVecHalf2 = kVecHalf / 2;
constexpr int kNeighbors = 4;
constexpr int kMaxLevels = 4;

__device__ __forceinline__ size_t align_up(size_t offset, size_t alignment)
{
    return (offset + alignment - 1) & ~(alignment - 1);
}
} // namespace

__global__ void ms_deform_attention_kernel_general(
    const __half2 *__restrict__ data_value,
    __half2 *__restrict__ data_col,
    const __half2 *__restrict__ data_reference_points,
    const __half2 *__restrict__ data_sampling_offsets,
    const __half *__restrict__ data_attn_weight,
    const int32_t *__restrict__ data_spatial_shapes,
    const int num_query,
    const int spatial_size,
    const int total_query,
    const int num_heads,
    const int num_levels,
    const int num_points,
    const int channels)
{
    extern __shared__ uint8_t shared_mem[];

    const int lane_id = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int per_block_points = blockDim.y;
    const int global_point_id = blockIdx.x * per_block_points + warp_id;
    if (global_point_id >= total_query)
    {
        return;
    }

    const int lp_num = num_levels * num_points;
    const int hlp_num = num_heads * lp_num;
    const int head_dim = channels / num_heads;
    if (head_dim * num_heads != channels)
    {
        if (lane_id == 0 && warp_id == 0 && blockIdx.x == 0)
        {
            printf("ms_deform_attention: embed dims must be divisible by num_heads.\n");
        }
        return;
    }
    if (head_dim % kVecHalf != 0)
    {
        if (lane_id == 0 && warp_id == 0 && blockIdx.x == 0)
        {
            printf("ms_deform_attention: head_dim %d must be divisible by %d for int4 path.\n", head_dim, kVecHalf);
        }
        return;
    }
    if (num_levels > kMaxLevels)
    {
        if (lane_id == 0 && warp_id == 0 && blockIdx.x == 0)
        {
            printf("ms_deform_attention: num_levels %d exceeds supported %d.\n", num_levels, kMaxLevels);
        }
        return;
    }
    const int batch_idx = global_point_id / num_query;
    const int valueout_offset = global_point_id * channels;

    size_t smem_offset = 0;
    smem_offset = align_up(smem_offset, alignof(__half));
    __half *attn_weights = reinterpret_cast<__half *>(shared_mem + smem_offset);
    smem_offset += sizeof(__half) * per_block_points * hlp_num;

    smem_offset = align_up(smem_offset, alignof(__half));
    __half *interp_weights = reinterpret_cast<__half *>(shared_mem + smem_offset);
    smem_offset += sizeof(__half) * per_block_points * hlp_num * kNeighbors;

    smem_offset = align_up(smem_offset, alignof(int));
    int *interp_offsets = reinterpret_cast<int *>(shared_mem + smem_offset);
    smem_offset += sizeof(int) * per_block_points * hlp_num * kNeighbors;

    smem_offset = align_up(smem_offset, alignof(int8_t));
    int8_t *valid_mask = reinterpret_cast<int8_t *>(shared_mem + smem_offset);

    __shared__ int2 spatial_shapes[kMaxLevels];
    __shared__ int level_offset[kMaxLevels + 1];

    if (lane_id == 0 && warp_id == 0)
    {
        level_offset[0] = batch_idx * spatial_size * channels;
        for (int i = 0; i < num_levels; ++i)
        {
            spatial_shapes[i] = reinterpret_cast<const int2 *>(data_spatial_shapes)[i];
            const int level_elems = spatial_shapes[i].x * spatial_shapes[i].y * channels;
            level_offset[i + 1] = level_offset[i] + level_elems;
        }
    }
    __syncthreads();

    const int warp_stride = num_heads * lp_num;
    __half *attn_weights_warp = attn_weights + warp_id * warp_stride;
    __half *interp_weights_warp = interp_weights + warp_id * warp_stride * kNeighbors;
    int *interp_offsets_warp = interp_offsets + warp_id * warp_stride * kNeighbors;
    int8_t *valid_mask_warp = valid_mask + warp_id * warp_stride;

    const int attn_base = global_point_id * hlp_num;
    const __half2 reference_point_xy = __ldg(data_reference_points + global_point_id);
    const __half2 *sampling_offsets = data_sampling_offsets + global_point_id * hlp_num;

    for (int head = lane_id; head < num_heads; head += kWarpSize)
    {
        __half max_weight = -cuda::std::numeric_limits<__half>::infinity();
        const int head_offset = attn_base + head * lp_num;
        for (int point = 0; point < lp_num; ++point)
        {
            const __half val = data_attn_weight[head_offset + point];
            max_weight = __hmax(max_weight, val);
        }
        __half sum_weight = __float2half(0.f);
        __half *dst = attn_weights_warp + head * lp_num;
        for (int point = 0; point < lp_num; ++point)
        {
            const __half exp_val = hexp(__hsub(data_attn_weight[head_offset + point], max_weight));
            dst[point] = exp_val;
            sum_weight = __hadd(sum_weight, exp_val);
        }
        const __half inv_sum = hrcp(sum_weight);
        for (int point = 0; point < lp_num; ++point)
        {
            dst[point] = __hmul(dst[point], inv_sum);
        }
    }
    __syncwarp();

    const __half2 scale = __floats2half2_rn(1.0f, 1.0f);
    for (int idx = lane_id; idx < hlp_num; idx += kWarpSize)
    {
        const int head = idx / lp_num;
        const int point = idx % lp_num;
        const int level = point / num_points;

        const int2 shape = spatial_shapes[level];
        const int32_t spatial_h = shape.x;
        const int32_t spatial_w = shape.y;
        const int32_t h_max = spatial_h - 1;
        const int32_t w_max = spatial_w - 1;
        const __half2 sampling_offset_xy = __ldg(sampling_offsets + idx);
        const __half2 offset = __hsub2(sampling_offset_xy, __float2half2_rn(0.5f));
        const __half2 grid_xy = __hfma2(reference_point_xy,
                                        __floats2half2_rn(static_cast<float>(spatial_w),
                                                          static_cast<float>(spatial_h)),
                                        offset);
        const __half2 condition = __hmul2(
            __hgt2(grid_xy, __float2half2_rn(-1.f)),
            __hlt2(grid_xy, __floats2half2_rn(static_cast<float>(spatial_w),
                                              static_cast<float>(spatial_h))));
        const int head_point_index = head * lp_num + point;
        valid_mask_warp[head_point_index] = 0;
        if (__low2float(condition) * __high2float(condition) == 0.f)
        {
            continue;
        }

        const __half2 grid_low = h2floor(grid_xy);
        const __half2 grid_high = __hadd2(grid_low, scale);
        const __half2 lhw = __hsub2(grid_xy, grid_low);
        const __half2 nhw = __hsub2(scale, lhw);

        const int row_low = __high2float(grid_low) * spatial_w;
        const int row_high = row_low + spatial_w;
        const int col_low = __low2float(grid_low);
        const int col_high = col_low + 1;

        int valid_bits = 1 << 5;
        const int level_base = level_offset[level];
        __half *weight_base = interp_weights_warp + head_point_index * kNeighbors;
        int *offset_base = interp_offsets_warp + head_point_index * kNeighbors;
        const __half attn_weight = attn_weights_warp[head * lp_num + point];

        if (__hge(__high2half(grid_low), __float2half(0.f)) && __hge(__low2half(grid_low), __float2half(0.f)))
        {
            valid_bits |= 1;
            const int offset_00 = level_base + (row_low + col_low) * channels;
            offset_base[0] = offset_00;
            weight_base[0] = __hmul(__hmul(__high2half(nhw), __low2half(nhw)), attn_weight);
        }
        if (__hge(__high2half(grid_low), __float2half(0.f)) && __hle(__low2half(grid_high), w_max))
        {
            valid_bits |= (1 << 1);
            const int offset_01 = level_base + (row_low + col_high) * channels;
            offset_base[1] = offset_01;
            weight_base[1] = __hmul(__hmul(__high2half(nhw), __low2half(lhw)), attn_weight);
        }
        if (__hle(__high2half(grid_high), __float2half(static_cast<float>(h_max))) &&
            __hge(__low2half(grid_low), __float2half(0.f)))
        {
            valid_bits |= (1 << 2);
            const int offset_10 = level_base + (row_high + col_low) * channels;
            offset_base[2] = offset_10;
            weight_base[2] = __hmul(__hmul(__high2half(lhw), __low2half(nhw)), attn_weight);
        }
        if (__hle(__high2half(grid_high), __float2half(static_cast<float>(h_max))) &&
            __hle(__low2half(grid_high), w_max))
        {
            valid_bits |= (1 << 3);
            const int offset_11 = level_base + (row_high + col_high) * channels;
            offset_base[3] = offset_11;
            weight_base[3] = __hmul(__hmul(__high2half(lhw), __low2half(lhw)), attn_weight);
        }
        valid_mask_warp[head_point_index] = valid_bits;
    }
    __syncwarp();

    const __half *value = reinterpret_cast<const __half *>(data_value);
    __half *output = reinterpret_cast<__half *>(data_col);

    for (int head = 0; head < num_heads; ++head)
    {
        __half *head_out = output + valueout_offset + head * head_dim;
        const int head_base_offset = head * head_dim;
        const int vec_chunks = head_dim / kVecHalf;
        for (int vec_idx = lane_id; vec_idx < vec_chunks; vec_idx += kWarpSize)
        {
            __half2 acc[kVecHalf2];
#pragma unroll
            for (int i = 0; i < kVecHalf2; ++i)
            {
                acc[i] = __float2half2_rn(0.f);
            }
            const int channel_offset = head_base_offset + vec_idx * kVecHalf;
            for (int point = 0; point < lp_num; ++point)
            {
                const int head_point_index = head * lp_num + point;
                const int8_t valid = valid_mask_warp[head_point_index];
                if (!valid)
                {
                    continue;
                }
                const int neighbor_base = head_point_index * kNeighbors;
#pragma unroll
                for (int nb = 0; nb < kNeighbors; ++nb)
                {
                    if (!(valid & (1 << nb)))
                    {
                        continue;
                    }
                    const int total_offset = interp_offsets_warp[neighbor_base + nb] + channel_offset;
                    const int4 vec_val = __ldg(reinterpret_cast<const int4 *>(value + total_offset));
                    const __half2 *vec_half2 = reinterpret_cast<const __half2 *>(&vec_val);
                    const __half2 weight = __half2half2(interp_weights_warp[neighbor_base + nb]);
#pragma unroll
                    for (int i = 0; i < kVecHalf2; ++i)
                    {
                        acc[i] = __hfma2(vec_half2[i], weight, acc[i]);
                    }
                }
            }
            reinterpret_cast<int4 *>(head_out)[vec_idx] = *reinterpret_cast<const int4 *>(acc);
        }
        __syncwarp();
    }
}

void ms_deformable_im2col_cuda_spec_h2(
    const __half2 *data_value, const int32_t *data_spatial_shapes,
    const __half2 *data_reference_points, const __half2 *data_sampling_offsets,
    const __half *data_attn_weight, const int batch_size,
    const int spatial_size, const int num_heads, int channels,
    const int num_levels, const int num_query, const int num_point,
    const int points_per_group, __half2 *data_col, cudaStream_t stream)
{
    (void)points_per_group;

    int num_points = num_query * batch_size;
    int block_num = (num_points + 3) / 4;
    int per_block_points = 4;
    if (num_points % 8 == 0)
    {
        block_num = num_points / 8;
        per_block_points = 8;
    }
    else if (num_points % 4 == 0)
    {
        block_num = num_points / 4;
        per_block_points = 4;
    }

    if (channels % num_heads != 0)
    {
        printf("FlashAD kernel requires channels (embed_dims) to be divisible by num_heads.\n");
        return;
    }

    dim3 gridDim(block_num, 1, 1);
    dim3 blockDim(kWarpSize, per_block_points, 1);

    const int lp_num = num_levels * num_point;
    const int hlp_num = num_heads * lp_num;
    const int per_block_warp = per_block_points;
    const size_t attn_elems = static_cast<size_t>(per_block_warp) * hlp_num;
    const size_t interp_elems = attn_elems * kNeighbors;

    auto align_up_host = [](size_t offset, size_t alignment) {
        return (offset + alignment - 1) & ~(alignment - 1);
    };

    size_t shared_bytes = 0;
    shared_bytes = align_up_host(shared_bytes, alignof(__half));
    shared_bytes += sizeof(__half) * attn_elems;
    shared_bytes = align_up_host(shared_bytes, alignof(__half));
    shared_bytes += sizeof(__half) * interp_elems;
    shared_bytes = align_up_host(shared_bytes, alignof(int));
    shared_bytes += sizeof(int) * interp_elems;
    shared_bytes = align_up_host(shared_bytes, alignof(int8_t));
    shared_bytes += sizeof(int8_t) * attn_elems;

    ms_deform_attention_kernel_general<<<gridDim, blockDim, shared_bytes, stream>>>(
        data_value,
        data_col,
        data_reference_points,
        data_sampling_offsets,
        data_attn_weight,
        data_spatial_shapes,
        num_query,
        spatial_size,
        num_points,
        num_heads,
        num_levels,
        num_point,
        channels);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("error in ms_deformable_im2col_cuda: %s\n", cudaGetErrorString(err));
    }
}
