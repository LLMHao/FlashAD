#include "ms_deform_atten_spec.h"
#include <cuda/std/limits>
#include <cstdio>
using namespace std;
/**
 * warp level interpolate, support fp16 128 channels.
 * */
__device__ void mix_warp_level_bi_interpolate_device_emb256_h2(
    const int8_t valid,
    const half2 *point_weights,
    const int *point_offsets,
    const __half2 *__restrict__ value_in,
    const int ilane,
    __half2 *interpolation_value)
{
    constexpr int num_loop_per_thread = 4;
    constexpr int num_data_per_thread = 8;
    const half *base_ptr = ((half *)value_in + ilane * num_data_per_thread);
#pragma unroll
    for (int weight_idx = 0; weight_idx < 4; weight_idx++)
    {
        if (valid & (1 << weight_idx))
        {
            const int4 temp = __ldg(reinterpret_cast<const int4 *>(base_ptr + point_offsets[weight_idx]));

#pragma unroll
            for (int i = 0; i < num_loop_per_thread; i++)
            {
                interpolation_value[i] = __hfma2(*((const half2 *)&temp + i), point_weights[weight_idx], interpolation_value[i]);
            }
        }
    }
}

template <int NUM_LEVELS, int NUM_POINTS, int PER_BLOCK_POINTS, int NUM_HEADS = 8, int EMBED_DIMS = 256>
__global__ void ms_deform_attention_spec_kernel_h2(
    const __half2 *__restrict__ data_value,
    __half2 *__restrict__ data_col,
    const __half2 *__restrict__ data_reference_points,
    const __half2 *__restrict__ data_sampling_offsets,
    const __half *__restrict__ data_attn_weight,
    const int32_t *data_spatial_shapes,
    const int num_query,
    const int spatial_size,
    const int total_query)
{
    constexpr int MAX_LEVELS = 4;
    constexpr int HLP_NUM = NUM_HEADS * NUM_LEVELS * NUM_POINTS;
    constexpr int LP_NUM = NUM_LEVELS * NUM_POINTS;
    constexpr int PER_HEAD_THREDS = 32 / NUM_HEADS;
    __shared__ __half attn_weights[PER_BLOCK_POINTS][NUM_HEADS][LP_NUM];
    // interpolation weights and offsets
    __shared__ __half interpolation_weights[PER_BLOCK_POINTS][NUM_HEADS][LP_NUM][4];
    __shared__ int interpolation_offsets[PER_BLOCK_POINTS][NUM_HEADS][LP_NUM][4];
    __shared__ int8_t valid_mask[PER_BLOCK_POINTS][HLP_NUM];
    __shared__ int2 spatial_shapes[MAX_LEVELS];
    __shared__ int level_offset[MAX_LEVELS + 4];
    __half2 interpolation_value[4] = {
        __float2half2_rn(0.0f),
        __float2half2_rn(0.0f),
        __float2half2_rn(0.0f),
        __float2half2_rn(0.0f)};

    int lane_id = threadIdx.x;
    int warp_id = threadIdx.y;
    const int global_point_id = blockIdx.x * PER_BLOCK_POINTS + warp_id;
    if (global_point_id >= total_query)
    {
        return;
    }
    const int valueout_offset = global_point_id * EMBED_DIMS;
    __half2 reference_point_xy = __ldg(data_reference_points + global_point_id); // load reference point
    if (lane_id == 0 && warp_id == 0)
    {
        level_offset[0] = ((blockIdx.x * PER_BLOCK_POINTS) / num_query) * spatial_size * EMBED_DIMS;
#pragma unroll
        for (int i = 0; i < NUM_LEVELS; i++)
        {
            spatial_shapes[i] = reinterpret_cast<const int2 *>(data_spatial_shapes)[i];
            level_offset[i + 1] = level_offset[i] + spatial_shapes[i].x * spatial_shapes[i].y * EMBED_DIMS;
        }
    }
    __syncthreads();
    //  saft softmax. 1: max, 2: exp, 3: sum, 4: div
    if (lane_id < NUM_HEADS)
    {
        __half logits[LP_NUM];
        __half sum_weight = 0.f;
        int data_weight_ptr = global_point_id * HLP_NUM + lane_id * LP_NUM;
        __half max_weight = -cuda::std::numeric_limits<__half>::infinity();
#pragma unroll
        for (int j_point = 0; j_point < LP_NUM; j_point++)
        {
            logits[j_point] = data_attn_weight[data_weight_ptr + j_point];
            max_weight = __hmax(max_weight, logits[j_point]);
        }
#pragma unroll
        for (int j_point = 0; j_point < LP_NUM; j_point++)
        {
            const __half exp_val = hexp(__hsub(logits[j_point], max_weight));
            attn_weights[warp_id][lane_id][j_point] = exp_val;
            sum_weight = __hadd(sum_weight, exp_val);
        }
        __half sum_weight_inv = hrcp(sum_weight);
#pragma unroll
        for (int j_point = 0; j_point < LP_NUM; j_point++)
        {
            attn_weights[warp_id][lane_id][j_point] = __hmul(attn_weights[warp_id][lane_id][j_point], sum_weight_inv);
        }
    }
    for (int i_lane_id = lane_id; i_lane_id < HLP_NUM; i_lane_id += 32)
    {
        int i_head = i_lane_id / LP_NUM;
        int j_point = i_lane_id % LP_NUM;
        int j_level = (j_point / NUM_POINTS);
        const int32_t spatial_w = spatial_shapes[j_level].y;
        const int32_t spatial_h = spatial_shapes[j_level].x;
        const int32_t w_minus1 = spatial_w - 1;
        const int32_t h_minus1 = spatial_h - 1;
        const __half2 sampling_offset_xy = __ldg(data_sampling_offsets + global_point_id * HLP_NUM + i_lane_id); // load offset
        const __half2 offset = __hsub2(sampling_offset_xy, __float2half2_rn(0.5f));
        const __half2 wh_im = __hfma2(reference_point_xy,
                                      __floats2half2_rn(static_cast<float>(spatial_w),
                                                        static_cast<float>(spatial_h)),
                                      offset);
        const __half2 condition = __hmul2(
            __hgt2(wh_im, __float2half2_rn(-1.f)),
            __hlt2(wh_im, __floats2half2_rn(static_cast<float>(spatial_w),
                                            static_cast<float>(spatial_h))));
        valid_mask[warp_id][i_lane_id] = 0;
        if (__low2float(condition) * __high2float(condition))
        {
            int valid_bitmask = 1 << 5;
            const __half2 wh_low = h2floor(wh_im);
            const __half2 wh_high = __hadd2(wh_low, __float2half2_rn(1.f));
            const __half2 lwh = __hsub2(wh_im, wh_low);
            const __half2 hwh = __hsub2(__float2half2_rn(1.f), lwh);
            const int h_low_ptr_offset = __high2float(wh_low) * spatial_w;
            const int h_high_ptr_offset = h_low_ptr_offset + spatial_w;
            const int w_low_ptr_offset = __low2float(wh_low);
            const int w_high_ptr_offset = w_low_ptr_offset + 1;
            if (__hge(__high2half(wh_low), __float2half(0.f)) && __hge(__low2half(wh_low), __float2half(0.f)))
            {
                valid_bitmask |= 1;
                const int offset_00 = (h_low_ptr_offset + w_low_ptr_offset) * EMBED_DIMS;
                interpolation_offsets[warp_id][i_head][j_point][0] = offset_00;
                interpolation_weights[warp_id][i_head][j_point][0] = __hmul(__hmul(__high2half(hwh), __low2half(hwh)), attn_weights[warp_id][i_head][j_point]);
            }
            if (__hge(__high2half(wh_low), __float2half(0.f)) && __hle(__low2half(wh_high), w_minus1))
            {
                valid_bitmask |= (1 << 1);
                const int offset_01 = (h_low_ptr_offset + w_high_ptr_offset) * EMBED_DIMS;
                interpolation_offsets[warp_id][i_head][j_point][1] = offset_01;
                interpolation_weights[warp_id][i_head][j_point][1] = __hmul(__hmul(__high2half(hwh), __low2half(lwh)), attn_weights[warp_id][i_head][j_point]);
            }
            if (__hle(__high2half(wh_high),
                      __float2half(static_cast<float>(h_minus1))) &&
                __hge(__low2half(wh_low), __float2half(0.f)))
            {
                valid_bitmask |= (1 << 2);
                const int offset_10 = (h_high_ptr_offset + w_low_ptr_offset) * EMBED_DIMS;
                interpolation_offsets[warp_id][i_head][j_point][2] = offset_10;
                interpolation_weights[warp_id][i_head][j_point][2] = __hmul(__hmul(__high2half(lwh), __low2half(hwh)), attn_weights[warp_id][i_head][j_point]);
            }
            if (__hle(__high2half(wh_high), h_minus1) &&
                __hle(__low2half(wh_high), w_minus1))
            {
                valid_bitmask |= (1 << 3);
                const int offset_11 = (h_high_ptr_offset + w_high_ptr_offset) * EMBED_DIMS;
                interpolation_offsets[warp_id][i_head][j_point][3] = offset_11;
                interpolation_weights[warp_id][i_head][j_point][3] = __hmul(__hmul(__high2half(lwh), __low2half(lwh)), attn_weights[warp_id][i_head][j_point]);
            }
            valid_mask[warp_id][i_lane_id] = valid_bitmask;
        }
    }
    __syncwarp();
    int i_head = lane_id / PER_HEAD_THREDS;
    const int valid_offset = i_head * LP_NUM;
    __half2 *value_out = (__half2 *)((__half *)data_col + valueout_offset);
    for (int j_point = 0; j_point < LP_NUM; j_point++)
    {
        __half *point_interpolation_weights = interpolation_weights[warp_id][i_head][j_point];
        int8_t valid = valid_mask[warp_id][valid_offset + j_point];
        if (valid)
        {
            int j_level = (j_point / NUM_POINTS);
            int *point_offsets = reinterpret_cast<int *>(interpolation_offsets[warp_id][i_head][j_point]);
            const half2 point_weights[4] = {__half2half2(point_interpolation_weights[0]),
                                            __half2half2(point_interpolation_weights[1]),
                                            __half2half2(point_interpolation_weights[2]),
                                            __half2half2(point_interpolation_weights[3])};
            mix_warp_level_bi_interpolate_device_emb256_h2(
                valid,
                point_weights,
                point_offsets,
                (__half2 *)((__half *)data_value + level_offset[j_level]),
                lane_id,
                interpolation_value);
        }
    }
    *reinterpret_cast<int4 *>((half *)value_out + lane_id * 8) = *reinterpret_cast<int4 *>(interpolation_value);
}

void ms_deformable_im2col_cuda_spec_h2(
    const __half2 *data_value, const int32_t *data_spatial_shapes,
    const __half2 *data_reference_points, const __half2 *data_sampling_offsets,
    const __half *data_attn_weight, const int batch_size,
    const int spatial_size, const int num_heads, int channels,
    const int num_levels, const int num_query, const int num_point,
    const int points_per_group, __half2 *data_col, cudaStream_t stream)
{
    int num_points = num_query * batch_size;
    // total points/ each block process 4 points
    int block_num = 1;
    int per_block_points = 1;
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
    dim3 gridDim(block_num, 1, 1);
    dim3 blockDim(32, per_block_points, 1); // each block process 4 points
    if (num_levels == 1)
    {
        if (num_point == 4 && num_heads == 8 && per_block_points == 4)
        {
            ms_deform_attention_spec_kernel_h2<1, 4, 4><<<gridDim, blockDim, 0, stream>>>(
                data_value,
                data_col,
                data_reference_points,
                data_sampling_offsets,
                data_attn_weight,
                data_spatial_shapes,
                num_query,
                spatial_size,
                num_points);
        }
        else if (num_point == 4 && num_heads == 8 && per_block_points == 8)
        {
            ms_deform_attention_spec_kernel_h2<1, 4, 8><<<gridDim, blockDim, 0, stream>>>(
                data_value,
                data_col,
                data_reference_points,
                data_sampling_offsets,
                data_attn_weight,
                data_spatial_shapes,
                num_query,
                spatial_size,
                num_points);
        }
        else if (num_point == 6 && num_heads == 8 && per_block_points == 4)
        {
            ms_deform_attention_spec_kernel_h2<1, 6, 4><<<gridDim, blockDim, 0, stream>>>(
                data_value,
                data_col,
                data_reference_points,
                data_sampling_offsets,
                data_attn_weight,
                data_spatial_shapes,
                num_query,
                spatial_size,
                num_points);
        }
        else if (num_point == 6 && num_heads == 8 && per_block_points == 8)
        {
            ms_deform_attention_spec_kernel_h2<1, 6, 8><<<gridDim, blockDim, 0, stream>>>(
                data_value,
                data_col,
                data_reference_points,
                data_sampling_offsets,
                data_attn_weight,
                data_spatial_shapes,
                num_query,
                spatial_size,
                num_points);
        }
    }
    else if (num_levels == 2)
    {
        if (num_point == 4 && num_heads == 8 && per_block_points == 4)
        {
            ms_deform_attention_spec_kernel_h2<2, 4, 4><<<gridDim, blockDim, 0, stream>>>(
                data_value,
                data_col,
                data_reference_points,
                data_sampling_offsets,
                data_attn_weight,
                data_spatial_shapes,
                num_query,
                spatial_size,
                num_points);
        }
        else if (num_point == 4 && num_heads == 8 && per_block_points == 8)
        {
            ms_deform_attention_spec_kernel_h2<2, 4, 8><<<gridDim, blockDim, 0, stream>>>(
                data_value,
                data_col,
                data_reference_points,
                data_sampling_offsets,
                data_attn_weight,
                data_spatial_shapes,
                num_query,
                spatial_size,
                num_points);
        }
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("error in ms_deformable_im2col_cuda: %s\n", cudaGetErrorString(err));
    }
}
