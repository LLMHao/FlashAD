#ifndef TENSORRT_OPS_MULTISCALEDEFORMABLEATTNSPECKERNEL_H
#define TENSORRT_OPS_MULTISCALEDEFORMABLEATTNSPECKERNEL_H

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

void ms_deformable_im2col_cuda_spec_h2(
    const __half2 *data_value, const int32_t *data_spatial_shapes,
    const __half2 *data_reference_points, const __half2 *data_sampling_offsets,
    const __half *data_attn_weight, const int batch_size,
    const int spatial_size, const int num_heads, int channels,
    const int num_levels, const int num_query, const int num_point,
    const int points_per_group, __half2 *data_col, cudaStream_t stream);

#endif // TENSORRT_OPS_MULTISCALEDEFORMABLEATTNSPECKERNEL_H