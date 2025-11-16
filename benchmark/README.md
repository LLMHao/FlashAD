# FlashAD
自动驾驶极致算法加速库

## 快速开始（RTX 3090）

```bash
cd ./benchmark
cmake -S . -B build -DFLASHAD_GPU_ARCH=86
cmake --build build -j
```

## 推理 Demo

`build/flashad_demo` 会构造一组可控的半精度输入，调用 `ms_deformable_im2col_cuda_spec_h2` 并测量平均耗时。

常用参数：

```bash
./build/flashad_demo \
  --batch 1 \
  --num-query 512 \
  --num-levels 2 \
  --num-points 8 \
  --warmup 10 \
  --iters 100 \
  --level-shapes 64x64,32x32 \
  --dump-output flashad_fp16.bin
```

- `--level-shapes`：逗号分隔的 `HxW`，只取前 `num-levels` 项，决定 `data_spatial_shapes`。
- `--dump-output`：可选，把结果写到二进制文件，方便与其他实现做逐元素对比。
- demo 固定 `num_heads=8`、`embed_dims=256`，与当前 FlashAD kernel 的模板参数保持一致。

## 与 NVIDIA 官方实现对比

1. **获取官方插件代码**：TensorRT 开源仓库提供 `plugin/multiscaleDeformableAttnPlugin`（`git clone https://github.com/NVIDIA/TensorRT.git`）。在同一台 RTX 3090 上编译即可。
   ```bash
   cd TensorRT
   cmake -S . -B build -DTRT_LIB_DIR=/usr/lib/x86_64-linux-gnu \
     -DTRT_OUT_DIR=$PWD/build/out -DGPU_ARCHS=86 \
     -DBUILD_PLUGINS=ON -DBUILD_SAMPLES=ON \
     -DBUILD_PLUGIN_MULTISCALE_DEFORMABLE_ATTENTION=ON
   cmake --build build -j multiscaleDeformableAttnPlugin sample_ms_deformable_attn
   ```
2. **复现相同输入**：运行 FlashAD demo 一次，保存输入/输出（`--dump-output` + `--seed` 保证可重复）。若需要把输入喂给 TensorRT 插件，可在官方 sample 中加入读取二进制文件的逻辑，或直接复用相同随机种子和 shape。
3. **计时**：在官方 sample 中用 `cudaEventRecord` 包裹 `enqueue`/kernel launch，预热若干次后记录 `avg latency`。保持 batch、query、level/shape 与 FlashAD demo 完全一致。
4. **对比**：
   - 功能正确性：把 sample 的输出写成二进制，与 `flashad_fp16.bin`（或 demo 的 stdout checksum）比较；允许细小的 fp16 数值偏差（1e-2）。
   - 性能：记录两边的平均耗时、吞吐（queries/s），建议附上 `nvidia-smi -q -d CLOCK` 确认频率一致，必要时用 Nsight Compute 抓取瓶颈指标。

这样即可在 3090 上获得 FlashAD kernel 与 NVIDIA 官方 TensorRT 插件的“同形状、同输入、同计时方法”的对比结果。
