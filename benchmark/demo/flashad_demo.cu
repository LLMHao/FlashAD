#include "ms_deform_atten_spec.h"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#define CHECK_CUDA(expr)                                                                     \
    do                                                                                       \
    {                                                                                        \
        cudaError_t _err = (expr);                                                           \
        if (_err != cudaSuccess)                                                             \
        {                                                                                    \
            std::cerr << "CUDA error: " << cudaGetErrorString(_err) << " at " << __FILE__    \
                      << ":" << __LINE__ << std::endl;                                       \
            std::exit(EXIT_FAILURE);                                                         \
        }                                                                                    \
    } while (0)

struct DemoConfig
{
    int batch_size = 1;
    int num_heads = 8;
    int num_levels = 2;
    int num_points = 4;
    int num_query = 512;
    int embed_dims = 256;
    int warmup = 10;
    int iters = 100;
    int random_seed = 2024;
    std::string level_shapes_arg = "64x64,32x32,16x16,8x8";
    std::string dump_output_path;
};

std::vector<std::pair<int, int>> parseLevelShapes(const std::string &arg)
{
    std::vector<std::pair<int, int>> shapes;
    size_t start = 0;
    while (start < arg.size())
    {
        size_t delim = arg.find(',', start);
        std::string token = arg.substr(start, delim == std::string::npos ? std::string::npos : delim - start);
        if (!token.empty())
        {
            size_t sep = token.find('x');
            if (sep == std::string::npos)
            {
                throw std::runtime_error("Level shape token must be formatted as HxW.");
            }
            int h = std::stoi(token.substr(0, sep));
            int w = std::stoi(token.substr(sep + 1));
            shapes.emplace_back(h, w);
        }
        if (delim == std::string::npos)
        {
            break;
        }
        start = delim + 1;
    }
    return shapes;
}

void parseArgs(int argc, char **argv, DemoConfig &config)
{
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        auto requireValue = [&](const char *name) -> std::string {
            if (i + 1 >= argc)
            {
                throw std::runtime_error(std::string("Missing value for ") + name);
            }
            return argv[++i];
        };

        if (arg == "--batch")
        {
            config.batch_size = std::stoi(requireValue(arg.c_str()));
        }
        else if (arg == "--num-query")
        {
            config.num_query = std::stoi(requireValue(arg.c_str()));
        }
        else if (arg == "--num-heads")
        {
            config.num_heads = std::stoi(requireValue(arg.c_str()));
        }
        else if (arg == "--num-levels")
        {
            config.num_levels = std::stoi(requireValue(arg.c_str()));
        }
        else if (arg == "--num-points")
        {
            config.num_points = std::stoi(requireValue(arg.c_str()));
        }
        else if (arg == "--warmup")
        {
            config.warmup = std::stoi(requireValue(arg.c_str()));
        }
        else if (arg == "--iters")
        {
            config.iters = std::stoi(requireValue(arg.c_str()));
        }
        else if (arg == "--seed")
        {
            config.random_seed = std::stoi(requireValue(arg.c_str()));
        }
        else if (arg == "--level-shapes")
        {
            config.level_shapes_arg = requireValue(arg.c_str());
        }
        else if (arg == "--dump-output")
        {
            config.dump_output_path = requireValue(arg.c_str());
        }
        else if (arg == "--help" || arg == "-h")
        {
            std::cout << "Usage: flashad_demo [options]\n"
                         "  --batch <int>          Batch size (default 1)\n"
                         "  --num-query <int>      Number of queries per batch (default 512)\n"
                         "  --num-heads <int>      Attention heads (default 8)\n"
                         "  --num-levels <int>     Feature pyramid levels (default 2)\n"
                         "  --num-points <int>     Sampling points per level (default 4)\n"
                         "  --warmup <int>         Warmup iterations (default 10)\n"
                         "  --iters <int>          Timed iterations (default 100)\n"
                         "  --seed <int>           RNG seed (default 2024)\n"
                         "  --level-shapes <spec>  Comma separated list like 64x64,32x32,...\n"
                         "  --dump-output <path>   Optional path to dump output tensor\n";
            std::exit(0);
        }
        else
        {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
}

void fillUniformHalf(std::vector<__half> &dst, float low, float high, std::mt19937 &rng)
{
    std::uniform_real_distribution<float> dist(low, high);
    for (auto &elem : dst)
    {
        elem = __float2half(dist(rng));
    }
}

void fillUniformHalf2(std::vector<__half2> &dst, float low, float high, std::mt19937 &rng)
{
    std::uniform_real_distribution<float> dist(low, high);
    for (auto &elem : dst)
    {
        elem = __floats2half2_rn(dist(rng), dist(rng));
    }
}

float checksumHalf(const std::vector<__half> &data)
{
    double acc = 0.0;
    for (const auto &val : data)
    {
        acc += __half2float(val);
    }
    return static_cast<float>(acc);
}

int main(int argc, char **argv)
{
    DemoConfig config;
    try
    {
        parseArgs(argc, argv, config);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Argument error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    if (config.embed_dims != 256)
    {
        std::cerr << "This demo is hardcoded for embed_dims == 256 (FlashAD kernel constraint)." << std::endl;
        return EXIT_FAILURE;
    }
    if (config.num_heads != 8)
    {
        std::cerr << "FlashAD kernel currently supports num_heads == 8 only." << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<std::pair<int, int>> level_shapes;
    try
    {
        level_shapes = parseLevelShapes(config.level_shapes_arg);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Failed to parse --level-shapes: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    if (config.num_levels > static_cast<int>(level_shapes.size()))
    {
        std::cerr << "Requested num_levels exceeds provided level shapes." << std::endl;
        return EXIT_FAILURE;
    }

    level_shapes.resize(config.num_levels);

    int spatial_size = 0;
    std::vector<int32_t> spatial_shapes(config.num_levels * 2);
    for (int i = 0; i < config.num_levels; ++i)
    {
        spatial_shapes[2 * i] = level_shapes[i].first;
        spatial_shapes[2 * i + 1] = level_shapes[i].second;
        spatial_size += level_shapes[i].first * level_shapes[i].second;
    }

    const int total_queries = config.num_query * config.batch_size;
    if (total_queries % 4 != 0)
    {
        std::cerr << "num_query * batch must be divisible by 4 to match current kernel launch options." << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Running FlashAD deformable attention kernel with config:\n"
              << "  batch=" << config.batch_size << ", num_query=" << config.num_query
              << ", num_heads=" << config.num_heads << ", num_levels=" << config.num_levels
              << ", num_points=" << config.num_points << ", spatial_size=" << spatial_size << std::endl;

    std::mt19937 rng(config.random_seed);

    const size_t value_elems = static_cast<size_t>(config.batch_size) * spatial_size * config.embed_dims;
    const size_t reference_elems = static_cast<size_t>(config.batch_size) * config.num_query;
    const size_t attn_elems = static_cast<size_t>(config.batch_size) * config.num_query *
                              config.num_heads * config.num_levels * config.num_points;
    const size_t output_elems = static_cast<size_t>(config.batch_size) * config.num_query * config.embed_dims;

    std::vector<__half> h_value(value_elems);
    std::vector<__half2> h_reference(reference_elems);
    std::vector<__half2> h_offsets(attn_elems);
    std::vector<__half> h_attn(attn_elems);
    std::vector<__half> h_output(output_elems, __float2half(0.0f));

    fillUniformHalf(h_value, -1.0f, 1.0f, rng);
    fillUniformHalf2(h_reference, 0.05f, 0.95f, rng);
    fillUniformHalf2(h_offsets, -0.5f, 0.5f, rng);
    fillUniformHalf(h_attn, -1.0f, 1.0f, rng);

    __half *d_value = nullptr;
    __half2 *d_reference = nullptr;
    __half2 *d_offsets = nullptr;
    __half *d_attn = nullptr;
    __half *d_output = nullptr;
    int32_t *d_spatial_shapes = nullptr;

    CHECK_CUDA(cudaMalloc(&d_value, value_elems * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_reference, reference_elems * sizeof(__half2)));
    CHECK_CUDA(cudaMalloc(&d_offsets, attn_elems * sizeof(__half2)));
    CHECK_CUDA(cudaMalloc(&d_attn, attn_elems * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_output, output_elems * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_spatial_shapes, spatial_shapes.size() * sizeof(int32_t)));

    CHECK_CUDA(cudaMemcpy(d_value, h_value.data(), value_elems * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_reference, h_reference.data(), reference_elems * sizeof(__half2), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_offsets, h_offsets.data(), attn_elems * sizeof(__half2), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_attn, h_attn.data(), attn_elems * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_spatial_shapes, spatial_shapes.data(),
                          spatial_shapes.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_output, 0, output_elems * sizeof(__half)));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    for (int i = 0; i < config.warmup; ++i)
    {
        ms_deformable_im2col_cuda_spec_h2(
            reinterpret_cast<const __half2 *>(d_value),
            d_spatial_shapes,
            d_reference,
            d_offsets,
            d_attn,
            config.batch_size,
            spatial_size,
            config.num_heads,
            config.embed_dims,
            config.num_levels,
            config.num_query,
            config.num_points,
            config.num_points,
            reinterpret_cast<__half2 *>(d_output),
            stream);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    cudaEvent_t start_evt, stop_evt;
    CHECK_CUDA(cudaEventCreate(&start_evt));
    CHECK_CUDA(cudaEventCreate(&stop_evt));

    CHECK_CUDA(cudaEventRecord(start_evt, stream));
    for (int i = 0; i < config.iters; ++i)
    {
        ms_deformable_im2col_cuda_spec_h2(
            reinterpret_cast<const __half2 *>(d_value),
            d_spatial_shapes,
            d_reference,
            d_offsets,
            d_attn,
            config.batch_size,
            spatial_size,
            config.num_heads,
            config.embed_dims,
            config.num_levels,
            config.num_query,
            config.num_points,
            config.num_points,
            reinterpret_cast<__half2 *>(d_output),
            stream);
    }
    CHECK_CUDA(cudaEventRecord(stop_evt, stream));
    CHECK_CUDA(cudaEventSynchronize(stop_evt));

    float total_ms = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start_evt, stop_evt));
    const float avg_ms = total_ms / static_cast<float>(config.iters);
    const float qps = (static_cast<float>(config.batch_size) * config.num_query) / (avg_ms / 1000.0f);

    std::cout << "Average latency: " << avg_ms << " ms | Throughput: " << qps << " queries/sec" << std::endl;

    CHECK_CUDA(cudaMemcpyAsync(h_output.data(), d_output, output_elems * sizeof(__half),
                               cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    const float cksum = checksumHalf(h_output);
    std::cout << "Checksum (sum of all fp16 outputs converted to fp32): " << cksum << std::endl;
    std::cout << "Sample outputs (first 8 values): ";
    for (int i = 0; i < 8 && i < static_cast<int>(h_output.size()); ++i)
    {
        std::cout << __half2float(h_output[i]) << " ";
    }
    std::cout << std::endl;

    if (!config.dump_output_path.empty())
    {
        std::ofstream ofs(config.dump_output_path, std::ios::binary);
        ofs.write(reinterpret_cast<const char *>(h_output.data()),
                  h_output.size() * sizeof(__half));
        ofs.close();
        std::cout << "Output tensor dumped to " << config.dump_output_path << std::endl;
    }

    CHECK_CUDA(cudaEventDestroy(start_evt));
    CHECK_CUDA(cudaEventDestroy(stop_evt));
    CHECK_CUDA(cudaStreamDestroy(stream));

    CHECK_CUDA(cudaFree(d_value));
    CHECK_CUDA(cudaFree(d_reference));
    CHECK_CUDA(cudaFree(d_offsets));
    CHECK_CUDA(cudaFree(d_attn));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_spatial_shapes));
    return 0;
}
