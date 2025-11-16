cmake_minimum_required(VERSION 3.16)
project(FlashAD LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(FLASHAD_GPU_ARCH "86" CACHE STRING "Target SM architecture, e.g. 80 for A100, 86 for RTX 30 series.")

function(flashad_set_arch target)
    if (CMAKE_VERSION VERSION_LESS 3.18)
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_${FLASHAD_GPU_ARCH},code=sm_${FLASHAD_GPU_ARCH}>)
    else ()
        set_property(TARGET ${target} PROPERTY CUDA_ARCHITECTURES ${FLASHAD_GPU_ARCH})
    endif ()
endfunction()

add_library(flashad_kernel STATIC src/ms_deform_atten_spec.cu)
target_include_directories(flashad_kernel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
flashad_set_arch(flashad_kernel)

add_executable(flashad_demo demo/flashad_demo.cu)
target_include_directories(flashad_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(flashad_demo PRIVATE flashad_kernel)
flashad_set_arch(flashad_demo)
