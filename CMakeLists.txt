cmake_minimum_required(VERSION 3.25)
project(double-blocking-getrf VERSION 0.1.0 LANGUAGES C CXX CUDA)

# --- Standard Settings ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# --- Include Directory ---
include_directories(${PROJECT_SOURCE_DIR}/external/Qui1Framework/include) # Corrected path

# --- CUDA Architectures ---
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Default architectures if not specified by the user/environment
    set(CMAKE_CUDA_ARCHITECTURES 87 89 90) # Example: Ampere, Hopper. Adjust as needed.
endif()
message(STATUS "Targeting CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

add_subdirectory(external/Qui1Framework)

# --- main build target ---
add_executable(main main.cu)
target_link_libraries(main PRIVATE fmt CUDA::cudart CUDA::cusolver CUDA::curand) # Link against fmt library

# --- CUDA Check (Optional, informational) ---
find_package(CUDAToolkit)
if (CUDAToolkit_FOUND)
    message(STATUS "CUDA Toolkit found (Version: ${CUDAToolkit_VERSION})")
else()
    message(WARNING "CUDA Toolkit not found by find_package. Compilation might still work if nvcc is in PATH.")
endif()
