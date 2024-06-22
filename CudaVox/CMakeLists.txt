CMAKE_MINIMUM_REQUIRED(VERSION 3.10 FATAL_ERROR)

PROJECT(cad2vox LANGUAGES CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
FIND_PACKAGE(glm CONFIG REQUIRED)
FIND_PACKAGE(OpenMP REQUIRED)
FIND_PACKAGE(xtensor REQUIRED)

FIND_PACKAGE(CUDAToolkit)
if(CUDAToolkit_FOUND)
   enable_language(CUDA)
   add_compile_definitions(WITH_CUDA)
   SET(CUDA_VOXELIZER_SRCS_CU
  src/CUDA_specific/gpu_voxeliser.cpp
  src/CUDA_specific/voxelize.cu
  src/CUDA_specific/thrust_operations.cu
  src/CUDA_specific/voxelize_solid.cu
  src/CUDA_specific/util_cuda.cpp
)
endif()
set(XTL_INCLUDE_DIRS lib/xtensor-python/include)
SET(CUDA_VOXELIZER_EXECUTABLE cuda_voxelizer)

SET(CUDA_VOXELIZER_SRCS
  src/Common/cpu_voxelizer.cpp
)

file (GLOB PYTHON_FILES "python_bind.cpp")

find_package(pybind11 REQUIRED)
pybind11_add_module(CudaVox
	${CUDA_VOXELIZER_SRCS}
    ${CUDA_VOXELIZER_SRCS_CU}
	${PYTHON_FILES}
)

if(CUDAToolkit_FOUND)
TARGET_LINK_LIBRARIES(CudaVox PRIVATE OpenMP::OpenMP_CXX PRIVATE CUDA::cudart glm::glm PRIVATE xtensor)
set_property(TARGET CudaVox PROPERTY CUDA_ARCHITECTURES 70 72 75 80)
else()
TARGET_LINK_LIBRARIES(CudaVox PRIVATE OpenMP::OpenMP_CXX PRIVATE glm::glm PRIVATE xtensor)
endif()

target_include_directories(CudaVox PUBLIC ${XTL_INCLUDE_DIRS} PUBLIC ${NUMPY_DIR} PUBLIC "include")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
