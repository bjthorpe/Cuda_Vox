#include "xtensor-python/pyarray.hpp"     // Numpy bindings
#include <xtensor/xarray.hpp>
#include <xtensor/xcontainer.hpp>
#include <xtensor/xadapt.hpp>
// Utils
#include "util.h"
#include "cpu_voxelizer.h"
//#include "util_io.h"
#include "util_cuda.h"
#include "managed_mem.h"

// Forward declaration of CUDA functions
float* meshToGPU_thrust(const Mesh *mesh); // METHOD 3 to transfer triangles can be found in thrust_operations.cu(h)
void cleanup_thrust();
void voxelize(const voxinfo & v, float* triangle_data, long* Greyscale, unsigned int* vtable, bool useThrustPath, unsigned short* result_ptr, bool use_tetra);
void voxelize_solid(const voxinfo& v, float* triangle_data, unsigned int* vtable, bool useThrustPath);


// Function to take in the result array and write greyscale values to it. This is done in parallel on the CPU at present.
xt::pyarray<unsigned short> write_greyscale_GPU(xt::pyarray<unsigned short> result,voxinfo info, unsigned int* vtable){

  size_t voxels_seen = 0;
  const size_t write_stats_25 = (size_t(info.gridsize.x) * size_t(info.gridsize.y) * size_t(info.gridsize.z)) / 4.0f;
  fprintf(stdout, "writing greyscale values to array\n");
  
#pragma omp parallel for 
  for (int x = 0; x < info.gridsize.x; x++) {
    for (int y = 0; y < info.gridsize.y; y++) {
      for (int z = 0; z < info.gridsize.z; z++) {
	voxels_seen++;
	if (voxels_seen == write_stats_25) { fprintf(stdout, "25%%...\n");}
	else if (voxels_seen == write_stats_25 * size_t(2)) { fprintf(stdout, "50%%...\n");}
	else if (voxels_seen == write_stats_25 * size_t(3)) {fprintf(stdout, "75%%...\n");}
	if (checkVoxel(x, y, z, info.gridsize, vtable)) {
	  result(x,y,z) = 255;
	}
      }
    }	  
  }

  return result;
}

// Note: A dummy version of this function is defined in python_bind.cpp for the non-CUDA case
xt::pyarray<unsigned short> voxelise_GPU(Mesh* themesh ,voxinfo info, unsigned int* vtable, size_t vtable_size,bool use_tetra, bool useThrustPath, bool solid){
  
  xt::pyarray<unsigned short> result;
  unsigned short* result_ptr; // pointer to the start of the pyarray
  size_t result_size = info.gridsize.x*info.gridsize.y*info.gridsize.z*sizeof(unsigned short);
  fprintf(stdout, "\n## TRANSFERRING MESH ELEMENTS TO GPU \n");
  
  float* device_elements;
  long* device_greyscale;
  // Transfer triangles to GPU using either thrust or managed cuda memory. 
  // Greyscale is curently only CUDA manage memory as it is much less data.
  if (useThrustPath) { 
    device_elements = meshToGPU_thrust(themesh);
    device_greyscale = GreyscaleToGPU_managed(themesh);
    // ALLOCATE MEMORY ON HOST
    fprintf(stdout, "[Voxel Grid] Allocating %s kB of page-locked HOST memory for Voxel Grid\n",readableSize(vtable_size).c_str());
			checkCudaErrors(cudaHostAlloc((void**)&vtable, vtable_size, cudaHostAllocDefault));
            fprintf(stdout, "[Voxel Grid] Allocating %s of page-locked HOST memory for Result\n", readableSize(result_size).c_str());
			checkCudaErrors(cudaHostAlloc((void**)&result_ptr, result_size, cudaHostAllocDefault));
  }
  else { // Use Managed Memory
    if(use_tetra){
      device_elements = meshToGPU_managed_tets(themesh);
      device_greyscale = GreyscaleToGPU_managed(themesh);
    }
    else{
      device_elements = meshToGPU_managed_tri(themesh);
      device_greyscale = GreyscaleToGPU_managed(themesh); 
    }
    fprintf(stdout, "[Voxel Grid] Allocating %s of CUDA-managed UNIFIED memory for Voxel Grid\n", readableSize(vtable_size).c_str());
    checkCudaErrors(cudaMallocManaged((void**)&vtable, vtable_size));
    fprintf(stdout, "[Voxel Grid] Allocating %s of CUDA-managed UNIFIED memory for Result\n", readableSize(result_size).c_str());
    checkCudaErrors(cudaMallocManaged((void**)&result_ptr, result_size));
  }
  
  fprintf(stdout, "\n## GPU VOXELISATION \n");
  if (solid){
    voxelize_solid(info, device_elements, vtable, useThrustPath);
    // update this at some point to actually create result with greyscale on the GPU
    result = xt::zeros<unsigned short>({info.gridsize.x,info.gridsize.y,info.gridsize.z});
    result = write_greyscale_GPU(result, info, vtable);
  }
  else{
    voxelize(info, device_elements, device_greyscale, vtable, useThrustPath, result_ptr, use_tetra);
  std::vector<std::size_t> shape = {info.gridsize.x,info.gridsize.y,info.gridsize.z};
  result = xt::adapt(result_ptr, info.gridsize.x*info.gridsize.y*info.gridsize.z, xt::no_ownership(), shape);
  }
  return result;
}	    

