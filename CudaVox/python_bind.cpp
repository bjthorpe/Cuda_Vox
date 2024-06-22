#include <pybind11/pybind11.h>
// XTENSOR Python
#define FORCE_IMPORT_ARRAY                // numpy C api loading
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
#include <xtensor/xarray.hpp>
#include <xtensor/xcontainer.hpp>
#include <xtensor/xadapt.hpp>
// Standard libs
#include <string>
#include <cstdio>
// GLM for maths
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
// Utils
#include "util.h"
#include "timer.h"
#include "cpu_voxelizer.h"


#ifdef WITH_CUDA
xt::pyarray<unsigned short> voxelise_GPU(Mesh* themesh, voxinfo info, unsigned int* vtable,
	size_t vtable_size, bool use_tetra, bool useThrustPath, bool solid);
bool Check_CUDA();
#else
// Forward declaration of function, this function is needed in both CUDA and non-CUDA builds. For builds using CUDA
// it is defined in "gpu_voxelizer.h". However, that file contains CUDA specific code we don't use in the non-CUDA case.
// Therefore in NON-CUDA builds we simply define a dummy function here to satisfy the compiler. This is OK since it will
// never be called due to the way in which we have setup the check_CUDA function (see comments in managed_mem.h for details).
// The reason for this insanity is to allow for a unified source file when compiling with and without CUDA which will,
// hopefully, make future development easier as we only need to maintain one version.
xt::pyarray<unsigned short> voxelise_GPU(Mesh* themesh, voxinfo info, unsigned int* vtable,
	size_t vtable_size, bool use_tetra, bool useThrustPath, bool solid){
	return 0;
}
#endif
/////

namespace py = pybind11;
using namespace pybind11::literals;


// Function to take in the result array and write greyscale values to it. This is done in parallel on the CPU at present.
xt::pyarray<unsigned short> write_greyscale(xt::pyarray<unsigned short> result,voxinfo info, unsigned int* vtable){

  size_t voxels_seen = 0;
  const size_t write_stats_25 = (size_t(info.gridsize.x) * size_t(info.gridsize.y) * size_t(info.gridsize.z)) / 4.0f;
  fprintf(stdout, "writing greyscale values to array\n");
  
#pragma omp parallel for 
  for (signed int x = 0; x < info.gridsize.x; x++) {
    for (signed int y = 0; y < info.gridsize.y; y++) {
      for (signed int z = 0; z < info.gridsize.z; z++) {
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

xt::pyarray<unsigned short> run_vox(xt::pyarray<long> Triangles, xt::pyarray<long> Tetra, xt::pyarray<long> Greyscale, xt::pyarray<float> Points,
		      xt::pyarray<float> Bbox_min, xt::pyarray<float> Bbox_max, xt::pyarray<int> gridsize, bool useThrustPath = false,
		      bool forceCPU = false, bool solid = true, bool use_tetra=true){

  	Timer t; t.start();
	fprintf(stdout, "\n## PROGRAM PARAMETERS \n");
	fflush(stdout);
        xt::pyarray<unsigned short> result;
	fprintf(stdout, "\n## READ MESH \n");
	
        Mesh *themesh = new Mesh(Triangles,Tetra,Greyscale,Points);
	int num_elem;
	// SECTION: Compute some information needed for voxelization (bounding box, unit vector, ...)
	fprintf(stdout, "\n## VOXELISATION SETUP \n");
	// Initialize our own AABox

	glm::vec3 bbmin = Xt_to_glm(Bbox_min);
	glm::vec3 bbmax = Xt_to_glm(Bbox_max);
	glm::uvec3 voxgrid = Xt_to_glm(gridsize);
	AABox<glm::vec3> bbox_mesh(bbmin,bbmax);
	if (use_tetra){
	  num_elem  = themesh->Volume.shape(0);
	}
	else {
	  num_elem = themesh->Surface.shape(0);
	  } 

	voxinfo voxelization_info(createMeshBBCube<glm::vec3>(bbox_mesh), voxgrid, num_elem);
	voxelization_info.print();
	// Compute space needed to hold voxel table (1 voxel / bit)
	size_t vtable_size = static_cast<size_t>(ceil(static_cast<size_t>(voxelization_info.gridsize.x) * static_cast<size_t>(voxelization_info.gridsize.y) * static_cast<size_t>(voxelization_info.gridsize.z)) / 8.0f);
	unsigned int* vtable; // Both voxelization paths (GPU and CPU) need this
#ifdef WITH_CUDA
	bool cuda_ok = Check_CUDA();
#else
	bool cuda_ok = false;
#endif
	if (!forceCPU && cuda_ok)
	{
	  // GPU VOXELIZATION
	  result = voxelise_GPU(themesh, voxelization_info,vtable, vtable_size, use_tetra, useThrustPath, solid);
	}
     else { 
		// CPU VOXELIZATION FALLBACK
       fprintf(stdout, "\n## CPU VOXELISATION \n");
       if (forceCPU) {fprintf(stdout, "[Info] Doing CPU Voxelisation (forced using flag forceCPU)\n");}
       
       if(use_tetra){
	 result = cpu_voxelizer::cpu_voxelize_volume(voxelization_info, themesh);
       }
       else if (!solid) {
	 result = cpu_voxelizer::cpu_voxelize_surface(voxelization_info, themesh);
	 
       }
       else {
	 // allocate zero-filled array
	 vtable = (unsigned int*) calloc(1, vtable_size);
	 fprintf(stdout, "[WARN] Using option solid to auto-fill surface data.\n");
	 fprintf(stdout, "This option is quite slow and not very robust.\n Also custom greyscale values will be ignored.\n");
	 cpu_voxelizer::cpu_voxelize_surface_solid(voxelization_info, themesh, vtable);
	 result= xt::zeros<unsigned short>({voxelization_info.gridsize.x,voxelization_info.gridsize.y,voxelization_info.gridsize.z});
	 result = write_greyscale(result, voxelization_info, vtable);
       }
     }
	
	fprintf(stdout, "\n## STATS \n");
	t.stop(); fprintf(stdout, "[Perf] Total runtime: %.1f ms \n", t.elapsed_time_milliseconds);
	return result;
}


PYBIND11_MODULE(CudaVox, m) {
  
  xt::import_numpy();
    // Optional docstring
    m.doc() = "python  link into CudaVox";
    m.def("run",&run_vox,"function to perform the voxelization",
	  "Triangles"_a, "Tetra"_a, "Greyscale"_a,
	  "Points"_a, "Bbox_min"_a,
	  "Bbox_max"_a,"gridsize"_a, "useThrustPath"_a = false,
	  "forceCPU"_a = false, "solid"_a = false,
	  "use_tetra"_a =true);
#ifdef WITH_CUDA
    m.def("Check_CUDA",&Check_CUDA,"function to check if a CUDA GPU is present.");
#endif
}
