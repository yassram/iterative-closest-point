#include <fstream>
#include <string>
#include <limits>
#include <iostream>

#include "GPU/gpu.hh"
#include "GPU/compute.hh"

GPU::Matrix load_matrix(const char *filename, GPU::Matrix min_coord,
                        GPU::Matrix max_coord);
void write_matrix(GPU::Matrix matrix);
