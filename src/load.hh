#include <fstream>
#include <string>
#include <limits>
#include <iostream>

#include "GPU/gpu.hh"

GPU::Matrix load_matrix(const char *filename, double min_coord[3], double max_coord[3]);
void write_matrix(GPU::Matrix matrix);
