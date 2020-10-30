#include <fstream>
#include <string>
#include <limits>
#include <iostream>

#include "GPU/gpu.hh"
#include "GPU/compute.hh"

GPU::Matrix load_matrix(const char *filename, GPU::Matrix min_coord, GPU::Matrix max_coor);
void write_matrix(GPU::Matrix matrix);
void write_matrix(Eigen::MatrixXd matrix);
MatrixXd cpu_load_matrix(const char *filename, double min_coord[3], double max_coord[3]);
