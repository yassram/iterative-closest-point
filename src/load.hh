#include <fstream>
#include <string>
#include <limits>
#include <iostream>

#include "cpu.hh"

MatrixXd load_matrix(const char *filename, double min_coord[3], double max_coord[3]);
void write_matrix(MatrixXd matrix);
