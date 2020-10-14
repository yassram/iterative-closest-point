#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>

#include "cpu.hh"

MatrixXd load_matrix(const char *filename);
void write_matrix(MatrixXd matrix);
