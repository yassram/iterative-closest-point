#pragma once
#include <fstream>
#include <string>
#include <iostream>
#include <exception>

#include "GPU/gpu.hh"

GPU::Matrix load_matrix(const char *filename);
void write_matrix(GPU::Matrix matrix);
