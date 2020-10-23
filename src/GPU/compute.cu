#include "compute.hh"

__global__ void test() {
    std::cout << "GPU!" << std::endl;
}
