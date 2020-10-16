#include <iostream>
#include "cpu.hh"
#include "load.hh"
#include <complex>


int main(int argc, char* argv[])
{
    if (argc < 4) {
        std::cout << "Usage: ./icp [path_to_ref_cloud] [path_to_transform_cloud] [nb_iter]\n";
        return -1;
    }

    int max_iter = atoi(argv[3]);

    auto matrix_ref = load_matrix(argv[1]);
    auto matrix_tr1 = load_matrix(argv[2]);

    CPU::ICP icp(matrix_ref, matrix_tr1, max_iter);
    icp.find_corresponding();
    write_matrix(icp.new_p);

    return 0;
}
