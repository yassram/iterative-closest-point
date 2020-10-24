#include <iostream>
#include "header.hh"
#include "../load.hh"
#include <complex>
#include "gpu.hh"
#include "compute.hh"
void translate_overlap(double min_coord_ref[3], double max_coord_ref[3],
                       double min_coord_scene[3], double max_coord_scene[3])
{
    // translate to avoid overlapping issue
}

int main(int argc, char* argv[])
{
    if (argc < 4) {
        std::cout << "Usage: ./icp-gpu [path_to_ref_cloud] [path_to_transform_cloud] [nb_iter]\n";
        return -1;
    }

    double min_coord_ref[3];
    double max_coord_ref[3];

    double min_coord_scene[3];
    double max_coord_scene[3];

    int max_iter = atoi(argv[3]);

    auto matrix_ref = load_matrix(argv[1], min_coord_ref, max_coord_ref);
    auto matrix_scene = load_matrix(argv[2], min_coord_scene, max_coord_scene);

    GPU::ICP icp(matrix_ref, matrix_scene, max_iter);

    icp.find_corresponding();
    write_matrix(icp.new_p);

    return 0;
}
