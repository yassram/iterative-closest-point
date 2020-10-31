#include <iostream>
#include "header.hh"
#include "../load.hh"
#include <complex>
#include "gpu.hh"
#include "compute.hh"
GPU::Matrix translate_overlap(GPU::Matrix& min_coord_ref,
                              GPU::Matrix& max_coord_ref,
                              GPU::Matrix& min_coord_scene,
                              GPU::Matrix& max_coord_scene)
{
    GPU::Matrix translate{MatrixXd{3, 1}};

    translate << 2, 2, 2;

    return translate;
}

int main(int argc, char* argv[])
{
    if (argc < 4) {
        std::cout << "Usage: ./icp-gpu [path_to_ref_cloud] [path_to_transform_cloud] [nb_iter]\n";
        return -1;
    }

    GPU::Matrix min_coord_ref{MatrixXd::Zero(3,1)};
    GPU::Matrix max_coord_ref{MatrixXd::Zero(3,1)};

    GPU::Matrix min_coord_scene{MatrixXd::Zero(3,1)};
    GPU::Matrix max_coord_scene{MatrixXd::Zero(3,1)};

    int max_iter = atoi(argv[3]);

    auto matrix_ref = load_matrix(argv[1], min_coord_ref, max_coord_ref);
    auto matrix_scene = load_matrix(argv[2], min_coord_scene, max_coord_scene);

    GPU::Matrix t = translate_overlap(min_coord_ref, max_coord_ref,
                                      min_coord_scene, max_coord_scene);
    matrix_scene = matrix_scene + t;
    GPU::ICP icp(matrix_ref, matrix_scene, max_iter);

    icp.find_corresponding();
    write_matrix(icp.new_p);

    return 0;
}
