#include <iostream>
#include "cpu.hh"
#include "load.hh"
// #include "xtensor/xarray.hpp"
// #include "xtensor/xio.hpp"
// #include "xtensor/xview.hpp"
// #include "xtensor-blas/xlinalg.hpp"

int main(int argc, char* argv[])
{
    // xt::xarray<double> arr1
    //   {{1.0, 2.0, 3.0},
    //    {2.0, 5.0, 7.0},
    //    {2.0, 5.0, 7.0}};

    // xt::xarray<double> arr2
    //   {5.0, 6.0, 7.0};

    // xt::xarray<double> res = xt::view(arr1, 1) + arr2;

    // std::cout << res << std::endl;

    // xt::xarray<double> a = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    // auto d = xt::linalg::det(a);
    // std::cout << d << std::endl;  // 6.661338e-16

    std::cout << "hello\n";
    load_matrix("./data_students/cow_ref.txt");

    return 0;
}
