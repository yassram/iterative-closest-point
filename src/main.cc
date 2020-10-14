#include <iostream>
#include "cpu.hh"
#include "load.hh"
#include <complex>


int main(int argc, char* argv[])
{
    auto matrix_ref = load_matrix("./data_students/test.txt");
    auto matrix_tr1 = load_matrix("./data_students/test_ref.txt");
    CPU::ICP icp(matrix_ref, matrix_tr1);
    icp.find_corresponding();
    write_matrix(icp.getNewP());
    return 0;
}
