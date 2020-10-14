#include <iostream>
#include "cpu.hh"
#include "load.hh"

int main(int argc, char* argv[])
{
    auto matrix_ref = load_matrix("./data_students/cow_ref.txt");
    auto matrix_tr1 = load_matrix("./data_students/cow_tr1.txt");
    CPU::ICP icp(matrix_ref, matrix_tr1);
    icp.find_corresponding();
    /* std::cout << "S: "; */
    /* std::cout << icp.getS() << std::endl; */
    /* std::cout << "R: "; */
    /* std::cout << icp.getR() << std::endl; */
    /* std::cout << "T: "; */
    /* std::cout << icp.getT() << std::endl; */

    /* std::cout << "New P: " << icp.getNewP().transpose() << std::endl; */
    /* std::cout << "end" << std::endl; */
    write_matrix(icp.getNewP());
    return 0;
}
