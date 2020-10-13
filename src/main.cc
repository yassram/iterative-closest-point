#include <iostream>
#include "cpu.hh"
#include "load.hh"

int main(int argc, char* argv[])
{
    auto matrix = load_matrix("./data_students/cow_ref.txt");
    std::cout << matrix << std::endl;

    return 0;
}
