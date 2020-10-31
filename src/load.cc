#include "load.hh"

GPU::Matrix load_matrix(const char *filename)
{
    std::ifstream infile;
    std::string line;
    int n = 0;

    std::cerr << "[load] opening " << filename << std::endl;
    infile.open(filename);
    if (not infile.is_open()) {
        std::cerr << "[load] " << filename << " could not be opened" << std::endl;
        exit(2);
    }
    while (getline(infile, line))
        n++;
    n--;

    infile.clear();
    infile.seekg(0, std::ios::beg);
    std::getline(infile, line);
    MatrixXd matrix = MatrixXd::Zero(n, 3);

    std::cout << "[load] loading file into matrix" << std::endl;
    for (int i = 0; i < n; i++) {
        getline(infile, line);
        double x = 0., y = 0., z = 0.;
        std::sscanf(line.c_str(), "%lf,%lf,%lf", &x, &y, &z);
        matrix.row(i) << x, y, z;
    }
    GPU::Matrix res {matrix.transpose()};
    return res;
}

void write_matrix(GPU::Matrix matrix)
{
    auto matrix_t = matrix.transpose();
    std::ofstream output;
    output.open("output.txt");
    output << "Points_0,Points_1,Points_2" << std::endl;
    for (int i = 0; i < matrix_t.rows(); i++) {
        output << matrix_t.row(i)(0) << ','
               << matrix_t.row(i)(1) << ','
               << matrix_t.row(i)(2) << std::endl;
    }
    std::cout << "[output] output file \"output.txt\" was generated." << std::endl;
    output.close();
}
