#include "load.hh"

void update_coord(double x, double y, double z, GPU::Matrix& min_coord,
                  GPU::Matrix& max_coord)
{
    if (x > max_coord(0,0))
        max_coord(0,0) = x;
    if (x < min_coord(0,0))
        min_coord(0,0) = x;
    if (y > max_coord(1,0))
        max_coord(1,0) = y;
    if (y < min_coord(1,0))
        min_coord(1,0) = y;
    if (z > max_coord(2,0))
        max_coord(2,0) = z;
    if (z < min_coord(2,0))
        min_coord(2,0) = z;
}


GPU::Matrix load_matrix(const char *filename, GPU::Matrix& min_coord,
                        GPU::Matrix& max_coord)
{
    std::ifstream infile;
    std::string line;
    int n = 0;

    std::cerr << "[load] opening " << filename << std::endl;
    infile.open(filename);
    if (not infile.is_open())
        std::cerr << "[load] " << filename << "could not be opened" << std::endl;
    while (getline(infile, line))
        n++;
    n--;

    infile.clear();
    infile.seekg(0, std::ios::beg);
    std::getline(infile, line);
    MatrixXd matrix = MatrixXd::Zero(n, 3);

    std::cerr << "[load] loading file into matrix" << std::endl;
    for (int i = 0; i < n; i++) {
        getline(infile, line);
        double x = 0., y = 0., z = 0.;
        std::sscanf(line.c_str(), "%lf,%lf,%lf", &x, &y, &z);
        update_coord(x, y, z, min_coord, max_coord);
        matrix.row(i) << x, y, z;
    }
    GPU::Matrix res {matrix.transpose()};
    return res;
}

//matrix 3,n du coup
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
    output.close();
}
