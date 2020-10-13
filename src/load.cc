#include "load.hh"

MatrixXd load_matrix(const char *filename)
{
    std::ifstream infile;
    std::string line;
    int n = 0;

    infile.open(filename);
    if (not infile.is_open())
        std::cout << "not opened" << std::endl;
    /* FILE* infile = fopen(filename); */
    while (getline(infile, line))
        n++;
    n--;
    std::cout << n << std::endl;
    infile.clear();
    infile.seekg(0, std::ios::beg);
    /* fseek(infile, 0, SEEK_SET); */

    std::getline(infile, line);
    MatrixXd matrix = MatrixXd::Zero(n, 3);

    for (int i = 0; i < n; i++) {
        getline(infile, line);
        std::stringstream ss(line);
        std::vector<double> vect;
        for (double j; ss >> j;) {
            vect.push_back(j);
            if (ss.peek() == ',')
                ss.ignore();
        }
        /* double a, b, c; */
        /* std::sscanf(line.c_str(), "%lf,%lf,%lf", &a, &b, &c); */
        /* matrix.row(i) << a, b, c; */
        matrix.row(i) << vect[0], vect[1], vect[2];
    }

    std::cout << matrix << std::endl;
    return matrix;
}
