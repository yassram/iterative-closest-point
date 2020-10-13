#include "load.hh"

MatrixXd load_matrix(const char *filename)
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
        std::stringstream ss(line);
        std::vector<double> vect;
        for (double j; ss >> j;) {
            vect.push_back(j);
            if (ss.peek() == ',')
                ss.ignore();
        }
        matrix.row(i) << vect[0], vect[1], vect[2];
    }

    return matrix.transpose();
}
