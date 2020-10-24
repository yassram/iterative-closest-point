#pragma once
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include "compute.hh"

using Eigen::MatrixXd;
using Eigen::ArrayXd;
using Eigen::Vector3d;
using Eigen::MatrixXcd;


namespace GPU
{
    class Matrix: public MatrixXd
    {
        public:
             void fromGpu(double *gpu_rep, unsigned row, unsigned col);
             double *toGpu();
    };

    class ICP
    {
    public:
        ICP(Matrix m_, Matrix p_, int max_iter_)
            : m{m_},
              p{p_},
              new_p{p_},
              np{(unsigned int)p_.cols()},
              nm{(unsigned int)m_.cols()},
              dim{3},
              max_iter{max_iter_}
        {
            this->s = 1.;
            this->r = {Matrix::Identity(m_.rows(), m_.rows())};
            this->t = {Matrix::Zero(m_.rows(), 1)};
        }

        ~ICP()
        {}

        void find_corresponding();
        double find_alignment(Matrix y);

    public:
        Matrix new_p;

    private:
        double s;
        Matrix t;
        Matrix r;

        Matrix m;
        Matrix p;

        unsigned int np;
        unsigned int nm;
        unsigned int dim;

        int max_iter;
        const double threshold = 1e-5;
    };
} // namespace GPU
