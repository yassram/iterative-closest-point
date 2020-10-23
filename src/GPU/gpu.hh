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
    class ICP
    {
    public:
        ICP(MatrixXd m_, MatrixXd p_, int max_iter_)
            : m{m_},
              p{p_},
              new_p{p_},
              np{(unsigned int)p_.cols()},
              nm{(unsigned int)m_.cols()},
              dim{3},
              max_iter{max_iter_}
        {
            this->s = 1.;
            this->r = MatrixXd::Identity(m_.rows(), m_.rows());
            this->t = MatrixXd::Zero(m_.rows(), 1);
        }

        ~ICP()
        {}

        void find_corresponding();
        double find_alignment(MatrixXd y);

    public:
        MatrixXd new_p;

    private:
        double s;
        MatrixXd t;
        MatrixXd r;

        MatrixXd m;
        MatrixXd p;

        unsigned int np;
        unsigned int nm;
        unsigned int dim;

        int max_iter;
        const double threshold = 1e-5;
    };
} // namespace CPU
