#pragma once
#include <cmath>
#include <iostream>
#include <exception>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

using Eigen::ArrayXd;
using Eigen::MatrixXcd;
using Eigen::MatrixXd;
using Eigen::Vector3d;

namespace CPU
{
    struct closest_matrix_params
    {
        unsigned int dim;
        unsigned int np;
        unsigned int nm;
        const MatrixXd &p;
        const MatrixXd &m;
    };

    struct err_compute_params
    {
        unsigned int np;
        MatrixXd &p;
        double s;
        const MatrixXd &r;
        const MatrixXd &t;
        const MatrixXd &Y;
    };

    struct err_compute_alignment_params
    {
        unsigned int np;
        const MatrixXd &p;
        double s;
        const MatrixXd &r;
        const MatrixXd &t;
        const MatrixXd &y;
    };

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
        {
        }

        struct closest_matrix_params get_closest_matrix_params()
        {
            struct closest_matrix_params cmp
                {
                    dim, np, nm, new_p, m
                };
            return cmp;
        }

        struct err_compute_params get_err_compute_params(Eigen::MatrixXd &Y)
        {
            struct err_compute_params ecp
                {
                    np, new_p, s, r, t, Y
                };
            return ecp;
        }

        struct err_compute_alignment_params get_err_compute_alignment_params(Eigen::MatrixXd &Y)
        {
            struct err_compute_alignment_params ecap
                {
                    np, new_p, s, r, t, Y
                };
            return ecap;
        }

        void find_corresponding();
        void alignement_check();
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

    MatrixXd closest_matrix(struct closest_matrix_params pa);
    double err_compute(struct err_compute_params pa);
    double err_compute_alignment(struct err_compute_alignment_params pa);
    int max_element_index(Eigen::EigenSolver<Eigen::MatrixXd>::EigenvalueType &eigen_value);

} // namespace CPU
