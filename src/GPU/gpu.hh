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

namespace GPU
{
    class Matrix : public MatrixXd
    {
    public:
        void fromGpu(double *gpu_rep, unsigned row, unsigned col, size_t pitch);
        double *toGpu(size_t *pitch) const;
        int toGpu(double** p, size_t *pitch, size_t offset, size_t batch_size,
                  bool iscol) const;
    };

    struct gpu_closest_matrix_params
    {
        const Matrix &p;
        const Matrix &m;
        Matrix &y;
    };

    struct gpu_err_compute_params
    {
        const Matrix &Y;
        Matrix &p;
        bool s;
        const Matrix sr;
        const Matrix &t;
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

        struct gpu_closest_matrix_params get_closest_matrix_params(Matrix &Y)
        {
            struct gpu_closest_matrix_params cmp
                {
                    new_p, m, Y
                };
            return cmp;
        }
        struct gpu_err_compute_params get_err_compute_params(Matrix &Y, bool s)
        {
            struct gpu_err_compute_params cmp
                {
                    Y, new_p, s, s*r, t
                };
            return cmp;
        }

        double getDim() {return dim;}
        double getNp() {return np;}

        ~ICP()
        {
        }

        void find_corresponding_naive();
        void find_corresponding_opti();
        Matrix compute_y_naive();
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


// Compute wrapper functions

void compute_Y_w_opti(const GPU::Matrix &m, const GPU::Matrix &pi, GPU::Matrix &Y);
void compute_distance_w_naive(const GPU::Matrix &m, const GPU::Matrix &pi);
double compute_err_w(const GPU::Matrix &Y, GPU::Matrix &p, bool in_place,
                     const GPU::Matrix &sr, const GPU::Matrix &t);

GPU::Matrix substract_col_w(const GPU::Matrix &M, const GPU::Matrix &m);
void y_p_norm_w(const GPU::Matrix &y, const GPU::Matrix &p, size_t size_arr, double &d_caps, double &sp);
