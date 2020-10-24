#include "cpu.hh"
#include <cmath>
#include <iostream>


namespace CPU
{
    void ICP::find_corresponding() {

        for (int i = 0; i < this->max_iter; i++) {
            std::cerr << "[ICP] iteration number " << i << " | ";

            MatrixXd Y = MatrixXd::Zero(this->dim, this->np);
            for (int j = 0; j < this->np; j++) {

                MatrixXd pi = this->new_p.col(j);
                MatrixXd d = MatrixXd::Zero(1, this->nm);

                for (int k = 0; k < this->nm; k++) {
                    MatrixXd mk = this->m.col(k);
                    auto t1 = pi - mk;
                    auto t2 = t1.array().pow(2).sum();
                    d(k) = sqrt(t2);
                }
                MatrixXd::Index minRow, minCol;
                int m = d.minCoeff(&minRow, &minCol);

                std::cout << "minCol: " << (int)minCol << std::endl;
                Y.col(j) = this->m.col((int)minCol);
            }
            double err = ICP::find_alignment(Y);

            for (int j = 0; j < this->np; j++) {
                this->new_p.col(j) = (this->s * this->r) * this->new_p.col(j) + this->t;

                MatrixXd e = Y.col(j) - this->new_p.col(j);
                err = err + (e.transpose() * e)(0);
            }

            err /= this->np;
            std::cerr << "err = " << err << std::endl;

            if (err < this->threshold)
                break;
        }

    }

    int max_element_index(Eigen::EigenSolver<Eigen::MatrixXd>::EigenvalueType& eigen_value)
    {
        int index = 0;
        double max = real(eigen_value(0));
        for (int i = 1; i < 4; i++) {
            if (real(eigen_value(i)) > max)
                index = i;
        }
        return index;
    }

    double ICP::find_alignment(MatrixXd y)
    {
        auto dim_new_p = this->new_p.rows();
        auto n_new_p = this->new_p.cols();

        auto dim_y = y.rows();
        auto n_y = y.cols();

        if (n_new_p != n_y) {
            std::cerr << "Point sets need to have the same number of points.\n";
            return -1;
        }

        if (dim_new_p != 3 || dim_y != 3) {
            std::cerr << "Need points of dimension 3\n";
            return -1;
        }

        if (n_new_p < 4) {
            std::cerr << "Need at least 4 point pairs\n";
        }

        auto mu_p = this->new_p.rowwise().mean();
        auto mu_y = y.rowwise().mean();

        MatrixXd p_prime = this->new_p.colwise() - mu_p;
        MatrixXd y_prime = y.colwise() - mu_y;

        auto px = p_prime.row(0);
        auto py = p_prime.row(1);
        auto pz = p_prime.row(2);

        auto yx = y_prime.row(0);
        auto yy = y_prime.row(1);
        auto yz = y_prime.row(2);

        auto sxx = (px.array() * yx.array()).sum();
        auto sxy = (px.array() * yy.array()).sum();
        auto sxz = (px.array() * yz.array()).sum();
        auto syx = (py.array() * yx.array()).sum();
        auto syy = (py.array() * yy.array()).sum();
        auto syz = (py.array() * yz.array()).sum();
        auto szx = (pz.array() * yx.array()).sum();
        auto szy = (pz.array() * yy.array()).sum();
        auto szz = (pz.array() * yz.array()).sum();

        MatrixXd n_matrix{4, 4};
        n_matrix << sxx + syy + szz, syz - szy, -1 * sxz + szx, sxy - syx,
            -1 * szy + syz, sxx - szz - syy, sxy + syx, sxz + szx,
            szx - sxz, syx + sxy, syy - szz - sxx, syz + szy,
            -1 * syx + sxy, szx + sxz, szy + syz, szz - syy - sxx;


        Eigen::EigenSolver<MatrixXd> eigen_solver(n_matrix);
        auto eigen_values = eigen_solver.eigenvalues();
        auto eigen_vectors = eigen_solver.eigenvectors();
        auto max_eigen_value_index = max_element_index(eigen_values);

        double q0 = real(eigen_vectors(0, max_eigen_value_index));
        double q1 = real(eigen_vectors(1, max_eigen_value_index));
        double q2 = real(eigen_vectors(2, max_eigen_value_index));
        double q3 = real(eigen_vectors(3, max_eigen_value_index));

        MatrixXd q_bar{4, 4};
        q_bar << q0, -1. * q1, -1. * q2, -1. * q3,
            q1, q0, q3, -1. * q2,
            q2, -1. * q3, q0, q1,
            q3, q2, -1. * q1, q0;

        MatrixXd q_caps{4, 4};
        q_caps << q0, -1. * q1, -1. * q2, -1. * q3,
            q1, q0, -1. * q3, q2,
            q2, q3, q0, -1. * q1,
            q3, -1. * q2, q1, q0;

        MatrixXd temp_r = (q_bar.conjugate().transpose() * q_caps).real();

        this->r = temp_r.block(1, 1, 3, 3);

        auto sp = 0.;
        auto d_caps = 0.;

        for (auto i = 0; i < n_new_p; i++) {
            auto y_prime_view = y_prime.col(i);
            auto p_prime_view = p_prime.col(i);
            d_caps = d_caps + (y_prime_view.transpose() * y_prime_view)(0);
            sp = sp + (p_prime_view.transpose() * p_prime_view)(0);
        }

        this->s = sqrt(d_caps / sp);
        this->t = mu_y - this->s * r * mu_p;

        auto err = 0.;
        for (auto i = 0; i < n_new_p; i++) {
            auto d = y.col(i) - ((this->s * this->r) * this->new_p.col(i) + this->t);
            err += (d.transpose() * d)(0);
        }

        return err;
    }

}
