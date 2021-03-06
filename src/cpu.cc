#include "cpu.hh"

namespace CPU
{
    MatrixXd closest_matrix(struct closest_matrix_params pa)
    {
        MatrixXd res = MatrixXd::Zero(pa.dim, pa.np);

        for (int j = 0; j < pa.np; j++)
        {
            MatrixXd pi = pa.p.col(j);
            MatrixXd d = MatrixXd::Zero(1, pa.nm);

            for (int k = 0; k < pa.nm; k++)
            {
                MatrixXd mk = pa.m.col(k);
                auto t1 = pi - mk;
                auto t2 = t1.array().pow(2).sum();
                d(k) = sqrt(t2);
            }
            MatrixXd::Index minRow, minCol;
            int m = d.minCoeff(&minRow, &minCol);

            res.col(j) = pa.m.col((int)minCol);
        }
        return res;
    }

    double err_compute(struct err_compute_params pa)
    {
        double err{};
        auto sr = pa.s * pa.r;
        for (int j = 0; j < pa.np; j++)
        {
            pa.p.col(j) = sr * pa.p.col(j) + pa.t;
            MatrixXd e = pa.Y.col(j) - pa.p.col(j);
            err = err + (e.transpose() * e)(0);
        }
        return err;
    }

    void ICP::alignement_check()
    {
        if (this->p.cols() != this->m.cols()) {
            std::cerr << "[error] Point sets need to have the same number of points.\n";
            return exit(-1);
        }

        if (this->p.cols() < 4) {
            std::cerr << "[error] Need at least 4 point pairs\n";
            exit(-1);
        }
    }

    void ICP::find_corresponding()
    {
        alignement_check();

        for (int i = 0; i < this->max_iter; i++)
        {
            std::cerr << "[ICP] iteration number " << i << " | ";

            MatrixXd Y = closest_matrix(get_closest_matrix_params());

            double err = ICP::find_alignment(Y);

            struct err_compute_params ecp
                {
                    this->np, this->new_p, this->s, this->r, this->t, Y
                };
            err += err_compute(ecp);

            err /= this->np;
            std::cerr << "error value = " << err << std::endl;

            if (err < this->threshold)
                break;
        }
    }

    int max_element_index(Eigen::EigenSolver<Eigen::MatrixXd>::EigenvalueType &eigen_value)
    {
        int index = 0;
        double max = real(eigen_value(0));
        for (int i = 1; i < 4; i++)
        {
            if (real(eigen_value(i)) > max)
                index = i;
        }
        return index;
    }

    double err_compute_alignment(struct err_compute_alignment_params pa)
    {
        double err{};
        auto sr = pa.s * pa.r;
        for (int j = 0; j < pa.np; j++)
        {
            auto d = pa.y.col(j) - (sr * pa.p.col(j) + pa.t);
            err += (d.transpose() * d)(0);
        }
        return err;
    }

    double ICP::find_alignment(MatrixXd y)
    {
        auto dim_new_p = this->new_p.rows();
        auto n_new_p = this->new_p.cols();

        auto dim_y = y.rows();
        auto n_y = y.cols();

        auto mu_p = this->new_p.rowwise().mean();
        auto mu_y = y.rowwise().mean();

        MatrixXd p_prime = this->new_p.colwise() - mu_p;
        MatrixXd y_prime = y.colwise() - mu_y;

        MatrixXd s{p_prime * y_prime.transpose()}; //multiplication matricielle

        MatrixXd n_matrix{4, 4};

        n_matrix << s(0,0) + s(1,1) + s(2,2), s(1,2) - s(2,1), -1 * s(0,2) + s(2,0), s(0,1) - s(1,0),
            -1 * s(2,1) + s(1,2), s(0,0) - s(2,2) - s(1,1), s(0,1) + s(1,0), s(0,2) + s(2,0),
            s(2,0) - s(0,2), s(1,0) + s(0,1), s(1,1) - s(2,2) - s(0,0), s(1,2) + s(2,1),
            -1 * s(1,0) + s(0,1), s(2,0) + s(0,2), s(2,1) + s(1,2), s(2,2) - s(1,1) - s(0,0);

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

        for (auto i = 0; i < n_new_p; i++)
        {
            auto y_prime_view = y_prime.col(i);
            auto p_prime_view = p_prime.col(i);
            d_caps = d_caps + (y_prime_view.transpose() * y_prime_view)(0);
            sp = sp + (p_prime_view.transpose() * p_prime_view)(0);
        }

        this->s = sqrt(d_caps / sp);
        MatrixXd sr{this->r * this->s};
        this->t = mu_y - sr * mu_p;

        struct err_compute_alignment_params ecap
            {
                this->np, this->new_p, this->s, this->r, this->t, y
            };

        return err_compute_alignment(ecap);
    }
} // namespace CPU
