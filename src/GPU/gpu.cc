#include "gpu.hh"

namespace GPU
{
    void ICP::find_corresponding() {

        if (this->p.cols() != this->m.cols()) {
            std::cerr << "[error] Point sets need to have the same number of points.\n";
            return exit(-1);
        }

        if (this->p.cols() < 4) {
            std::cerr << "[error] Need at least 4 point pairs\n";
            exit(-1);
        }

        for (int i = 0; i < this->max_iter; i++) {
            std::cerr << "[ICP] iteration number " << i << " | ";

            Matrix Y{Matrix::Zero(this->dim, this->np)};

            compute_Y_w(m,new_p,Y);

            double err = ICP::find_alignment(Y);

            Matrix sr {this->s * this->r};
            err += compute_err_w(Y, this->new_p, true, sr, this->t);

            err /= this->np;
            std::cerr << "error value = " << err << std::endl;

            if (err < this->threshold)
                break;
        }

    }

    int max_element_index(Eigen::EigenSolver<MatrixXd>::EigenvalueType& eigen_value)
    {
        int index = 0;
        double max = real(eigen_value(0));
        for (int i = 1; i < 4; i++)
            if (real(eigen_value(i)) > max)
                index = i;
        return index;
    }

    double ICP::find_alignment(Matrix y)
    {

        Matrix mu_p{this->new_p.rowwise().mean()};
        Matrix mu_y{y.rowwise().mean()};

        Matrix p_prime = substract_col_w(this->new_p, mu_p);
        Matrix y_prime = substract_col_w(y, mu_y);

        MatrixXd s{p_prime * y_prime.transpose()};

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

        Matrix q_bar{MatrixXd{4, 4}};
        q_bar << q0, -1. * q1, -1. * q2, -1. * q3,
            q1, q0, q3, -1. * q2,
            q2, -1. * q3, q0, q1,
            q3, q2, -1. * q1, q0;

        MatrixXd q_caps{MatrixXd{4, 4}};
        q_caps << q0, -1. * q1, -1. * q2, -1. * q3,
            q1, q0, -1. * q3, q2,
            q2, q3, q0, -1. * q1,
            q3, -1. * q2, q1, q0;

        Matrix temp_r = {(q_bar.transpose() * q_caps)};

        this->r = {temp_r.block(1, 1, 3, 3)};

        auto sp = 0.;
        auto d_caps = 0.;

        y_p_norm_w(y_prime, p_prime, this->new_p.cols(), d_caps, sp);

        this->s = sqrt(d_caps / sp);
        Matrix sr {this->s * this->r};
        this->t = {mu_y - sr * mu_p};

        double err = compute_err_w(y, this->new_p, false, sr, this->t);

        return err;
    }

}
