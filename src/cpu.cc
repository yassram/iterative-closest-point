#include "cpu.hh"
#include <cmath>
#include <iostream>


namespace CPU
{
    void ICP::find_corresponding() {
        MatrixXd new_p = ICP::getNewP();
        MatrixXd M = ICP::getM();

        for (int i = 1; i < ICP::getMaxIter(); i++) {
            std::cerr << "[ICP] iteration number " << i << std::endl;
            MatrixXd Y = MatrixXd::Zero(ICP::getDim(), ICP::getNP());
            for (int j = 0; j < ICP::getNP(); j++) {

                // MatrixXd pi = xt::col(new_p, j);
                MatrixXd pi = new_p.col(j);
                MatrixXd d = MatrixXd::Zero(1, ICP::getNM());                

                for (int k = 0; k < ICP::getNM(); k++) {
                    MatrixXd mk = M.col(k);
                    auto t1 = pi - mk;
                    auto t2 = t1.array().pow(2).sum();
                    d(k) = sqrt(t2);
                    //d(k) = sqrt(xt::sum(xt::pow((pi - mk), 2)));
                }
                
                // int m = xt::argmin(d);
                MatrixXd::Index minRow, minCol;
                int m = d.minCoeff(&minRow, &minCol);
            
                // Y(j) = M.col((double)(minCol));
                Y.col(j) = M.col((double)minCol);
                //xt::col(Y, j) = xt::col(M,m);
            }
            
            double err = ICP::find_alignment(Y);
            //ICP::setY(Y);

            double s = ICP::getS();
            MatrixXd t = ICP::getT();
            MatrixXd r = ICP::getR();
            for (int j = 0; j < ICP::getNP(); j++) {
                new_p.col(j) = s * r * new_p.col(j) + t;
                MatrixXd e = Y.col(j) - new_p.col(j);
                err = err + (e.transpose() * e)(0);
            }
            
            ICP::setNewP(new_p);
            err /= ICP::getNP();
            
            if (err < ICP::getThreshold())
                break;
        }

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

        // auto dv = xt::linalg::eig(n_matrix);
        // auto v = std::get<1>(dv);
        Eigen::EigenSolver<MatrixXd> dv{n_matrix};
        auto v = dv.eigenvectors();

        // auto q = xt::view(v, xt::all(), 4);
        auto q = v.col(3);
        auto q0 = q(0).real();
        auto q1 = q(1).real();
        auto q2 = q(2).real();
        auto q3 = q(3).real();
        
        MatrixXd q_bar{4, 4};
        q_bar << q0, -1 * q1, -1 * q2, -1 * q3,
                 q1, q0, q3, -1 * q2,
                 q2, -1 * q3, q0, q1,
                 q3, q2, -1 * q1, q0;
        
        MatrixXd q_caps{4, 4};
        q_caps << q0, -1 * q1, -1 * q2, -1 * q3,
                  q1, q0, -1 * q3, q2,
                  q2, q3, q0, -1 * q1,
                  q3, -1 * q2, q1, q0;

        // xt::transpose(q_bar);
        q_bar.transposeInPlace();
        auto temp_r = q_bar * q_caps;
        // this->r = xt::view(temp_r, xt::range(2, 4), xt::range(2, 4));
        this->r = temp_r.block(1, 1, 3, 3);

        auto sp = 0.;
        auto d_caps = 0.;

        for (auto i = 0; i < n_new_p; i++) {
            // auto y_prime_view = xt::view(y_prime, xt::all(), i);
            // auto p_prime_view = xt::view(p_prime, xt::all(), i);
            auto y_prime_view = y_prime.col(i);
            auto p_prime_view = p_prime.col(i);
            d_caps = d_caps + (y_prime_view.transpose() * y_prime_view)(0);
            sp = sp + (p_prime_view.transpose() * p_prime_view)(0);
        }

        this->s = sqrt(d_caps / sp);
        this->t = mu_y - s * r * mu_p;

        auto err = 0.;
        for (auto i = 0; i < n_new_p; i++) {
            // auto d = (xt::view(y, xt::all(), i) - (this->s * this->r * xt::view(this->new_p, xt::all(), i) + t));
            auto d = y.col(i) - (this->s * this->r * this->new_p.col(i) + t);
            err += (d.transpose() * d)(0);
        }

        return err;
    }

}
