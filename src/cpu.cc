#include "cpu.hh"
#include <cmath>
#include <iostream>


namespace CPU
{
    void ICP::find_corresponding() {
        xt::xarray<double> new_p = ICP::getNewP();
        xt::xarray<double> M = ICP::getM();

        for (int i = 1; i < ICP::getMaxIter(); i++) {
            xt::xarray<double> Y = xt::zeros<int>({ICP::getDim(), ICP::getNP()});
            for (int j = 1; j <= ICP::getNP(); j++) {

                xt::xarray<double> pi = xt::col(new_p, j);
                xt::xarray<double> d = xt::zeros<int>({1, ICP::getNM()});
                

                for (int k = 1; 1 < ICP::getNM(); k++) {
                    xt::xarray<double> mk = xt::col(M, k);
                    d(k) = sqrt(xt::sum(xt::pow((pi - mk), 2)));
                }
                
                int m = xt::argmin(d);
                //double mind = d(m);

                xt::col(Y, j) = xt::col(M,m);
            }
            
            double err = ICP::find_alignment(Y);
            //ICP::setY(Y);

            double s = ICP::getS();
            xt::xarray<double> t = ICP::getT();
            xt::xarray<double> r = ICP::getR();
            for (int j = 1; j < ICP::getNP(); j++) {
                xt::col(new_p, j) = s * r * xt::col(new_p, j) + t;
                xt::xarray<double> e = xt::col(Y, j) - xt::col(new_p, j);
                err += xt::transpose(e) * e;
            }
            
            ICP::setNewP(new_p);
            err /= ICP::getNP();
            
            if (err < ICP::getThreshold())
                break;
        }

    }
    
    double ICP::find_alignment(xt::xarray<double> y)
    {
        dim_new_p = this.new_p.dimension();
        n_new_p = this.new_p.shape(1);

        dim_y = y.dimension();
        n_y = y.shape(1);

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

        mu_p = xt::mean(new_p, 1);
        mu_y = xt::mean(y, 1);
        
        p_prime = this.new_p - mu_p;
        y_prime = y - mu_y;

        px = xt::view(p_prime, 1, xt::xall());
        py = xt::view(p_prime, 2, xt::xall());
        pz = xt::view(p_prime, 3, xt::xall());
        yx = xt::view(y_prime, 1, xt::xall());
        yy = xt::view(y_prime, 2, xt::xall());
        yz = xt::view(y_prime, 3, xt::xall());

        sxx = xt::sum(px * yx);
        sxy = xt::sum(px * yy);
        sxz = xt::sum(px * yz);
        syx = xt::sum(py * yx);
        syy = xt::sum(py * yy);
        syz = xt::sum(py * yz);
        szx = xt::sum(pz * yx);
        szy = xt::sum(pz * yy);
        szz = xt::sum(pz * yz);

        xt::xarray<double> n_matrix = {{sxx + syy + szz, syz - szy, -sxz + szx, sxy - syx},
                                       {-szy + syz, sxx - szz - syy, sxy + syx, sxz + szx},
                                       {szx - sxz, syx + sxy, syy - szz - sxx, syz + szy},
                                       {-syx + sxy, szx + sxz, szy + syz, szz - syy - sxx}};
        
        auto dv = xt::linalg::eig(n_matrix);
        auto v = std::get<1>(dv);

        auto q = xt::view(v, xt::xall(), 4);
        auto q0 = q(0);
        auto q1 = q(1);
        auto q2 = q(2);
        auto q3 = q(3);

        xt::xarray<double> q_bar = {{q0, -q1, -q2, -q3},
                                    {q1, q0, q3, -q2},
                                    {q2, -q3, q0, q1},
                                    {q3, q2, -q1, q0}};
        
        xt::xarray<double> q_caps = {{q0, -q1, -q2, -q3},
                                     {q1, q0, -q3, q2},
                                     {q2, q3, q0, -q1},
                                     {q3, -q2, q1, q0}};
        
        auto temp_r = xt::transpose(q_bar) * q_caps;
        r = xt::view(temp_r, xt::range(2, 4), xt::range(2, 4));

        auto sp = 0.;
        auto d = 0.;

        for (auto i = 0; i < n_new_p; i++) {
            d += xt::transpose(xt::view(y_prime, xt::xall(), i)) * xt::view(y_prime, xt::xall(), i);
            sp += xt::transpose(xt::view(p_prime, xt::xall(), i)) * xt::view(p_prime, xt::xall(), i);
        }

        this.s = xt::sqrt(d / sp);
        this.t = 

        return 0.;
    }
}