#pragma once
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"

namespace CPU
{
    class ICP
    {
    public:
        ICP(double s_, xt::xarray<double> t_, xt::xarray<double> r_,
            xt::xarray<double> m_, xt::xarray<double> p_)
            : s{s_},
              t{t_},
              r{r_},
              m{m_},
              p{p_},
              new_p{p_},
              np{(unsigned int)p_.shape(1)},
              nm{(unsigned int)m_.shape(1)},
              dim{(unsigned int)p_.dimension()}
        {
        }

        ~ICP();

        void find_corresponding();
        double find_alignment(xt::xarray<double> y);

        //Getter & Setter
        inline double getS() { return s; }
        inline void setS(double s_) { s = s_; }

        inline xt::xarray<double> getR() { return r; }
        inline void setR(xt::xarray<double> r_) { r = r_; }

        inline xt::xarray<double> getT() { return t; }
        inline void setT(xt::xarray<double> t_) { t = t_; }

        inline xt::xarray<double> getP() { return p; }
        inline void setP(xt::xarray<double> p_) { p = p_; }

        inline xt::xarray<double> getM() { return t; }
        inline void setM(xt::xarray<double> m_) { m = m_; }

        inline xt::xarray<double> getNewP() { return new_p; }
        inline void setNewP(xt::xarray<double> new_p_) { new_p = new_p_; }

        inline int getMaxIter() { return max_iter; }
        inline double getThreshold() { return threshold; }
        inline unsigned int getNP() { return np; }
        inline unsigned int getNM() { return nm; }
        inline unsigned int getDim() { return dim; }

    private:
        double s;
        xt::xarray<double> t;
        xt::xarray<double> r;

        xt::xarray<double> m;
        xt::xarray<double> p;
        xt::xarray<double> new_p;

        unsigned int np;
        unsigned int nm;
        unsigned int dim;

        const int max_iter = 200;
        const double threshold = 1e-5;
    };
} // namespace CPU