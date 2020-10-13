#pragma once
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

using Eigen::MatrixXd;
using Eigen::ArrayXd;
using Eigen::Vector3d;

namespace CPU
{
    class ICP
    {
    public:
        ICP(double s_, MatrixXd t_, MatrixXd r_,
            MatrixXd m_, MatrixXd p_)
            : s{s_},
              t{t_},
              r{r_},
              m{m_},
              p{p_},
              new_p{p_},
              np{(unsigned int)p_.cols()},
              nm{(unsigned int)m_.cols()},
              dim{(unsigned int)p_.rows()}
        {
        }

        ~ICP();

        void find_corresponding();
        double find_alignment(MatrixXd y);

        //Getter & Setter
        inline double getS() { return s; }
        inline void setS(double s_) { s = s_; }

        inline MatrixXd getR() { return r; }
        inline void setR(MatrixXd r_) { r = r_; }

        inline MatrixXd getT() { return t; }
        inline void setT(MatrixXd t_) { t = t_; }

        inline MatrixXd getP() { return p; }
        inline void setP(MatrixXd p_) { p = p_; }

        inline MatrixXd getM() { return t; }
        inline void setM(MatrixXd m_) { m = m_; }

        inline MatrixXd getNewP() { return new_p; }
        inline void setNewP(MatrixXd new_p_) { new_p = new_p_; }

        inline int getMaxIter() { return max_iter; }
        inline double getThreshold() { return threshold; }
        inline unsigned int getNP() { return np; }
        inline unsigned int getNM() { return nm; }
        inline unsigned int getDim() { return dim; }

    private:
        double s;
        MatrixXd t;
        MatrixXd r;

        MatrixXd m;
        MatrixXd p;
        MatrixXd new_p;

        unsigned int np;
        unsigned int nm;
        unsigned int dim;

        const int max_iter = 200;
        const double threshold = 1e-5;
    };
} // namespace CPU
