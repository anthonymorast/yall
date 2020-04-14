#include <YALL/Models/MultipleRegression.hpp>

namespace yall
{
    void MultipleRegression::train(double** x, double* y, int n, int cols)
    {
        arma::mat alpha = arma::ones<arma::mat>(n, 1);
        double* flat_x = new double[n*cols];
        int fx_idx = 0;
        for(int i = 0; i < cols; i++)       // need to flip the rows/cols due to how the matrices are created
        {
            for(int j = 0; j < n; j++)
            {
                flat_x[fx_idx] = x[j][i];
                fx_idx++;
            }
        }
        arma::mat xmat(flat_x, n, cols);
        arma::mat ymat(y, n, 1);
        xmat.insert_cols(0, alpha);      // add alpha i.e. all ones

        // beta = (X.T*X)^(-1)*X.T*Y
        arma::mat beta = (xmat.t()*xmat).i();
        beta *= xmat.t();
        beta *= ymat;       // col vector containing coefficients + intercept

        _betas = new double[cols];
        _variables = new std::string[cols];
        _alpha = beta(0);
        for(int i = 0; i < cols; i++)
        {
            _betas[i] = beta(i+1);
            _variables[i] = "x" + std::to_string(i);
        }

        _beta_size = cols;
        _training_complete = true;
        delete[] flat_x;
    }

    void MultipleRegression::train(DataTable t)
    {
        train(t.get_all_explanatory(), t.get_response(), t.nrows(), t.ncols()-1);
        delete[] _variables;
        _variables = t.get_explanatory_headers();
    }

    double* MultipleRegression::predict(double** x, int rows)
    {
        if(!_training_complete)
            std::cout << "WARNING (MultipleRegression): getting predictions from untrained model." << std::endl;

        double* y_hat = new double[rows];
        for(int i = 0; i < rows; i++)
        {
            y_hat[i] = _alpha;
            for(int j = 0; j < _beta_size; j++)
                y_hat[i] += (_betas[j]*x[i][j]);
        }
        return y_hat;
    }

    double* MultipleRegression::predict(DataTable t)
    {
        // -1 for response
        if(t.ncols()-1 != _beta_size)
        {
            std::cout << "ERROR (MulipleRegression): data table columns do not match number "
                << "of explanatory variables used for training." 
                << std::endl;
            return 0;
        }
        return predict(t.get_all_explanatory(), t.nrows());
    }

    double MultipleRegression::get_alpha() { return _alpha; }
    const double* MultipleRegression::get_betas() { return (const double*)_betas; }

    void MultipleRegression::print_equation(std::ostream& stream)
    {
        if(!_training_complete)
        {
            stream << "ERROR print_equation(stream): the model has not yet been trained" << std::endl;
            return;
        }

        stream << "y = ";
        for(int i = 0; i < _beta_size; i++)
        {
            stream << "(" << _betas[i] <<  "*" << _variables[i] << ")" << " + ";
        } 
        stream << _alpha << std::endl;
    }
}
