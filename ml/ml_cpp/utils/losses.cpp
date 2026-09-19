#include "losses.hpp"
double bce(const Eigen :: MatrixXd& predction, const Eigen ::MatrixXd& true_value);
double mse(const Eigen::MatrixXd& predictions , const Eigen::MatrixXd&true_val);
Eigen::MatrixXd mse_derivative(const Eigen::MatrixXd& prediction, const Eigen::MatrixXd& true_value);

double mse(const Eigen::MatrixXd& predictions , const Eigen::MatrixXd&true_val){
    return (predictions-true_val).array().square().mean();
}
Eigen::MatrixXd mse_derivative(const Eigen::MatrixXd& prediction, const Eigen::MatrixXd& true_value){
    return 2*(prediction-true_value)/true_value.rows();
}

double bce(const Eigen :: MatrixXd& predction, const Eigen ::MatrixXd& true_value){
    return -1*(true_value.array()*(Eigen::log(predction.array())) + (1-true_value.array())*(Eigen::log(1-predction.array()))).mean();
}