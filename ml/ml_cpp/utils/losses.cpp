#include "losses.hpp"
double mse(const Eigen::MatrixXd& predictions , const Eigen::MatrixXd&true_val);
double mse(const Eigen::MatrixXd& predictions , const Eigen::MatrixXd&true_val){
    return (predictions-true_val).array().square().mean();
}
Eigen::MatrixXd mse_derivative(const Eigen::MatrixXd& prediction, const Eigen::MatrixXd& true_value);

Eigen::MatrixXd mse_derivative(const Eigen::MatrixXd& prediction, const Eigen::MatrixXd& true_value){
    return 2*(prediction-true_value)/true_value.rows();
}