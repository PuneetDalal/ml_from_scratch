#pragma once
#include <Eigen/Dense>
double mse(const Eigen::MatrixXd& predictions , const Eigen::MatrixXd&true_val); 
Eigen::MatrixXd mse_derivative(const Eigen::MatrixXd& prediction, const Eigen::MatrixXd& true_value);