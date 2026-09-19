#pragma once

#include <Eigen/Dense>

Eigen::MatrixXd sigmoid(const Eigen::MatrixXd& x);

Eigen::MatrixXd sigmoid_derivative(const Eigen::MatrixXd& x);