#include "activations.hpp"

Eigen::MatrixXd sigmoid_derivative( const Eigen::MatrixXd& x);
Eigen::MatrixXd sigmoid(const Eigen ::MatrixXd& x);


Eigen::MatrixXd sigmoid(const Eigen ::MatrixXd& x){
    Eigen::MatrixXd sig = 1.0 / (1.0 + (-x).array().exp());
    return sig;
}

Eigen::MatrixXd sigmoid_derivative(const Eigen ::MatrixXd& x){
    Eigen::MatrixXd s = sigmoid(x);
    return (s.array() * (1.0 - s.array())).matrix();
}
