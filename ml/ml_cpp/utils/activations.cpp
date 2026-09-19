#include "activations.hpp"
#include <algorithm>

Eigen::MatrixXd sigmoid_derivative( const Eigen::MatrixXd& x);
Eigen::MatrixXd sigmoid(const Eigen ::MatrixXd& x);

Eigen::MatrixXd relu(const Eigen::MatrixXd& x);
Eigen::MatrixXd relu_derivative(const Eigen::MatrixXd& x);

Eigen::MatrixXd sigmoid(const Eigen ::MatrixXd& x){
    Eigen::MatrixXd sig = 1.0 / (1.0 + (-x).array().exp());
    return sig;
}

Eigen::MatrixXd sigmoid_derivative(const Eigen ::MatrixXd& x){
    Eigen::MatrixXd s = sigmoid(x);
    return (s.array() * (1.0 - s.array())).matrix();
}
Eigen::MatrixXd relu(const Eigen::MatrixXd& x){
    Eigen::MatrixXd rel= x.cwiseMax(0.0);
    return rel;
}

Eigen::MatrixXd relu_derivative(const Eigen::MatrixXd& x){
    Eigen::MatrixXd der = (x.array() > 0).cast<double>();
    return der;
}