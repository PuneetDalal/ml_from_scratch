#include <iostream>
#include <Eigen/Dense>
#include "../utils/activations.hpp"
int main()
{
    Eigen::MatrixXd x(5, 1);
    x << 1, 2, 4, 0, 3.0;

    Eigen::MatrixXd result = sigmoid(x);

    std::cout << result << '\n';
    Eigen::MatrixXd der = sigmoid_derivative(x);
    std::cout<< "dervivatives\n";
    std::cout<< der<<"\n";
    return 0;
}