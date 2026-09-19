#include <iostream>
#include <Eigen/Dense>
#include "../utils/activations.hpp"
int main()
{
    Eigen::MatrixXd x(9, 1);
    x << 1, 2, 4, 0, 3.0,-2,-8,9,23;

    Eigen::MatrixXd result = relu(x);

    std::cout << result << '\n';
    Eigen::MatrixXd der = relu_derivative(x);
    std::cout<< "\ndervivatives\n";
    std::cout<< der<<"\n";
    return 0;
}