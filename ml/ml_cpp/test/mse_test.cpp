#include <iostream>
#include <Eigen/Dense>
#include "../utils/losses.hpp"
int main()
{
    Eigen::MatrixXd x(9, 1);
    x << 1, 2, 4, 0, 3.0,-2,-8,9,23;

    Eigen::MatrixXd y(9, 1);
    x << 1, 2,34,33,-5,-4,89,10,2;

    std::cout<<"mse : \n";
    std::cout << mse(x,y) << '\n';
    
    Eigen::MatrixXd der = mse_derivative(x,y);
    std::cout<< "\nmse dervivatives\n";
    std::cout<< der<<"\n";
    return 0;
}