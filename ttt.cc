#include <iostream>
#include <string>
#include <istream>
#include <vector>
#include <typeinfo>
#include <cmath>
#include <Eigen/Dense>
#include <lazycsv.hpp>
#include <fstream>

std::vector<std::vector<double>> readRCSVFromCSV(std::string& filename) {
    std::vector<std::vector<double>> rcsv{};
    {
        lazycsv::parser parser{ filename };
        for (const auto row : parser)
        {
            std::vector<double> r{};
            for (const auto cell : row)
            {
                r.push_back(std::stod(std::string(cell.raw())));
            }
            rcsv.push_back(r);
        }
    }
    return rcsv;
}

Eigen::Matrix<double, -1, -1, Eigen::RowMajor> convertRCSVToEigen(std::vector<std::vector<double>> rcsv) {
    // Создаем матрицы с использованием Eigen
    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> A(rcsv.size(), rcsv.begin()->size());

    int ir = 0;
    for(auto r : rcsv)
    {
        int ic = 0;
        for(double e: r) {
            A(ir, ic++) = e;
        }
        ir++;
    }
    return A;
}

void writeEigenVectorToCSV(Eigen::VectorXd& x, std::string& filename) {
    std::ofstream ofs(filename);
    for (int i = 0; i < x.size(); i++) {
        ofs << x(i) << std::endl;
    }
}


Eigen::VectorXd gauss(Eigen::Matrix<double, -1, -1, Eigen::RowMajor> A) { 
    int n = A.rows();
    if (A.cols() != n + 1) throw std::runtime_error("Матрица A без последнего столбца должна быть квадратной и этот столбец должен быть столбцом свободных коэффициентов");
    // Приводим к виду ступенчатой матрицы
    for (int i = 0; i < n; i++) {
        // Находим для данного столбца строку с наибольшим значением в этом столбце и поднимаем ее наверх
        int maxRow = i;

        for (int j = i + 1; j < n; j++) {
            if (std::abs(A(j, i)) > std::abs(A(maxRow, i))) {
                maxRow = j;
            }
        }

        A.row(i).swap(A.row(maxRow));

        if (A(i, i) == 0) {
            throw std::runtime_error("Ошибка в матрице A, столбец из нулей");
        }

        // Приводим все остальные строки к виду, в котором этот столбец равен 0
        for (int j = i + 1; j < n; j++) {
            double factor = A(j, i) / A(i, i);
            A.row(j) -= A.row(i) * factor;
        }
    }
   
    // Подставляем обратно
    Eigen::VectorXd x(n);
    for (int i = n - 1; i >= 0; i--) {
        double sum = A(i, n);
        for (int j = i + 1; j < n; j++) {
            sum -= A(i, j) * x(j);
        }
        x(i) = sum / A(i, i);
    }

    return x;
}


int main() {
    std::string filename = "AB.csv";
    std::vector<std::vector<double>> rcsv = readRCSVFromCSV(filename); 
    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> A = convertRCSVToEigen(rcsv);

    std::cout << "A = " << std::endl << A << std::endl;
    Eigen::VectorXd x = gauss(A);
    std::cout << "x = " << std::endl << x << std::endl;

    std::string resultFilename = "x.csv";
    writeEigenVectorToCSV(x, resultFilename);

    return 0;
}
