#include <gtest/gtest.h>
#include <fstream>
#include <cstdio>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <random>

std::vector<std::vector<double>> readRCSVFromCSV(std::string& filename);
Eigen::MatrixXd convertRCSVToEigen(std::vector<std::vector<double>> rcsv);
Eigen::VectorXd gauss(Eigen::Matrix<double, -1, -1, Eigen::RowMajor> A);

// Хелпер для создания файла с CSV
void createCSV(const std::string& filename, const std::string& content) {
    std::ofstream ofs(filename);
    ofs << content;
    ofs.close();
}

// Хелпер для удаления файла
void removeFile(const std::string& filename) {
    std::remove(filename.c_str());
}

TEST(CSVParsingTest, ReadsSimpleCSV) {
    std::string filename = "test_simple.csv";
    createCSV(filename, "A,B,C\n4,5,6\n7,8,9\n");

    auto rcsv = readRCSVFromCSV(filename);
    ASSERT_EQ(rcsv.size(), 2);
    ASSERT_EQ(rcsv[0].size(), 3);
    EXPECT_DOUBLE_EQ(rcsv[0][2], 6.0);

    removeFile(filename);
}

TEST(MatrixConversionTest, ConvertsToEigenMatrix) {
    std::vector<std::vector<double>> rcsv = {
        {1, 2, 3},
        {4, 5, 6}
    };
    Eigen::MatrixXd A = convertRCSVToEigen(rcsv);

    ASSERT_EQ(A.rows(), 2);
    ASSERT_EQ(A.cols(), 3);
    EXPECT_DOUBLE_EQ(A(1, 2), 6.0);
}

TEST(GaussianEliminationTest, Solves3x3System) {
    // Система 3x3
    Eigen::MatrixXd A(3, 4);
    A << 2, 1, -1, 8,
        -3, -1, 2, -11,
        -2, 1, 2, -3;

    Eigen::VectorXd x = gauss(A);

    ASSERT_EQ(x.size(), 3);
    EXPECT_NEAR(x[0], 2.0, 1e-6);
    EXPECT_NEAR(x[1], 3.0, 1e-6);
    EXPECT_NEAR(x[2], -1.0, 1e-6);
}

TEST(GaussianEliminationTest, ThrowsOnSingularMatrix) {
    Eigen::MatrixXd A(2, 3);
    A << 1, 2, 3,
         2, 4, 6; // 2 строка линейно зависимая

    EXPECT_THROW(gauss(A), std::runtime_error);
}

TEST(GaussianEliminationTest, SolvesBigRandomMatrix) {
    int n = 1000;

    // Mersenne Twister RNG для генерации случайных чисел
    std::mt19937 rng(12345); // Фиксируем seed для тестирования
    std::uniform_real_distribution<double> dist(0, 100);

    // Матрица n x n
    Eigen::MatrixXd A(n, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A(i, j) = dist(rng);
        }
    }

    Eigen::VectorXd b(n);
    for (int i = 0; i < n; i++) {
        b(i) = dist(rng);
    }

    // Создаем матрицу с стобцом свободных коэффициентов [A | b] (n x (n+1))
    Eigen::MatrixXd augmentedMatrix(n, n + 1);
    augmentedMatrix << A, b;
    Eigen::VectorXd x = gauss(augmentedMatrix);

    ASSERT_EQ(x.size(), n);

    // Проверяем решение. Должно быть: A * x = b
    Eigen::VectorXd Ax = A * x;
    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(Ax(i), b(i), 1e-6);
    }
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}