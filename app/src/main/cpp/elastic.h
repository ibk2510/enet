//
// Created by IBK on 5/11/2021.
//
#ifndef SHALLOLEARNING_ELASTIC_H
#define SHALLOLEARNING_ELASTIC_H


#include <iostream>
#include <fstream>
#include <vector>
#include"eigen3/Eigen/Dense"


using namespace std;
using namespace Eigen;

class ETL
{
    string data_path;
    string delimiter;
    bool include_headers;

public:
    ETL(string data_path, string delimiter, bool include_headers) : data_path(data_path), delimiter(delimiter), include_headers(include_headers) {}
    vector<vector<string>> readCSV();
    Eigen::MatrixXd CSVtoEigen(vector<vector<string>> dataset, int rows, int cols);
    MatrixXd Normalize(MatrixXd data);
    auto Mean(MatrixXd data) -> decltype(data.colwise().mean());
    auto Std(MatrixXd data) -> decltype(((data.array().square().colwise().sum()) / (data.rows() - 1)).sqrt());

    std::tuple<MatrixXd, MatrixXd, MatrixXd, MatrixXd> TrainTestSplit(MatrixXd data, float train_size);
//    int csvWrite(const MatrixXd &inputMatrix, const string &fileName, const streamsize dPrec);
};

class ElasticNet
{

    int iterations, m, n; // m---> rows n---> cols
    float learning_rate, l1_penality, l2_penality;
    double b, db;
    MatrixXd X;
    VectorXd Y, y_pred, W, dw;

public:
    ElasticNet(float learning_rate, int iterations, float l1_penality, float l2_penality) : learning_rate(learning_rate), iterations(iterations), l1_penality(l1_penality), l2_penality(l2_penality) {}
    void fit(MatrixXd X, VectorXd Y);
    void updateWeights();
    VectorXd predict(MatrixXd X);
};

#endif SHALLOLEARNING_ELASTIC_H
