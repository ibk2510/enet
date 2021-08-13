//
// Created by IBK on 5/11/2021.
//
#include "elastic.h"
#include <vector>
#include <cstdlib>
#include <cmath>
#include <jni.h>
#include "eigen3/Eigen/Dense"


using namespace std;
using namespace Eigen;

// etl class member functions

vector<vector<string>> ETL::readCSV() {
    ifstream file(data_path);
    vector<vector<string>> dataString;
    string line = "";
    while (getline(file, line)) {
        stringstream vectstring(line);
        string each_entry;
        vector<string> vect = {};
        while (getline(vectstring, each_entry, ',')) {
            vect.push_back(each_entry);
        }
        // algorithm::split(vect, line, is_any_of(delimiter));
        dataString.push_back(vect);
    }
    file.close();

    return dataString;
}

MatrixXd ETL::CSVtoEigen(vector<vector<string>> dataset, int rows, int cols) {

    rows = include_headers ? rows - 1 : rows;
    MatrixXd mat(rows, cols);

    if (include_headers) {
        for (long int i = 0; i < rows; i++) {
            for (long int j = 0; j < cols; ++j) {
                // converting the string to float value
                mat(i, j) = atof(dataset[i + 1][j].c_str());
            }
        }
    } else {
        for (long int i = 0; i < rows; i++) {
            for (long int j = 0; j < cols; ++j) {
                // if(dataset[i][j] == ""){
                //     mat(i,j;
                //     continue;
                // }
                // converting the string to float value
                mat(i, j) = atof(dataset[i][j].c_str());
            }
        }
    }
    return mat;
}

tuple<MatrixXd, MatrixXd, MatrixXd, MatrixXd> ETL::TrainTestSplit(MatrixXd data, float train_size) {
    int rows = data.rows();
    // cout<<"rows are" << rows << endl;
    int train_rows = round(train_size * rows);
    int test_rows = rows - train_rows;
    cout << "train rows are" << train_rows << endl;
    cout << "test rows  are" << test_rows << endl;

    MatrixXd train = data.topRows(train_rows);
    MatrixXd x_train = train.leftCols(data.cols() - 1);
    MatrixXd y_train = train.rightCols(1);
    MatrixXd test = data.bottomRows(test_rows);
    MatrixXd x_test = test.leftCols(data.cols() - 1);
    MatrixXd y_test = test.rightCols(1);

    return make_tuple(x_train, y_train, x_test, y_test);
}

auto ETL::Mean(Eigen::MatrixXd data) -> decltype(data.colwise().mean()) {
    return data.colwise().mean();
}

auto ETL::Std(Eigen::MatrixXd data) -> decltype(((data.array().square().colwise().sum()) /
                                                 (data.rows() - 1)).sqrt()) {
    return ((data.array().square().colwise().sum()) / (data.rows() - 1)).sqrt();
}

Eigen::MatrixXd ETL::Normalize(Eigen::MatrixXd data) {
    Eigen::MatrixXd scaled_data = data.rowwise() - Mean(data);
    auto std = Std(scaled_data);
    Eigen::MatrixXd norm = scaled_data.array().rowwise() / std;
    return norm;
}




// elasticnet class member functions

void ElasticNet::fit(MatrixXd X, VectorXd Y) {

    // // updating lambda_1 and lambda_2
    // l1_penality = alpha * l1_ratio;
    // l2_penality = 0.5 * alpha * (1-l1_ratio);
    this->m = X.rows();
    this->n = X.cols();

    this->W = VectorXd::Zero(n);
    this->b = 0;
    this->X = X;
    this->Y = Y;

    for (int i = 0; i < iterations; i++) {
        this->updateWeights();
    }
}

void ElasticNet::updateWeights() {
    y_pred = this->predict(this->X);
    // calculating gradients
    dw = VectorXd::Zero(this->n);

    for (int j = 0; j < this->n; j++) {
        if (this->W[j] > 0) {
            dw[j] = (-(2 * (this->X.col(j)).dot(this->Y - y_pred)) + this->l1_penality +
                     2 * this->l2_penality * this->W[j]) / m;
        } else {
            dw[j] = (-(2 * (this->X.col(j)).dot(this->Y - y_pred)) - this->l1_penality +
                     2 * this->l2_penality * this->W[j]) / m;
        }
    }

    VectorXd Y_minus_y_pred = Y - y_pred;

    db = -2 * (Y_minus_y_pred.sum()) / m;

    // update weights

    this->W = this->W - learning_rate * dw;
    this->b = this->b - learning_rate * db;
};

VectorXd ElasticNet::predict(MatrixXd X) {
    VectorXd vect = X * W;
    for (int i = 0; i < vect.size(); i++) {
        /* code */
        vect[i] += b;
    }
    return vect;
};





// utility functions

double meanAbsoluteError(VectorXd y, VectorXd y_pred) {
    VectorXd result = y - y_pred;
    for (int i = 0; i < result.size(); i++) {
        result[i] = abs(result[i]);
    }
    double mae = result.sum();
    mae /= result.size();
    return mae;
}

int csvWrite(const MatrixXd &inputMatrix, const string &fileName, const streamsize dPrec) {
    int i, j;
    ofstream outputData;
    outputData.open(fileName);
    if (!outputData)
        return -1;
    outputData.precision(dPrec);
    for (i = 0; i < inputMatrix.rows(); i++) {
        for (j = 0; j < inputMatrix.cols(); j++) {
            outputData << inputMatrix(i, j);
            if (j < (inputMatrix.cols() - 1))
                outputData << ",";
        }
        if (i < (inputMatrix.rows() - 1))
            outputData << endl;
    }
    outputData.close();
    if (!outputData)
        return -1;
    return 0;
}

double meanSquareError(VectorXd y, VectorXd y_pred) {
    VectorXd result = y - y_pred;
    for (int i = 0; i < result.size(); i++) {
        result[i] *= result[i];
    }
    double mse = result.sum();
    mse /= result.size();
    return mse;
}

double r2score(VectorXd y, VectorXd y_pred) {
//    tss
    VectorXd rss(y_pred.size());
    for (int i = 0; i < y_pred.size(); i++) {
        rss[i] = y[i] - y_pred[i];
        rss[i] *= rss[i];
    }
    VectorXd tss(y.size());
    double mean = y.mean();
    for (int i = 0; i < y.size(); i++) {
        tss[i] = y[i] - mean;
        tss[i] *= tss[i];
    }
    double tssSum = tss.sum();
    double rssSum = rss.sum();
    double r_2 = 1 - (rssSum / tssSum);
    return r_2;
}


// jni functions
string jstring2string(JNIEnv *env, jstring jStr) {
    if (!jStr)
        return "";

    const jclass stringClass = env->GetObjectClass(jStr);
    const jmethodID getBytes = env->GetMethodID(stringClass, "getBytes", "(Ljava/lang/String;)[B");
    const jbyteArray stringJbytes = (jbyteArray) env->CallObjectMethod(jStr, getBytes,
                                                                       env->NewStringUTF("UTF-8"));

    size_t length = (size_t) env->GetArrayLength(stringJbytes);
    jbyte *pBytes = env->GetByteArrayElements(stringJbytes, NULL);

    string ret = string((char *) pBytes, length);
    env->ReleaseByteArrayElements(stringJbytes, pBytes, JNI_ABORT);

    env->DeleteLocalRef(stringJbytes);
    env->DeleteLocalRef(stringClass);
    return ret;
}

// end of utility functions




// common accessable variable for both test train functions
MatrixXd x_train, y_train, x_test, y_test;
// end of common properties



// JNI calls begin
extern "C" JNIEXPORT jlong JNICALL
Java_com_samsung_shallolearning_MainActivity_train(JNIEnv *env, jobject obj, jstring str) {
    string filename = jstring2string(env, str);

//train
    ETL etl(filename, ",", false);
    vector<vector<string>> data_csv = etl.readCSV();
    std::string hello = data_csv[1][0];
    MatrixXd data_mat = etl.CSVtoEigen(data_csv, data_csv.size(), data_csv[0].size());
    MatrixXd norm = etl.Normalize(data_mat);
    tuple<MatrixXd, MatrixXd, Eigen::MatrixXd, MatrixXd> split_data = etl.TrainTestSplit(data_mat,
                                                                                         0.7);
    tie(x_train, y_train, x_test, y_test) = split_data;
    ElasticNet *enr = new ElasticNet(0.01, 1000, 500, 1);

    x_train = etl.Normalize(x_train);
    x_test = etl.Normalize(x_test);

    enr->fit(x_train, y_train);
    return (jlong) enr; // returning the pointer of the object

}


extern "C" JNIEXPORT jstring JNICALL
Java_com_samsung_shallolearning_MainActivity_test(
        JNIEnv *env,
        jobject, jlong objptr , jdoubleArray inputs) {
    ElasticNet *enr = (ElasticNet *) objptr;
    MatrixXd output = enr->predict(x_test);
//    double mae = meanAbsoluteError(y_test, output);
//    double r_2 = r2score(y_test, output);
//    double mse = meanSquareError(y_test, output);
//    int i =  csvWrite(output , "/storage/emulated/0/Download/out.csv" , 20);
//    delete &enr;
    double* arr = env->GetDoubleArrayElements(inputs , 0);
    MatrixXd matIn(1, 8);
    for(int i =0 ; i<8;i++){
        matIn(0,i) = arr[i];
    }

    output = enr->predict(matIn);
    string out = to_string(output(0,0));
    return env->NewStringUTF(out.c_str());
}



