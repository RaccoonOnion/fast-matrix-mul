#ifndef _MATRIX_H
#define _MATRIX_H

#include <stdbool.h> // for bool
#include <stdlib.h>// for size_t

#include <omp.h> // OpenMP

// Red color macro for error output
#define RESET   "\033[0m"
#define RED     "\033[31m"

// Matrix struct
typedef struct Matrix_{ 
    size_t rows;
    size_t cols;
    float * data;
} Matrix;

Matrix * createMat(size_t rows, size_t cols); // method to create an empty matrix
bool initMat(Matrix * p, float * data);// method to initialize data in a matrix
bool releaseMat(Matrix * p);// method to release a matrix
bool matmul_plain(const Matrix * input1, const Matrix * input2, Matrix * output);// naive method for matrix multiplicaiton
bool matmul_improved(const Matrix * input1, const Matrix * input2, Matrix * output);// improved method for matrix multiplicaiton
bool printMat(const Matrix * p);// method to print a matrix
Matrix * transposeMatFast(const Matrix * p);// a fast method to get the transpose of a matrix, assume the input is valid!
#endif