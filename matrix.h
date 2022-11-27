#ifndef _MATRIX_H
#define _MATRIX_H

#include <stdbool.h> // for bool
#include <stdlib.h>// for size_t

// Some colors
#define RESET   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */
#define BOLDBLACK   "\033[1m\033[30m"      /* Bold Black */
#define BOLDRED     "\033[1m\033[31m"      /* Bold Red */
#define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
#define BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
#define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m"      /* Bold Magenta */
#define BOLDCYAN    "\033[1m\033[36m"      /* Bold Cyan */
#define BOLDWHITE   "\033[1m\033[37m"      /* Bold White */

typedef struct Matrix_{
    size_t rows;
    size_t cols;
    float * data;
} Matrix;

Matrix * createMat(size_t rows, size_t cols);
bool initMat(Matrix * p, float * data);
bool releaseMat(Matrix * p);
bool matmul_plain(const Matrix * input1, const Matrix * input2, Matrix * output);
bool matmul_improved(const Matrix * input1, const Matrix * input2, Matrix * output);
bool printMat(const Matrix * p);
Matrix * transposeMat(const Matrix * p);
#endif