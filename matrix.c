#include <stdio.h>
#include "matrix.h"

// return NULL if failed
Matrix * createMat(size_t rows, size_t cols)
{
    Matrix * p = NULL;

    if(rows == 0 || cols == 0)
    {
        fprintf(stderr, RED"rows and/or cols is 0.\n"RESET);
        return NULL;
    }

    // allocate memory
    p = (Matrix *) malloc(sizeof(Matrix));
    if( p == NULL )
    {
        fprintf(stderr, RED"Failed to allocate memory for a matrix.\n"RESET);
        return NULL;
    }
    p->rows = rows;
    p->cols = cols;
    p->data = (float *) malloc( p->rows * p->cols * sizeof(float));

    if(p->data == NULL)
    {
        fprintf(stderr, RED"Failed to allocate memory for the matrix data.\n"RESET);
        free(p);
        return NULL;
    }

    return p;
}

bool releaseMat(Matrix * p)
{
    if (p == NULL)
    {
        fprintf(stderr, RED"The pointer you input is a NULL pointer.\n"RESET);
        return false;
    }

    if(p->data == NULL)
    {
        fprintf(stderr, RED"The data pointer in the matrix struct is a NULL pointer.\n"RESET);
        return false;
    }
    else
    {
        free(p);
    }
    return true;
}

bool matmul_plain(const Matrix * input1, const Matrix * input2, Matrix * output)
{
    if(input1 == NULL)
    {
        fprintf(stderr, RED"File %s, Line %d, Function %s(): The 1st parameter is NULL.\n"RESET, __FILE__, __LINE__, __FUNCTION__);
        return false;
    }
    else if(input1->data == NULL)
    {
        fprintf(stderr, RED"%s(): Data pointer of the 1st parameter matrix is NULL.\n"RESET, __FUNCTION__);
        return false;
    }

    if(input2 == NULL)
    {
        fprintf(stderr, RED"File %s, Line %d, Function %s(): The 2nd parameter is NULL.\n"RESET, __FILE__, __LINE__, __FUNCTION__);
        return false;
    }
    else if(input2->data == NULL)
    {
        fprintf(stderr, RED"%s(): Data pointer of the 2nd parameter matrix is NULL.\n"RESET, __FUNCTION__);
        return false;
    }

    if(output == NULL)
    {
        fprintf(stderr, RED"File %s, Line %d, Function %s(): The 3rd parameter is NULL.\n"RESET, __FILE__, __LINE__, __FUNCTION__);
        return false;
    }
    else if(output->data == NULL)
    {
        fprintf(stderr, RED"%s(): Data pointer of the 3rd parameter matrix is NULL.\n"RESET, __FUNCTION__);
        return false;
    }

    if (input1->rows != output->rows || input2->cols != output->cols || input1->cols != input2->rows)
    {
        fprintf(stderr, RED"The input and output sizes do not match.\n");
        fprintf(stderr, "Their sizes are (%zu, %zu), (%zu, %zu) and (%zu, %zu)\n"RESET,
        input1->rows, input1->cols,
        input2->rows, input2->cols,
        output->rows, output->cols);
        return false;
    }

    // Plain version of matrix multiplication using loops
    // type a = ptrMat1->rows;
    // type b = ptrMat1->cols;
    // type c = ptrMat2->cols;
    // type num = a*c;

    // float ptrData[num];

    for (int row = 0; row < input1->rows; row++)
    {
        for (int col = 0; col < input2->cols; col++)
        {
            float sum = 0.0f;
            for (int i = 0; i < input1->cols; i++)
            {
                sum += (input1->data[input1->cols * row + i]) * (input2->data[i * input2->cols + col]);
            }
            output->data[input2->cols * row + col] = sum;
        }
    }

    return true;
}