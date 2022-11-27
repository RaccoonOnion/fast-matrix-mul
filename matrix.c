#include <stdio.h>
#include <string.h>
#include "matrix.h"

#include <omp.h>
//  export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
//  export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"

#include <arm_neon.h>

//#include <cblas.h>
//export LDFLAGS="-L/opt/homebrew/opt/openblas/lib"
//export CPPFLAGS="-I/opt/homebrew/opt/openblas/include"

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
        free(p->data);
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

bool printMat(const Matrix * p)
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
        size_t rows = p->rows;
        size_t cols = p->cols;
        const float * pData = p->data;
        printf("The Matrix is:\n[");
        for(size_t i = 0; i < rows * cols; i++)
        {
            if(i%cols != 0)
                printf(", ");
            else if(i != 0)
                printf("]\n[");
            printf("%f", *(pData++));
        }
        printf("]\n");
        return true;
    }
}

bool initMat(Matrix * p, float * data)
{
    if (p == NULL)
    {
        fprintf(stderr, RED"The 1st argument you input is a NULL pointer.\n"RESET);
        return false;
    }

    if(p->data == NULL)
    {
        fprintf(stderr, RED"The data pointer in the matrix struct is a NULL pointer.\n"RESET);
        return false;
    }

    if(data == NULL)
    {
        fprintf(stderr, RED"The 2nd argument you input is a NULL pointer.\n"RESET);
        return false;
    }
    else
    {
        size_t rows = p->rows;
        size_t cols = p->cols;
        memcpy(p->data, data, sizeof(float)*rows*cols);
        return true;
    }
}

Matrix * transposeMat(const Matrix * p);

float dotproduct(const float * f1, const float * f2, size_t len)
{
    float result = 0.0f;

    for(size_t i = 0; i < len; i++)
    {
        result += *(f1++) * *(f2++);
    }
    return result;
}

float dotproduct_neon(const float * p1, const float * p2, size_t n)
{
    float sum[4] = {0};
    float32x4_t a, b;
    float32x4_t c = vdupq_n_f32(0);

    for (size_t i = 0; i < n; i+=4)
    {
        a = vld1q_f32(p1 + i);
        b = vld1q_f32(p2 + i);
        c = vaddq_f32(c, vmulq_f32(a, b));
    }
    vst1q_f32(sum, c);
    return (sum[0]+sum[1]+sum[2]+sum[3]);
}

bool matmul_improved(const Matrix * input1, const Matrix * input2, Matrix * output)
{
    // input1: a*b; input2 b*c; input2_T c*b; output: a*c
    Matrix * input2_T = transposeMat(input2);
    size_t a = input1->rows;
    size_t b = input1->cols;
    size_t c = input2_T->rows;
    size_t l = a * c;
    const float * ptr1 = input1->data;
    const float * ptr2 = input2_T->data;
    float * ptr_out = output->data; 

    #pragma omp parallel for
    for (size_t i = 0; i < l; i++)
    {
        *(ptr_out + i) = dotproduct_neon(( ptr1 + (i/c) * b ),( ptr2 + (i%c) * b ), b);
        //printf("i is %lu, product is: %f\n",i,dotproduct_neon(( ptr1 + (i/c) * b ),( ptr2 + (i%c) * b ), b));
    }
    releaseMat(input2_T);
    return true;
}

Matrix * transposeMat(const Matrix * p)
{
    size_t rp = p->rows;
    size_t cp = p->cols;
    const float * ptr_p = p->data;
    size_t l = rp * cp;

    Matrix * t = (Matrix *) malloc(sizeof(Matrix));
    t->rows = cp;
    t->cols = rp;
    t->data = (float *) malloc( l * sizeof(float));
    float * ptr_t = t->data;

    #pragma omp parallel for
    for(size_t i = 0; i < l; i++)
    {
        // printf("i is %lu, j is %lu\n",i,(i%rp)*cp + (i/rp));
        *(ptr_t++) = ptr_p[(i%rp)*cp + (i/rp)];
    }
    return t;
}