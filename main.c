#include <stdio.h>
#include <string.h>
#include "matrix.h"

int main()
{
    Matrix * matA = createMat(2, 3);
    Matrix * matB = createMat(3, 2);
    Matrix * matC = createMat(2, 2);
    Matrix * matD = createMat(3, 3);
    Matrix * matNULL = NULL;

    // write a method for value assignemnt!!
    float m1[6] = {1.0f,2.0f,3.0f,4.0f,5.0f,6.0f};
    memcpy(matA->data, m1, sizeof(float)*6);
    printf("A is:\n\
    [%f, %f, %f]\n\
    [%f, %f, %f]\n",
    matA->data[0], matA->data[1], matA->data[2],
    matA->data[3], matA->data[4], matA->data[5]);
    memcpy(matB->data, m1, sizeof(float)*6);
    printf("B is:\n\
    [%f, %f, %f]\n\
    [%f, %f, %f]\n",
    matB->data[0], matB->data[1], matB->data[2],
    matB->data[3], matB->data[4], matB->data[5]);
    if (! matmul_plain(matA, matB, matC))
        fprintf(stderr, "Matrix multiplication failed.\n");
    else // we need better way to print the matrix result!!
    {
        printf("Result is:\n\
        [%f, %f]\n\
        [%f, %f]\n",
        matC->data[0], matC->data[1],
        matC->data[2], matC->data[3]);
    }

    //more tests
    matmul_plain(matA, matB, matD);
    matmul_plain(matNULL, matB, matC);

    matmul_plain(matB, matA, matD);
    printf("Result is:\n\
    [%f, %f, %f]\n\
    [%f, %f, %f]\n\
    [%f, %f, %f]\n",
    matD->data[0], matD->data[1], matD->data[2],
    matD->data[3], matD->data[4], matD->data[5],
    matD->data[6], matD->data[7], matD->data[8]);
    return 0;

}