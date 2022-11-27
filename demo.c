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
    initMat(matA, m1);
    printMat(matA);
    initMat(matB, m1);
    printMat(matB);
    if (! matmul_plain(matA, matB, matC))
        fprintf(stderr, "Matrix multiplication failed.\n");
    else // we need better way to print the matrix result!!
    {
        printMat(matC);
    }

    //more tests
    matmul_plain(matA, matB, matD);
    matmul_plain(matNULL, matB, matC);

    matmul_plain(matB, matA, matD);
    printMat(matD);

    releaseMat(matA);
    releaseMat(matB);
    releaseMat(matC);
    releaseMat(matD);
    releaseMat(matNULL);

    return 0;

}