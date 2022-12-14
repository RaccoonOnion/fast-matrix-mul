#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "matrix.h"
// Driver code


int main(int argc, char* argv[])
{
    if (argc < 4)
    {
        fprintf(stderr, RED"We need three arguments to compute! The number of inputs you give is: %d\n"RESET, argc-1);
        return 0;
    }
    FILE* ptr1;
    FILE* ptr2;
    int size = atoi(argv[3]);
    size_t N = (size_t) size * (size_t) size;

	// Read matrix 1
	ptr1 = fopen(argv[1], "r");

	if (ptr1 == NULL) {
		fprintf(stderr, RED"File %s can't be opened \n"RESET, argv[1]);
        return 0;
	}
	printf("Start reading file %s.\n", argv[1]);

    float num;
    float * data1 = (float *) malloc( N * sizeof(float));
    for (size_t i = 0; i < N; i++)
    {
        fscanf(ptr1,"%f",&num);
        data1[i] = num;
    }
    printf("Dataset1 reading finished!\n");

	// Closing the file
	fclose(ptr1);
    Matrix * matA = createMat(size, size);
    initMat(matA, data1);
    printf("MatA init finished!\n");
    //printMat(matA);

	// Read matrix 2
	ptr2 = fopen(argv[2], "r");

	if (ptr2 == NULL) {
		fprintf(stderr, RED"File %s can't be opened \n"RESET, argv[2]);
        return 0;
	}
	printf("Start reading file %s.\n", argv[2]);

    float * data2 = (float *) malloc( N * sizeof(float));
    for (size_t i = 0; i < N; i++)
    {
        fscanf(ptr2,"%f",&num);
        data2[i] = num;
    }
    printf("Dataset1 reading finished!\n");
	// Closing the file
	fclose(ptr2);
    Matrix * matB = createMat(size, size);
    initMat(matB, data2);
    printf("MatB init finished!\n");
    //printMat(matB);
    free(data1);
    free(data2);

    Matrix * matC = createMat(size, size);

    // warmups
    matmul_improved(matA, matB, matC);
    matmul_improved(matA, matB, matC);
    matmul_improved(matA, matB, matC);

    double start, end;
    double runTime;
    //tests
    for (int i = 0; i < 3; i++)
    {
        start = omp_get_wtime();
        matmul_improved(matA, matB, matC);
        end = omp_get_wtime();
        runTime = end - start;
        printf("%g\n",runTime);

    }
    //printMat(matC);
    releaseMat(matA);
    releaseMat(matB);
    releaseMat(matC);
	return 0;
//[0.254490, 0.952151]
//[7.832922, 6.014445]

//[15.300124, 10.182752, 8.825740, 13.476320]
//[37.351601, 17.637680, 8.114429, 34.055958]
//[34.695450, 22.834732, 10.766151, 24.758505]
//[24.400372, 24.199692, 19.972271, 28.194981]
}
