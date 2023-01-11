#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "matrix.h"
// Driver code


int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        fprintf(stderr, RED"We need two arguments to compute! The number of inputs you give is: %d\n"RESET, argc-1);
        return 0;
    }
    FILE* ptr1;

    int size = atoi(argv[2]);
    size_t N = (size_t) size * (size_t) size;

	// Read matrix
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
    //printMat(matA); // print matA
    printf("MatA init finished!\n");

    free(data1);

    for(int i = 0; i < 3; i++)
    {
        //tests
        clock_t begin = clock();
        Matrix * matB = transposeMatFast(matA);
        clock_t end = clock();
        double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
        printf("%lfs\n", time_spent);
        releaseMat(matB);
    }


    
    //printMat(matB); // print matB

    releaseMat(matA);


	return 0;
}
