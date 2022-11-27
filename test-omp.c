#include <stdio.h>
#include <time.h>
#include <omp.h>

int main() {
    double start, end;
    double runTime;
    start = omp_get_wtime();
    int num = 1,primes = 0;

    int limit = 1000000;

#pragma omp parallel for schedule(dynamic) reduction(+ : primes)
    for (num = 1; num <= limit; num++) { 
        int i = 2; 
        while(i <= num) { 
            if(num % i == 0)
                break;
            i++; 
        }
        if(i == num)
            primes++;
//      printf("%d prime numbers calculated\n",primes);
    }

    end = omp_get_wtime();
    runTime = end - start;
    printf("This machine calculated all %d prime numbers under %d in %g seconds\n",primes,limit,runTime);

    return 0;
}
//no omp: 35.7507
//omp: 35.4989
// O3:23.601
// real OMP: 14.2138
// OMP O3: 7.17535