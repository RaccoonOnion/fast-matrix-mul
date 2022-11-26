#include <cstdlib>
#include <iostream>
#include <fstream>
#include <ctime>
using namespace std;
int main(int argc, char* argv[]) 
{
    int num = 1; // number of datasets to generate
    long upper_bound = RAND_MAX;
    size_t size = 16;
    if (argc >= 2)
    {
        num = stoi(argv[1]);
    }
    if (argc >= 3)
    {
        upper_bound = stol(argv[2]);
    }
    if (argc == 4)
    {
        size = stoi(argv[3]);
    }

    size_t N = (size_t) size * (size_t) size;
    cout << num << endl;
    cout << upper_bound << endl;
    cout << N << endl;

    for (int i = 0; i < num; i++)
    {
        // Create and open a text file
        ofstream file("./data/dataset-" + to_string(upper_bound) + "-" + to_string(size) + "-" + to_string(i) +".txt");
        srand(time(nullptr)); // use current time as seed for random generator
        string buffer = "";
        for (size_t j = 0; j < N; j++)
        {
            float x = ((float)rand()/(float)(RAND_MAX)) * upper_bound;
            // cout << x << endl;
            buffer += (to_string(x) + " ");
            if(j%1000000 == 0)
            {
                // Write to the file
                file << buffer;
                // Clear buffer string
                buffer = "";
            }
            if(j + 1 == N)
            {
                file << buffer;
            }
        }
        // Close the file
        file.close();
    }

}