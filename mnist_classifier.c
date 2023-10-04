#define NN_IMPLEMENTATION_
#include "NN.h"

#include <string.h>


#define CHUNK 4096
#define N 50000


int main(void) {
    srand(time(0));
    float learningrate = 0.1f;

    char buff[CHUNK];
    FILE * f1 = fopen("MNIST_train.txt", "r");
    assert(f1 != NULL);


    Matrix train_in = matrix_malloc(N, 28*28);
    Matrix train_out = matrix_malloc(N, 10);

    float * row = malloc(sizeof(int)*28*28);
    const char * delim = ",\n";
    int c = 0;
    for (int i = 0; i < N; i++) {
        char * str = fgets(buff, CHUNK, f1);
        char * token = strtok(str, delim);
        int out = atoi(token);

        matrix_fill(matrix_row(train_out, i), 0);
        MATRIX_ELEM(matrix_row(train_out, i), 0, out) = 1;
        token = strtok(str, delim);
        int count = 0;
        while (token && count < 28*28) {
            count++;
            row[c++] = atoi(token);
            token = strtok(NULL, delim);
        }
        if (count == 28*28) 
            str = fgets(buff, CHUNK, f1);

        Matrix mrow = {.rows = 1, .cols = 28*28, .stride = 0, .mptr = row};
        matrix_copy(matrix_row(train_in, i), mrow);
    }

    FILE * f2 = fopen("MNIST_test.txt", "r");
    assert(f2 != NULL);

    Matrix train_in1 = matrix_malloc(N, 28*28);
    Matrix train_out1 = matrix_malloc(N, 10);

    float * row1 = malloc(sizeof(int)*28*28);
    const char * delim1 = ",\n";
    int c1 = 0;
    for (int i = 0; i < 100; i++) {
        char * str1 = fgets(buff, CHUNK, f2);
        char * token1 = strtok(str1, delim1);
        int out1 = atoi(token1);

        matrix_fill(matrix_row(train_out1, i), 0);
        MATRIX_ELEM(matrix_row(train_out1, i), 0, out1) = 1;
        token1 = strtok(str1, delim1);
        while (token1) {
            row1[c1++] = atoi(token1);
            token1 = strtok(NULL, delim1);
        }
        Matrix mrow1 = {.rows = 1, .cols = 28*28, .stride = 0, .mptr = row1};
        matrix_copy(matrix_row(train_in1, i), mrow1);
    }



    int architecture[] = {28*28, 12, 12, 10};
    NN nn = nn_malloc(architecture, ARRAY_LENGTH(architecture));
    NN g = nn_malloc(architecture, ARRAY_LENGTH(architecture));
    nn_rand(nn, 0, 1);


    for (int i = 0; i < 100; i++) {
        nn_backpropogation(nn, g, train_in, train_out);
        nn_learn(nn, g, learningrate);
        if (i % 10 == 0)
            printf("c = %f\n", nn_cost(nn, train_in, train_out));
    }


    for (int i = 0; i < 100; i++) {
        matrix_copy(NN_INPUT(nn), matrix_row(train_in1, i));
        nn_forward(nn);
        printf("******************************************************\n");
        printf("expected values: ");
        MATRIX_PRINT(matrix_row(train_out1, i));
        printf("actual values: ");
        MATRIX_PRINT(matrix_row(NN_OUTPUT(nn), i));
        printf("******************************************************\n\n\n");

    }
}