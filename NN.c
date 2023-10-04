#define NN_IMPLEMENTATION_
#include "NN.h"


float training_data[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0
};


// void backpropogation(NN nn, NN g,)


int main(void) {
    srand(time(0));

    float learningrate = 1;


    int stride = 3;
    int n = (sizeof(training_data)/sizeof(training_data[0]))/3;
    Matrix train_in = {
        .rows = n,
        .cols = 2,
        .stride = stride,
        .mptr = training_data
    };
    Matrix train_out = {
        .rows = n,
        .cols = 1,
        .stride = stride,
        .mptr = training_data + 2
    };

    int architecture[] = {2, 3, 1};
    NN nn = nn_malloc(architecture, ARRAY_LENGTH(architecture));
    NN g = nn_malloc(architecture, ARRAY_LENGTH(architecture));

    nn_rand(nn, 0, 1);
    NN_PRINT(nn);
    matrix_copy(nn.a[0], matrix_row(train_in, 1));
    nn_forward(nn);
   // MATRIX_PRINT(NN_OUTPUT(nn));

    printf("%f\n", nn_cost(nn, train_in, train_out));
    for (int i = 0; i < 1000000; i++) {
        nn_backpropogation(nn, g, train_in, train_out);
        nn_learn(nn, g, learningrate);

        if (i % 50000 == 0)
            printf("%f\n", nn_cost(nn, train_in, train_out));
    }

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            MATRIX_ELEM(NN_INPUT(nn), 0, 0) = i;
            MATRIX_ELEM(NN_INPUT(nn), 0, 1) = j;
            nn_forward(nn);
            printf("%d ^ %d = %f\n", i, j, MATRIX_ELEM(NN_OUTPUT(nn), 0, 0));
        }
    }

    return 0;
}