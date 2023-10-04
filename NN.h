#ifndef NN_H_
#define NN_H_

#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <time.h>
#include <math.h>


#define ARRAY_LENGTH(xs) sizeof(xs)/sizeof(xs[0])


typedef struct {
    int rows, cols;
    int step_size;
    float * mptr;
} Matrix;


#define MATRIX_ELEM(M, i, j) (M).mptr[(i)*M.step_size + (j)]


Matrix matrix_malloc(int rows, int cols);
void matrix_mult(Matrix dest, Matrix A, Matrix B);
void matrix_rand(Matrix M, float low, float high);
Matrix matrix_row(Matrix M, int row);
void matrix_copy(Matrix dest, Matrix src);
void matrix_fill(Matrix M, float val);
void matrix_sum(Matrix dest, Matrix A);
void matrix_sigmoid(Matrix M);
void matrix_print(Matrix M, const char *name, int padding);
#define MATRIX_PRINT(nn) matrix_print(nn, #nn, 0) //# gives tokenized representation of nn


//n is some arbitrary integer, sizes change between layers
//W is in R^(nxn)
//x, b, a are in R^n
//y is in R^m 

//W*x = z -> a = sigmoid(z) -> W*a = a_next -> ... -> a_out
//c = (a_out - y)^2

//(p is for partial)
//pC/pw = 1/n(Summation(i=1->n)(2(a_i-y_i)a_i(1-a_i)x_i))
//pC/pb = 1/n(Summation(i=1->n)(2(a_i-y_i)a_i(1-a_i))


typedef struct {
    int lcount;
    Matrix * W;
    Matrix * b;
    Matrix * a; 
} NN;


#define NN_INPUT(nn) (nn).a[0]
#define NN_OUTPUT(nn) (nn).a[(nn).lcount]


NN nn_malloc(int * architecture, int arch_count);
void nn_clean(NN nn);
void nn_rand(NN nn, float low, float high);
void nn_forward(NN nn);
float nn_cost(NN nn, Matrix train_in, Matrix train_out);
void nn_finite_difference(NN nn, NN g, float delta, Matrix train_in, Matrix train_out);
void nn_backpropogation(NN nn, NN g, Matrix train_in, Matrix train_out);
void nn_learn(NN nn, NN g, float learningrate);
void nn_print(NN nn, const char *name);
#define NN_PRINT(nn) nn_print(nn, #nn) //# gives tokenized representation of nn

#endif //NN_H_


#ifdef NN_IMPLEMENTATION_

float sigmoid_f(float x) {
    return (1.0f / (1.0f + expf(-x)));
}


float rand_f(void) {
    return (float)rand() / (float)RAND_MAX;
}


void matrix_rand(Matrix M, float low, float high) {
    for (int i = 0; i < M.rows; i++) {
        for (int j = 0; j < M.cols; j++) {
            MATRIX_ELEM(M, i, j) = (rand_f()*(high - low) - low);
        }
    }
}


Matrix matrix_row(Matrix M, int row) {
    Matrix A;
    A.rows = 1;
    A.cols = M.cols;
    A.step_size = M.step_size;
    A.mptr = &MATRIX_ELEM(M, row, 0);

    return A;
}


void matrix_copy(Matrix dest, Matrix src) {
    assert(dest.rows == src.rows);
    assert(dest.cols == src.cols);
    for (int i = 0; i < dest.rows; i++) {
        for (int j = 0; j < dest.cols; j++) {
            MATRIX_ELEM(dest, i, j) = MATRIX_ELEM(src, i, j);
        }
    }
}


void matrix_fill(Matrix M, float val) {
    for (int i = 0; i < M.rows; i++) {
        for (int j = 0; j < M.cols; j++) {
            MATRIX_ELEM(M, i, j) = val;
        }
    }
}


void matrix_sum(Matrix dest, Matrix A) {
    assert(dest.rows == A.rows);
    assert(dest.cols == A.cols);

    for (int i = 0; i < dest.rows; i++) {
        for (int j = 0; j < dest.cols; j++) {
            MATRIX_ELEM(dest, i , j) += MATRIX_ELEM(A, i, j);
        }
    }
}


void matrix_mult(Matrix dest, Matrix A, Matrix B) {
    matrix_fill(dest, 0);
    assert(A.cols == B.rows);
    assert(dest.rows == A.rows);
    assert(dest.cols == B.cols);

    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < B.cols; j++) {
            for (int k = 0; k < A.cols; k++) 
                MATRIX_ELEM(dest, i , j) += (MATRIX_ELEM(A, i, k)*MATRIX_ELEM(B, k, j));
        }
    }
}


Matrix matrix_malloc(int rows, int cols) {
    Matrix M;
    M.rows = rows;
    M.cols = cols;
    M.step_size = cols;
    M.mptr = malloc(sizeof(float)*rows*cols); assert(M.mptr != NULL);

    return M;
}

void matrix_sigmoid(Matrix M) {
    for (int i = 0; i < M.rows; i++) {
        for (int j = 0; j < M.cols; j++) {
                MATRIX_ELEM(M, i, j) = sigmoid_f(MATRIX_ELEM(M, i, j)); 
        }
    }
}


void matrix_print(Matrix M, const char *name, int padding) {
    printf("%*s%s = [\n", padding, "", name);
    for (int i = 0; i < M.rows; i++) {
        printf("%*s    ", padding, "");
        for (int j = 0; j < M.cols; j++) {
            printf("%.4f  ", MATRIX_ELEM(M, i, j));
        }
        printf("\n");
    }
    printf("%*s]\n", padding, "");
}


NN nn_malloc(int * architecture, int arch_count) {
    assert(arch_count > 0);
    NN nn;
    nn.lcount = arch_count - 1;

    nn.W = malloc(sizeof(*nn.W)*nn.lcount);
    nn.b = malloc(sizeof(*nn.b)*nn.lcount);
    nn.a = malloc(sizeof(*nn.a)*(nn.lcount + 1));
    assert(nn.W != NULL);
    assert(nn.b != NULL);
    assert(nn.a != NULL);

    nn.a[0] = matrix_malloc(1, architecture[0]);
    for (int i = 1; i < arch_count; i++) {
        nn.W[i-1] = matrix_malloc(nn.a[i-1].cols, architecture[i]);
        nn.b[i-1] = matrix_malloc(1, architecture[i]);
        nn.a[i] = matrix_malloc(1, architecture[i]);
    }

    return nn;
}


void nn_clean(NN nn) {
    for (int l = 0; l < nn.lcount; l++) {
        matrix_fill(nn.W[l], 0);
        matrix_fill(nn.b[l], 0);
        matrix_fill(nn.a[l], 0);
    }
    matrix_fill(nn.a[nn.lcount], 0);
}


void nn_print(NN nn, const char *name) {
    char buff[256];
    printf("%s = [\n", name);
    for (int i = 0; i < nn.lcount; i++) {
        snprintf(buff, sizeof(buff), "W%d", i);
        matrix_print(nn.W[i], buff, 4);
        snprintf(buff, sizeof(buff), "b%d", i);
        matrix_print(nn.b[i], buff, 4);
    }
    printf("]\n");
}


void nn_rand(NN nn, float low, float high) {
    for (int i = 0; i < nn.lcount; i++) {
        matrix_rand(nn.W[i], low, high);
        matrix_rand(nn.b[i], low, high);
    }
}


void nn_forward(NN nn) {
    for (int i = 0; i < nn.lcount; i++) {
        matrix_mult(nn.a[i+1], nn.a[i], nn.W[i]);
        matrix_sum(nn.a[i+1], nn.b[i]);
        matrix_sigmoid(nn.a[i+1]);
    }
}


float nn_cost(NN nn, Matrix train_in, Matrix train_out) {
    assert(train_in.rows == train_out.rows);
    assert(train_out.cols == NN_OUTPUT(nn).cols);

    int n = train_in.rows;

    float cost = 0;
    for (int i = 0; i < n; i++) {
        Matrix x = matrix_row(train_in, i);
        Matrix y = matrix_row(train_out, i);

        matrix_copy(NN_INPUT(nn), x);
        nn_forward(nn);

        int q = train_out.cols;
        for (int j = 0; j < q; j++) {
            float d = MATRIX_ELEM(NN_OUTPUT(nn), 0, j) - MATRIX_ELEM(y, 0, j);
            cost += d*d;
        }
    }
    return cost/=n;
}


void nn_backpropogation(NN nn, NN g, Matrix train_in, Matrix train_out) {
    assert(train_in.rows == train_out.rows);
    assert(NN_OUTPUT(nn).cols == train_out.cols);

    nn_clean(g);

    int n = train_in.rows;

    for(int i = 0; i < n; i++) { //current sample
        matrix_copy(NN_INPUT(nn), matrix_row(train_in, i));
        nn_forward(nn);

        for (int j = 0; j <= nn.lcount; j++) {
            matrix_fill(g.a[j], 0);
        }

        for (int j = 0; j < train_out.cols; j++) {
            MATRIX_ELEM(NN_OUTPUT(g), 0, j) = MATRIX_ELEM(NN_OUTPUT(nn), 0, j) - MATRIX_ELEM(train_out, i, j);
        }

        for (int l = nn.lcount; l > 0; l--) { //current layer
            for (int j = 0; j < nn.a[l].cols; j++) { //current activation
                float a = MATRIX_ELEM(nn.a[l], 0, j);
                float da = MATRIX_ELEM(g.a[l], 0, j);

                MATRIX_ELEM(g.b[l-1], 0, j) += 2*da*a*(1 - a);
                for (int k = 0; k < nn.a[l-1].cols; k++) { //previous activation
                    float pa = MATRIX_ELEM(nn.a[l-1], 0, k);
                    float pw = MATRIX_ELEM(nn.W[l-1], k, j);

                    MATRIX_ELEM(g.W[l-1], k, j) += 2*da*a*(1 - a)*pa; //derivative of the cost using sigmoid
                    MATRIX_ELEM(g.a[l-1], 0, k) += 2*da*a*(1 - a)*pw; //sig()' = sig()(1 - sig())
                }
            }
        }
    }
    for(int i = 0; i < g.lcount; i++) {
        for (int j = 0; j < g.W[i].rows; j++) { 
            for(int k = 0; k < g.W[i].cols; k++) {
                MATRIX_ELEM(g.W[i], j, k) /= n;
            }
        }
        for (int j = 0; j < g.b[i].rows; j++) { 
            for(int k = 0; k < g.b[i].cols; k++) {
                MATRIX_ELEM(g.b[i], j, k) /= n;
            }
        }
    }
}


void nn_learn(NN nn, NN g, float learningrate) {
    for (int i = 0; i < nn.lcount; i++) {
        for (int j = 0; j < nn.W[i].rows; j++) {
            for (int k = 0; k < nn.W[i].cols; k++) {
                MATRIX_ELEM(nn.W[i], j, k) -= learningrate*MATRIX_ELEM(g.W[i], j, k);
            }
        }
        for (int j = 0; j < nn.b[i].rows; j++) {
            for (int k = 0; k < nn.b[i].cols; k++) {
                MATRIX_ELEM(nn.b[i], j, k) -= learningrate*MATRIX_ELEM(g.b[i], j, k);
            }
        }
    }
}

#endif //NN_IMPLEMENTATION