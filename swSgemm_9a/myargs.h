//#define DMA_FAST
//#define TIMING
#define CLOCKRATE    (1.45e9)

struct rcrArgs_t{
    float *A;
    int LDA;
    float *B;
    int LDB;
    float *C;
    int LDC;
    int M;
    int N;
    int K;
};
