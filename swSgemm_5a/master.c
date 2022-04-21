#include <athread.h>
#include "myargs.h"

//declare slave parallel method
extern void SLAVE_FUN(gemm_rcr_tiled_s)();

void slave_gemm_rcr(const float* A, const int LDA, const float* B, const int LDB, \
                    float* C, const int LDC, int M, int N, int K)
{
    struct rcrArgs_t myArgs;
    myArgs.A   = A;
    myArgs.LDA = LDA;
    myArgs.B   = B;
    myArgs.LDB = LDB;
    myArgs.C   = C;
    myArgs.LDC = LDC;
    myArgs.M   = M;
    myArgs.N   = N;
    myArgs.K   = K;
    athread_spawn(gemm_rcr_tiled_s, &myArgs);
    athread_join();
}

void slave_gemm_rrr(const float* A, const int LDA, const float* B, const int LDB, \
                    float* C, const int LDC, int M, int N, int K)
{
	float BT[N * K];
    for(int j = 0; j < N; j ++)
        for(int k = 0; k < K; k ++)
            BT[k + j * K] = B[k * N + j];
    struct rcrArgs_t myArgs;
    myArgs.A   = A;
    myArgs.LDA = LDA;
    myArgs.B   = BT;
    myArgs.LDB = K;
    myArgs.C   = C;
    myArgs.LDC = LDC;
    myArgs.M   = M;
    myArgs.N   = N;
    myArgs.K   = K;
    athread_spawn(gemm_rcr_tiled_s, &myArgs);
    athread_join();
}

void call_athread_init() {
	athread_init();
}

void call_athread_halt() {
	athread_halt();
}
