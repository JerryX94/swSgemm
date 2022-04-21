#include "myargs.h"
#include <slave.h>
#include <simd.h>

#define NTHREAD      (64)
#define BLKSZ        (32)
#define TILEK        (2048 / BLKSZ)

#define NSIMD        (4)
typedef floatv4 floatv;

void gemm_rcr_tiled_s(struct rcrArgs_t *myArgs_loc)
{
    struct rcrArgs_t myArgs;
    volatile int reply;
    
    reply = 0;
    athread_get(PE_MODE, myArgs_loc, &myArgs, sizeof(struct rcrArgs_t), &reply, 0, 0, 0);
    while (reply != 1);
    
    const int global_id  = athread_get_id(-1);
    const int nColBlk = (myArgs.N + BLKSZ - 1) / BLKSZ;
    const int nRowBlk = (myArgs.M + BLKSZ - 1) / BLKSZ;
    const int nBlk    = nColBlk * nRowBlk;
    
    int blk_id = global_id;
    while (blk_id < nBlk) {
        int blk_cid = blk_id % nColBlk;
        int blk_rid = blk_id / nColBlk;
        int xst = blk_cid * BLKSZ;
        int xed = (xst + BLKSZ > myArgs.N) ? myArgs.N : xst + BLKSZ;
        int xld = xed - xst;
        int yst = blk_rid * BLKSZ;
        int yed = (yst + BLKSZ > myArgs.M) ? myArgs.M : yst + BLKSZ;
        int yld = yed - yst;
        
        floatv Asv[BLKSZ * TILEK / NSIMD] = {0};
        floatv Bsv[BLKSZ * TILEK / NSIMD] = {0};
        floatv Csv[BLKSZ * BLKSZ] = {0};
        float Cs[BLKSZ * BLKSZ] = {0};
        float *A_lst = myArgs.A + yst * myArgs.K;
        float *B_lst = myArgs.B + xst * myArgs.K;
        for (int kst = 0; kst < myArgs.K; kst += TILEK) {
            int subk = (TILEK > myArgs.K - kst) ? myArgs.K - kst : TILEK;
            int subkv = (subk + NSIMD - 1) / NSIMD;
            float* A_loc = A_lst + kst;
            float* B_loc = B_lst + kst;
            for (int i = 0; i < yld; i++) {
                reply = 0;
                athread_get(PE_MODE, A_loc, (float*)Asv + i * TILEK, subk * sizeof(float), &reply, 0, 0, 0);
                while (reply != 1);
                A_loc += myArgs.K;
            }
            for (int j = 0; j < xld; j++) {
                reply = 0;
                athread_get(PE_MODE, B_loc, (float*)Bsv + j * TILEK, subk * sizeof(float), &reply, 0, 0, 0);
                while (reply != 1);
                B_loc += myArgs.K;
            }
            for (int i = 0; i < yld; i++) {
                int offsetA = i * TILEK / NSIMD;
                int offsetC = i * BLKSZ;
                for (int j = 0; j < xld; j++) {
                    int offsetB = j * TILEK / NSIMD;
                    for (int k = 0; k < subkv; k++)
                        Csv[offsetC + j] += Asv[offsetA + k] * Bsv[offsetB + k];
                }
            }
        }
        
        for (int i = 0; i < BLKSZ; i++)
            for (int j = 0; j < BLKSZ; j++) {
                int offsetC = i * BLKSZ + j;
                float Ctmp[NSIMD];
                simd_store(Csv[offsetC], Ctmp);
                Cs[offsetC] = 0;
                for (int k = 0; k < NSIMD; k += 4)
                    Cs[offsetC] += Ctmp[k] + Ctmp[k + 1] + Ctmp[k + 2] + Ctmp[k + 3];
            }
        
        float *C_loc = myArgs.C + yst * myArgs.N + xst;
        for (int i = 0; i < yld; i++) {
            reply = 0;
            athread_put(PE_MODE, Cs + i * BLKSZ, C_loc, xld * sizeof(float), &reply, 0, 0);
            while (reply != 1);
            C_loc += myArgs.N;
        }
        
        blk_id += NTHREAD;
    }
}
