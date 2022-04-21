#include "myargs.h"
#include "dma_macros.h"
#include <simd.h>

#define NTHREAD      (64)
#define BLKSZ        (32)
#define TILEK        (2048 / BLKSZ)

static inline unsigned long rpcc() {
    unsigned long time;
    asm volatile ("rcsr %0,4" :  "=r" (time));
    return time;
}

void gemm_rcr_tiled_s(struct rcrArgs_t *myArgs_loc)
{
    struct rcrArgs_t myArgs;
    
    dma_init();
    pe_get(myArgs_loc, &myArgs, sizeof(struct rcrArgs_t));
    dma_syn();
    
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
        
        floatv4 Asv4[BLKSZ * TILEK / 4] = {0};
		floatv4 Bsv4[BLKSZ * TILEK / 4] = {0};
		floatv4 Csv4[BLKSZ * BLKSZ] = {0};
        float Cs[BLKSZ * BLKSZ] = {0};
        float *A_lst = myArgs.A + yst * myArgs.K;
        float *B_lst = myArgs.B + xst * myArgs.K;
        for (int kst = 0; kst < myArgs.K; kst += TILEK) {
            int subk = (TILEK > myArgs.K - kst) ? myArgs.K - kst : TILEK;
            float* A_loc = A_lst + kst;
            float* B_loc = B_lst + kst;
            for (int i = 0; i < yld; i++) {
                pe_get(A_loc, (float*)Asv4 + i * TILEK, subk * sizeof(float));
                dma_syn();
                A_loc += myArgs.K;
            }
            for (int j = 0; j < xld; j++) {
                pe_get(B_loc, (float*)Bsv4 + j * TILEK, subk * sizeof(float));
                dma_syn();
                B_loc += myArgs.K;
            }
            for (int i = 0; i < yld; i++) {
				int offsetA = i * TILEK / 4;
				int offsetC = i * BLKSZ;
                for (int j = 0; j < xld; j++) {
					int offsetB = j * TILEK / 4;
                    for (int k = 0; k < (subk + 3) / 4; k++)
                        Csv4[offsetC + j] += Asv4[offsetA + k] * Bsv4[offsetB + k];
                }
			}
        }
		
		for (int i = 0; i < BLKSZ; i++)
			for (int j = 0; j < BLKSZ; j++) {
				int offsetC = i * BLKSZ + j;
				float Ctmp[4];
				simd_store(Csv4[offsetC], Ctmp);
				Cs[offsetC] = Ctmp[0] + Ctmp[1] + Ctmp[2] + Ctmp[3];
			}
        
        float *C_loc = myArgs.C + yst * myArgs.N + xst;
        for (int i = 0; i < yld; i++) {
            pe_put(C_loc, Cs + i * BLKSZ, xld * sizeof(float));
            dma_syn();
            C_loc += myArgs.N;
        }
        
        blk_id += NTHREAD;
    }
}
