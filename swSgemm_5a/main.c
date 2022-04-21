#include <stdio.h>
#include <stdlib.h>
#include "utils.h"

#define MSIZE 1280
#define NSIZE 1920
#define KSIZE 2560

void call_athread_init();
void call_athread_halt();
void slave_gemm_rcr(const float* A, const int LDA, const float* B, const int LDB, float* C, const int LDC, int M, int N, int K);
void slave_gemm_rrr(const float* A, const int LDA, const float* B, const int LDB, float* C, const int LDC, int M, int N, int K);

int main() {
	TIME_T start, end;
	float* A, * B, * C, * C0;

	long asize = MSIZE * KSIZE;
	long bsize = KSIZE * NSIZE;
	long csize = MSIZE * NSIZE;

	// Data_alloc
	MARK_TIME(start);
	A = (float*)malloc(asize * sizeof(float));
	B = (float*)malloc(bsize * sizeof(float));
	C = (float*)malloc(csize * sizeof(float));
	C0 = (float*)calloc(csize, sizeof(float));

	MARK_TIME(end);
	LOG("Init Mats Time : %.3f ms", 1.0 * DIFF_TIME(start, end));

	// Rand_init
	rand_init(A, asize);
	rand_init(B, bsize);
///*
	// Benchmark
	MARK_TIME(start);
	//_local_gemm_rrr(A, KSIZE, B, NSIZE, C0, NSIZE, MSIZE, NSIZE, KSIZE);
	_local_gemm_rcr(A, KSIZE, B, KSIZE, C0, NSIZE, MSIZE, NSIZE, KSIZE);
	MARK_TIME(end);
	LOG("Benchmark Sgemm Time : %.3f ms\n", 1.0 * DIFF_TIME(start, end));
//*/
	// Warm_up
	call_athread_init();
	LOG("Warm Up First ...");
	slave_gemm_rcr(A, KSIZE, B, NSIZE, C, NSIZE, MSIZE, NSIZE, KSIZE);
	LOG("Warm Up Done\n");

	// My_Sgemm
	MARK_TIME(start);
	//slave_gemm_rrr(A, KSIZE, B, NSIZE, C, NSIZE, MSIZE, NSIZE, KSIZE);
	slave_gemm_rcr(A, KSIZE, B, KSIZE, C, NSIZE, MSIZE, NSIZE, KSIZE);
	MARK_TIME(end);
	LOG("My Sgemm Time : %.3f ms", 1.0 * DIFF_TIME(start, end));
	check_value(C0, C, csize);

	// Finalize
	call_athread_halt();
	free(A);
	free(B);
	free(C);
	free(C0);

	return 0;
}
