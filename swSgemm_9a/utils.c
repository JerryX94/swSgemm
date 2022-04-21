#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "utils.h"

void check_value(float* buf0, float* buf1, long len) {
	for (int i = 0; i < len; i++)
		if (buf0[i] != 0 && buf1[i] != 0)
			if (fabs((buf0[i] - buf1[i]) / (buf0[i] + buf1[i])) > ERRLMT) {
				LOG("Check Value Failed: index = %ld, %f != %f\n", i, buf0[i], buf1[i]);
				//printf("Check Value Failed: index = %ld, %f != %f\n", i, buf0[i], buf1[i]);
				return;
			}
	LOG("Check Value Passed\n");
	//printf("Check Value Passed\n");
}

void rand_init(float* buf, long len) {
	for (long i = 0; i < len; i++) {
		float v = rand();
		buf[i] = v / RAND_MAX;
	}
}

void _local_gemm_rrr(const float* A, const int LDA, const float* B, const int LDB, float* C, const int LDC, int M, int N, int K) {
	for (int i = 0; i < M; i++)
		for (int j = 0; j < N; j++)
			for (int k = 0; k < K; k++)
				C[i * LDC + j] += A[i * LDA + k] * B[k * LDB + j];
}

void _local_gemm_rcr(const float* A, const int LDA, const float* B, const int LDB, float* C, const int LDC, int M, int N, int K) {
	for (int i = 0; i < M; i++)
		for (int j = 0; j < N; j++)
			for (int k = 0; k < K; k++)
				C[i * LDC + j] += A[i * LDA + k] * B[k + j * LDB];
}

void _local_trans(float* Mat, const int M, const int N) {
	float* T = (float*)malloc(M * N * sizeof(float));
	for (int i = 0; i < M; i++)
		for (int j = 0; j < N; j++)
			T[j * M + i] = Mat[i * N + j];
	memcpy(Mat, T, M * N * sizeof(float));

	free(T);
}
