#include <sys/time.h>
#include <stdio.h>

#define ERRLMT 1.e-5

#define LOG(format, ...) do{ \
    printf("INFO [%s %d]: %s => " format " \n", \
           __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__); \
    }while(0)

typedef struct timeval TIME_T;
#define MARK_TIME(t) gettimeofday(&t, NULL)
#define DIFF_TIME(start, end) ((end.tv_sec-start.tv_sec)*1e3+(end.tv_usec-start.tv_usec)*1e-3) // ms

void check_value(float* buf0, float* buf1, long len);
void rand_init(float* buf, long len);
void _local_gemm_rrr(const float* A, const int LDA, const float* B, const int LDB, float* C, const int LDC, int M, int N, int K);
void _local_gemm_rcr(const float* A, const int LDA, const float* B, const int LDB, float* C, const int LDC, int M, int N, int K);
void _local_trans(float* Mat, const int M, const int N);
