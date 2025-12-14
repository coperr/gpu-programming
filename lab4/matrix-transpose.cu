/**
  * Kernel 1: Implement a first kernel where each block (using BSXY x BSXY threads) transposes a BSXY x BSXY tile of A, and writes it into the corresponding location in At. Do without using shared memory.
  *
  * Kernel 2: In the second kernel, do the same, but using the shared memory. Each block should load a tile of BSXY x BSXY of A into the shared memory, then perform the transposition using this tile in the shared memory into At. Test the difference in speedup. Test the performance using shared memory without padding and with padding (to avoid shared memory bank conflicts).
  *
  * Kernel 3: In this kernel, perform the transpose in-place on the matrix A (do not use At). A block should be transpose two tiles simultenously to be able to do this.
  *
  */
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <ctime>
#include "cuda.h"
#include <cfloat>

#define BSXY 32

void transposeCPU(float *A, float *At, int N)
{
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      At[i * N + j] = A[j * N + i];
    }
  }
}

/* Kernel 1 */
__global__ void transposeNaive(float *A, float *At, int N)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < N && y < N) {
        At[x * N + y] = A[y * N + x];
    }
}

/* Kernel 2 */
__global__ void transposeShared(float *A, float *At, int N)
{

    __shared__ float tile[BSXY][BSXY + 1];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < N && y < N) {
        tile[threadIdx.y][threadIdx.x] = A[y * N + x];
    }
    __syncthreads();

    x = blockIdx.y * blockDim.x + threadIdx.x;
    y = blockIdx.x * blockDim.y + threadIdx.y;

    if (x < N && y < N) {
        At[y * N + x] = tile[threadIdx.x][threadIdx.y];
    }
}

/* Kernel 3 */
__global__ void transposeInPlace(float *A, int N)
{
    if (blockIdx.y < blockIdx.x) return;

    __shared__ float tile1[BSXY][BSXY + 1];
    __shared__ float tile2[BSXY][BSXY + 1];

    int bx = blockIdx.x * BSXY;
    int by = blockIdx.y * BSXY;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x1 = bx + tx;
    int y1 = by + ty;
    
    int x2 = by + tx;
    int y2 = bx + ty;

    if (x1 < N && y1 < N) tile1[ty][tx] = A[y1 * N + x1];
    
    bool isDiagonal = (blockIdx.x == blockIdx.y);
    if (!isDiagonal && x2 < N && y2 < N) {
        tile2[ty][tx] = A[y2 * N + x2];
    }
    __syncthreads();

    if (isDiagonal) {
        if (x1 < N && y1 < N) {
            A[y1 * N + x1] = tile1[tx][ty];
        }
    } else {
        if (x2 < N && y2 < N) {
            A[y2 * N + x2] = tile1[tx][ty];
        }
        if (x1 < N && y1 < N) {
            A[y1 * N + x1] = tile2[tx][ty];
        }
    }
}

bool checkResult(float *a, float *b, int N) {
    for(int i=0; i<N*N; i++) {
        if(a[i] != b[i]) return false;
    }
    return true;
}

int main()
{
  // Allocate A and At
  // A is an N * N matrix stored by rows, i.e. A(i, j) = A[i * N + j]
  // At is also stored by rows and is the transpose of A, i.e., At(i, j) = A(j, i)
  int N = 1024;
  float *A = (float *) malloc(N * N * sizeof(A[0]));
  float *At = (float *) malloc(N * N * sizeof(At[0]));
  float *At_ref = (float *) malloc(N * N * sizeof(float));

  srand(1234);
  for(int i=0; i<N*N; i++) A[i] = (float)(rand() % 100);
  transposeCPU(A, At_ref, N);
  
  // Allocate dA and dAt, and call the corresponding matrix transpose kernel
  // TODO / A FAIRE ...

  float *dA, *dAt;
  cudaMalloc(&dA, N * N * sizeof(float));
  cudaMalloc(&dAt, N * N * sizeof(float));

  cudaMemcpy(dA, A, N * N * sizeof(float), cudaMemcpyHostToDevice);

  dim3 block(BSXY, BSXY);
  dim3 grid((N + BSXY - 1) / BSXY, (N + BSXY - 1) / BSXY);

  cudaEvent_t start, stop;
  cudaEventCreate(&start); cudaEventCreate(&stop);
  float ms = 0;

  // Kernel 1
  cudaMemset(dAt, 0, N * N * sizeof(float));
  cudaEventRecord(start);
  transposeNaive<<<grid, block>>>(dA, dAt, N);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms, start, stop);

  cudaMemcpy(At, dAt, N * N * sizeof(float), cudaMemcpyDeviceToHost);

  printf("Naive Kernel Time: %.3f ms\n", ms);

  // Kernel 2
  cudaMemset(dAt, 0, N * N * sizeof(float)); 
  cudaEventRecord(start);
  transposeShared<<<grid, block>>>(dA, dAt, N);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms, start, stop);

  cudaMemcpy(At, dAt, N * N * sizeof(float), cudaMemcpyDeviceToHost);

  printf("Shared Kernel Time: %.3f ms\n", ms);

  // Kernel 3
  cudaMemcpy(dA, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
  
  cudaEventRecord(start);
  transposeInPlace<<<grid, block>>>(dA, N);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms, start, stop);

  cudaMemcpy(At, dA, N * N * sizeof(float), cudaMemcpyDeviceToHost);

  printf("In-Place Kernel Time: %.3f ms\n", ms);


  // Deallocate dA and dAt
  // TODO / A FAIRE ...

  cudaFree(dA);
  cudaFree(dAt);

  // Deallocate A and At
  free(A);
  free(At);
  free(At_ref);

  return 0;
}
