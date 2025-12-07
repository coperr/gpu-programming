/**
  * In this exercise, we will implement GPU kernels for computing the average of 9 points on a 2D array.
  * Dans cet exercice, nous implantons un kernel GPU pour un calcul de moyenne de 9 points sur un tableau 2D.
  *
  * Kernel 1: Use 1D grid of blocks (only blockIdx.x), no additional threads (1 thread per block)
  * Kernel 1: Utiliser grille 1D de blocs (seulement blockIdx.x), pas de threads (1 thread par bloc)
  *
  * Kernel 2: Use 2D grid of blocks (blockIdx.x/.y), no additional threads (1 thread per block)
  * Kernel 2: Utiliser grille 2D de blocs (blockIdx.x/.y), pas de threads (1 thread par bloc)
  *
  * Kernel 3: Use 2D grid of blocks and 2D threads (BSXY x BSXY), each thread computing 1 element of Aavg
  * Kernel 3: Utiliser grille 2D de blocs, threads de 2D (BSXY x BSXY), chaque thread calcule 1 element de Aavg
  *
  * Kernel 4: Use 2D grid of blocks and 2D threads, each thread computing 1 element of Aavg, use shared memory. Each block should load BSXY x BSXY elements of A, then compute (BSXY - 2) x (BSXY - 2) elements of Aavg. Borders of tiles loaded by different blocks must overlap to be able to compute all elements of Aavg.
  * Kernel 4: Utiliser grille 2D de blocs, threads de 2D, chaque thread calcule 1 element de Aavg, avec shared memory. Chaque bloc doit lire BSXY x BSXY elements de A, puis calculer avec ceci (BSXY - 2) x (BSXY - 2) elements de Aavg. Les bords des tuiles chargees par de differents blocs doivent chevaucher afin de pouvoir calculer tous les elements de Aavg.
  *
  * Kernel 5: Use 2D grid of blocks and 2D threads, use shared memory, each thread computes KxK elements of Aavg
  * Kernel 5: Utiliser grille 2D de blocs, threads de 2D, avec shared memory et KxK ops par thread
  *
  * For all kernels: Make necessary memory allocations/deallocations and memcpy in the main.
  * Pour tous les kernels: Effectuer les allocations/desallocations et memcpy necessaires dans le main.
  */

#include <iostream>
#include <cstdio>
#include "cuda.h"
#include "omp.h"

#define N 1024
#define K 4
#define BSXY 32

// The matrix is stored by columns, that is A(i, j) = A[i + j * N]. The average should be computed on Aavg array.
// La matrice A est stockee par colonnes, a savoir A(i, j) = A[i + j * N]
float *A;
float *Aavg;

float *dA, *dAavg;

// Reference CPU implementation
// Code de reference pour le CPU
void ninePointAverageCPU(const float *A, float *Aavg)
{
  for (int i = 1; i < N - 1; i++) {
    for (int j = 1; j < N - 1; j++) {
      Aavg[i + j * N] = (A[i - 1 + (j - 1) * N] + A[i - 1 + (j) * N] + A[i - 1 + (j + 1) * N] +
          A[i + (j - 1) * N] + A[i + (j) * N] + A[i + (j + 1) * N] +
          A[i + 1 + (j - 1) * N] + A[i + 1 + (j) * N] + A[i + 1 + (j + 1) * N]) * (1.0 / 9.0);
    }
  }
}

// Kernel 1
__global__ void avgKernel1(float *dA, float *dAavg, int n) {
    int idx = blockIdx.x;
    int i = idx % n;
    int j = idx / n;

    if (i > 0 && i < n - 1 && j > 0 && j < n - 1) {
        dAavg[i + j * n] = (dA[i - 1 + (j - 1) * n] + dA[i - 1 + j * n] + dA[i - 1 + (j + 1) * n] +
                            dA[i     + (j - 1) * n] + dA[i     + j * n] + dA[i     + (j + 1) * n] +
                            dA[i + 1 + (j - 1) * n] + dA[i + 1 + j * n] + dA[i + 1 + (j + 1) * n]) / 9.0f;
    }
}

// Kernel 2
__global__ void avgKernel2(float *dA, float *dAavg, int n) {
    int i = blockIdx.x;
    int j = blockIdx.y;

    if (i > 0 && i < n - 1 && j > 0 && j < n - 1) {
        dAavg[i + j * n] = (dA[i - 1 + (j - 1) * n] + dA[i - 1 + j * n] + dA[i - 1 + (j + 1) * n] +
                            dA[i     + (j - 1) * n] + dA[i     + j * n] + dA[i     + (j + 1) * n] +
                            dA[i + 1 + (j - 1) * n] + dA[i + 1 + j * n] + dA[i + 1 + (j + 1) * n]) / 9.0f;
    }
}

// Kernel 3
__global__ void avgKernel3(float *dA, float *dAavg, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i > 0 && i < n - 1 && j > 0 && j < n - 1) {
        dAavg[i + j * n] = (dA[i - 1 + (j - 1) * n] + dA[i - 1 + j * n] + dA[i - 1 + (j + 1) * n] +
                            dA[i     + (j - 1) * n] + dA[i     + j * n] + dA[i     + (j + 1) * n] +
                            dA[i + 1 + (j - 1) * n] + dA[i + 1 + j * n] + dA[i + 1 + (j + 1) * n]) / 9.0f;
    }
}

// Kernel 4
__global__ void avgKernel4(float *dA, float *dAavg, int n) {
    __shared__ float sA[BSXY][BSXY];

    int outDim = BSXY - 2;
    int startI = blockIdx.x * outDim;
    int startJ = blockIdx.y * outDim;

    int ti = threadIdx.x;
    int tj = threadIdx.y;

    int gi = startI + ti;
    int gj = startJ + tj;

    if (gi < n && gj < n) {
        sA[ti][tj] = dA[gi + gj * n];
    } else {
        sA[ti][tj] = 0.0f; 
    }
    
    __syncthreads();

    if (ti > 0 && ti < BSXY - 1 && tj > 0 && tj < BSXY - 1) {
        if (gi > 0 && gi < n - 1 && gj > 0 && gj < n - 1) {
             float sum = sA[ti - 1][tj - 1] + sA[ti - 1][tj] + sA[ti - 1][tj + 1] +
                         sA[ti][tj - 1]     + sA[ti][tj]     + sA[ti][tj + 1]     +
                         sA[ti + 1][tj - 1] + sA[ti + 1][tj] + sA[ti + 1][tj + 1];
             dAavg[gi + gj * n] = sum / 9.0f;
        }
    }
}

// Kernel 5
__global__ void avgKernel5(float *dA, float *dAavg, int n) {
    const int tileW = BSXY; 
    const int smW = tileW + 2; 

    __shared__ float sA[34][34]; 

    int startI = blockIdx.x * tileW;
    int startJ = blockIdx.y * tileW;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int tid = tx + ty * blockDim.x;
    int numThreads = blockDim.x * blockDim.y;
    
    for (int k = tid; k < smW * smW; k += numThreads) {
        int loc_i = k % smW;
        int loc_j = k / smW;
        
        int glob_i = startI + loc_i - 1;
        int glob_j = startJ + loc_j - 1;
        
        if (glob_i >= 0 && glob_i < n && glob_j >= 0 && glob_j < n) {
            sA[loc_i][loc_j] = dA[glob_i + glob_j * n];
        } else {
            sA[loc_i][loc_j] = 0.0f;
        }
    }

    __syncthreads();

    
    for (int v = 0; v < K; ++v) {
        for (int u = 0; u < K; ++u) {

            int out_i_local = tx * K + u;
            int out_j_local = ty * K + v;
            
            int sm_i = out_i_local + 1;
            int sm_j = out_j_local + 1;
            
            int gi = startI + out_i_local;
            int gj = startJ + out_j_local;

            if (gi > 0 && gi < n - 1 && gj > 0 && gj < n - 1 && out_i_local < tileW && out_j_local < tileW) {
                float sum = sA[sm_i - 1][sm_j - 1] + sA[sm_i - 1][sm_j] + sA[sm_i - 1][sm_j + 1] +
                            sA[sm_i][sm_j - 1]     + sA[sm_i][sm_j]     + sA[sm_i][sm_j + 1]     +
                            sA[sm_i + 1][sm_j - 1] + sA[sm_i + 1][sm_j] + sA[sm_i + 1][sm_j + 1];
                dAavg[gi + gj * n] = sum / 9.0f;
            }
        }
    }
}


int main()
{
  A = (float *) malloc (N * N * sizeof(float));
  Aavg = (float *) malloc (N * N * sizeof(float));

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i + j * N] = (float)i * (float)j;
    }
  }

  cudaMalloc(&dA, N * N * sizeof(float));
  cudaMalloc(&dAavg, N * N * sizeof(float));
  cudaMemcpy(dA, A, N * N * sizeof(float), cudaMemcpyHostToDevice);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float milliseconds = 0;

  // Kernel 1
  cudaMemset(dAavg, 0, N * N * sizeof(float));
  cudaEventRecord(start);
  avgKernel1<<<N * N, 1>>>(dA, dAavg, N);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Kernel 1 (1D Blocks): " << milliseconds << " ms" << std::endl;

  // Kernel 2
  cudaMemset(dAavg, 0, N * N * sizeof(float));
  dim3 grid2(N, N);
  cudaEventRecord(start);
  avgKernel2<<<grid2, 1>>>(dA, dAavg, N);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Kernel 2 (2D Blocks): " << milliseconds << " ms" << std::endl;

  // Kernel 3
  cudaMemset(dAavg, 0, N * N * sizeof(float));
  dim3 block3(BSXY, BSXY);
  dim3 grid3((N + BSXY - 1) / BSXY, (N + BSXY - 1) / BSXY);
  cudaEventRecord(start);
  avgKernel3<<<grid3, block3>>>(dA, dAavg, N);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Kernel 3 (2D Blocks/Threads): " << milliseconds << " ms" << std::endl;

  // Kernel 4
  cudaMemset(dAavg, 0, N * N * sizeof(float));

  int outDim4 = BSXY - 2;
  dim3 grid4((N + outDim4 - 1) / outDim4, (N + outDim4 - 1) / outDim4);
  dim3 block4(BSXY, BSXY);
  cudaEventRecord(start);
  avgKernel4<<<grid4, block4>>>(dA, dAavg, N);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Kernel 4 (Shared Mem): " << milliseconds << " ms" << std::endl;

  // Kernel 5
  cudaMemset(dAavg, 0, N * N * sizeof(float));

  dim3 grid5((N + BSXY - 1) / BSXY, (N + BSXY - 1) / BSXY);
  dim3 block5(BSXY / K, BSXY / K); 
  cudaEventRecord(start);
  avgKernel5<<<grid5, block5>>>(dA, dAavg, N);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Kernel 5 (Shared Mem, KxK): " << milliseconds << " ms" << std::endl;

  cudaMemcpy(Aavg, dAavg, N * N * sizeof(float), cudaMemcpyDeviceToHost);
  
  float *ref = (float*)malloc(N * N * sizeof(float));
  ninePointAverageCPU(A, ref);

  bool correct = true;
  for(int i=1; i<N-1; ++i) {
      for(int j=1; j<N-1; ++j) {
          float diff = std::abs(Aavg[i+j*N] - ref[i+j*N]);
          if (diff > 1e-4) {
              correct = false;
          }
      }
  }
  if(correct) std::cout << "Verification Passed" << std::endl;
  else std::cout << "Verification Failed" << std::endl;

  free(ref);
  free(A);
  free(Aavg);
  cudaFree(dA);
  cudaFree(dAavg);

  return 0;
}
