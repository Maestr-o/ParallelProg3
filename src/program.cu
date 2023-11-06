#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define CUDA_CHECK_MALLOC if (cudaStatus != cudaSuccess) {\
  fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));\
  return;\
}

#define CUDA_CHECK_MEMCPY if (cudaStatus != cudaSuccess) {\
  fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));\
  return;\
}

#define CUDA_CHECK_KERNEL if (cudaStatus != cudaSuccess) {\
  fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));\
  return;\
}

void calc(int n);
bool is_prime(int num);
int* generate_primes(int N, int* length);
__global__ void kernel(int* primes, int* size, int* res, int* n);

int main() {
  int n;
  printf("Enter N: ");
  if (scanf("%d", &n) != 1) {
    printf("Error\n");
    return 0;
  }
  
  clock_t start, end;
  start = clock();
  calc(n);
  end = clock();
  printf("Parallel time: %.3f s", ((double)(end - start)) / CLOCKS_PER_SEC);
  return 0;
}

void calc(int n) {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  cudaError_t cudaStatus;

  int size;
  int* primes = generate_primes(pow(10, 7), &size);
  int num_threads = prop.maxThreadsPerBlock;
  int num_blocks = (size + num_threads - 1) / num_threads;

  int* dev_primes;
  int* dev_size;
  int* dev_res;
  int* dev_n;
  int* res = (int*)calloc(size, sizeof(int));
  if (res == NULL) {
    printf("Error allocate memory\n");
    return;
  }

  cudaStatus = cudaMalloc((void**)&dev_n, sizeof(int));
  CUDA_CHECK_MALLOC
  cudaStatus = cudaMemcpy(dev_n, &n, sizeof(int), cudaMemcpyHostToDevice);
  CUDA_CHECK_MEMCPY
  
  cudaStatus = cudaMalloc((void**)&dev_size, sizeof(int));
  CUDA_CHECK_MALLOC
  cudaStatus = cudaMemcpy(dev_size, &size, sizeof(int), cudaMemcpyHostToDevice);
  CUDA_CHECK_MEMCPY

  cudaStatus = cudaMalloc((void**)&dev_primes, sizeof(int) * size);
  CUDA_CHECK_MALLOC
  cudaStatus = cudaMemcpy(dev_primes, primes, sizeof(int) * size,
                          cudaMemcpyHostToDevice);
  CUDA_CHECK_MEMCPY

  cudaStatus = cudaMalloc((void**)&dev_res, sizeof(int) * size);
  CUDA_CHECK_MALLOC

  kernel<<<num_blocks, num_threads>>>(dev_primes, dev_size, dev_res, dev_n);
  CUDA_CHECK_KERNEL

  cudaStatus = cudaMemcpy(res, dev_res, sizeof(int) * size,
                          cudaMemcpyDeviceToHost);
  CUDA_CHECK_MEMCPY

  for (int i = 0; i < size; i++) {
    if (res[i] > n) {
      printf(
          "Pair 1: %d, %d (mid: %d)\nPair 2: %d, %d (mid: %d)\nDiff: %d\n", primes[i],
             primes[i + 1], (primes[i + 1] + primes[i]) / 2, primes[i + 2],
             primes[i + 3], (primes[i + 3] + primes[i + 2]) / 2, res[i]);
      break;
    }
  }

  free(primes);
  cudaFree(dev_res);
  cudaFree(dev_primes);
  cudaFree(dev_size);
  cudaFree(dev_n);
}

__global__ void kernel(int* primes, int* size, int* res, int* n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid > *size - 3) return;
  int diff = (primes[tid + 3] + primes[tid + 2]) / 2 -
      (primes[tid + 1] + primes[tid]) / 2;
  if (primes[tid + 3] - primes[tid + 2] == 2 &&
      primes[tid + 1] - primes[tid] == 2 &&
      (diff > *n)) {
    res[tid] = diff;
  }
}

bool is_prime(int num) {
  if (num < 2) return false;
  if (num == 2) return true;
  if (num % 2 == 0) return false;
  for (int i = 3; i * i <= num; i += 2) {
    if (num % i == 0) return false;
  }
  return true;
}

int* generate_primes(int N, int* length) {
  int* primes = (int*)malloc(sizeof(int));
  *length = 0;

  for (int i = 2; i <= N; i++) {
    if (is_prime(i)) {
      (*length)++;
      primes = (int*)realloc(primes, sizeof(int) * (*length));
      primes[(*length) - 1] = i;
    }
  }

  return primes;
}
