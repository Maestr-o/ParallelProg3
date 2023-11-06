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

void calc(long n_primes, long n);
bool is_prime(long num);
long* generate_primes(long N, long* length);
__global__ void kernel(long* primes, long* size, long* res, long* n);

int main() {
  long n, n_primes;
  printf("Enter N: ");
  if (scanf("%d", &n) != 1) {
    printf("Error\n");
    return 0;
  }
  printf("Enter the maximum number of primes in array: ");
  if (scanf("%d", &n_primes) != 1) {
    printf("Error\n");
    return 0;
  }
  
  clock_t start, end;
  start = clock();
  calc(n_primes, n);
  end = clock();
  printf("Time: %.3f s", ((double)(end - start)) / CLOCKS_PER_SEC);
  return 0;
}

void calc(long n_primes, long n) {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  cudaError_t cudaStatus;

  long size;
  long* primes = generate_primes(n_primes, &size);
  long num_threads = prop.maxThreadsPerBlock;
  long num_blocks = (size + num_threads - 1) / num_threads;

  long* dev_primes;
  long* dev_size;
  long* dev_res;
  long* dev_n;
  long* res = (long*)calloc(size, sizeof(long));
  if (res == NULL) {
    printf("Error allocate memory\n");
    return;
  }

  cudaStatus = cudaMalloc((void**)&dev_n, sizeof(long));
  CUDA_CHECK_MALLOC
  cudaStatus = cudaMemcpy(dev_n, &n, sizeof(long), cudaMemcpyHostToDevice);
  CUDA_CHECK_MEMCPY
  
  cudaStatus = cudaMalloc((void**)&dev_size, sizeof(long));
  CUDA_CHECK_MALLOC
  cudaStatus = cudaMemcpy(dev_size, &size, sizeof(long), cudaMemcpyHostToDevice);
  CUDA_CHECK_MEMCPY

  cudaStatus = cudaMalloc((void**)&dev_primes, sizeof(long) * size);
  CUDA_CHECK_MALLOC
  cudaStatus = cudaMemcpy(dev_primes, primes, sizeof(long) * size,
                          cudaMemcpyHostToDevice);
  CUDA_CHECK_MEMCPY

  cudaStatus = cudaMalloc((void**)&dev_res, sizeof(long) * size);
  CUDA_CHECK_MALLOC

  kernel<<<num_blocks, num_threads>>>(dev_primes, dev_size, dev_res, dev_n);
  CUDA_CHECK_KERNEL

  cudaStatus = cudaMemcpy(res, dev_res, sizeof(long) * size,
                          cudaMemcpyDeviceToHost);
  CUDA_CHECK_MEMCPY

  bool is_answer_found = false;
  for (long i = 0; i < size; i++) {
    if (res[i] > n) {
      printf(
          "Pair 1: %d, %d (mid: %d)\nPair 2: %d, %d (mid: %d)\nDiff: %d\n", primes[i],
             primes[i + 1], (primes[i + 1] + primes[i]) / 2, primes[i + 2],
             primes[i + 3], (primes[i + 3] + primes[i + 2]) / 2, res[i]);
      is_answer_found = true;
      break;
    }
  }
  if (!is_answer_found)
    printf("Answer wasn't found. Please increase the maximum number of primes\n");

  free(primes);
  cudaFree(dev_res);
  cudaFree(dev_primes);
  cudaFree(dev_size);
  cudaFree(dev_n);
}

__global__ void kernel(long* primes, long* size, long* res, long* n) {
  long tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid > *size - 3) return;
  long diff = (primes[tid + 3] + primes[tid + 2]) / 2 -
      (primes[tid + 1] + primes[tid]) / 2;
  if (primes[tid + 3] - primes[tid + 2] == 2 &&
      primes[tid + 1] - primes[tid] == 2 &&
      (diff > *n)) {
    res[tid] = diff;
  }
}

bool is_prime(long num) {
  if (num < 2) return false;
  if (num == 2) return true;
  if (num % 2 == 0) return false;
  for (long i = 3; i * i <= num; i += 2) {
    if (num % i == 0) return false;
  }
  return true;
}

long* generate_primes(long N, long* length) {
  long* primes = (long*)malloc(sizeof(long));
  *length = 0;

  for (long i = 2; i <= N; i++) {
    if (is_prime(i)) {
      (*length)++;
      primes = (long*)realloc(primes, sizeof(long) * (*length));
      primes[(*length) - 1] = i;
    }
  }

  return primes;
}
