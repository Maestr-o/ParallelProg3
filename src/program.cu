#include <stdio.h>
#include <stdlib.h>

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

void calc(int n, int num_blocks, int num_threads);
int* load_primes_from_file(const char* filename, int* size);
__global__ void kernel(int* primes, int* size, int* res, int* n);

int main() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  int n;
  printf("Enter N: ");
  if (scanf("%d", &n) != 1) {
    printf("Error\n");
    return 0;
  }

  int num_threads = prop.maxThreadsPerBlock;
  int num_blocks = (n + num_threads - 1) / num_threads;

  calc(n, num_blocks, num_threads);
  return 0;
}

void calc(int n, int num_blocks, int num_threads) {
  cudaError_t cudaStatus;
  int size;
  int* primes = load_primes_from_file("primes.txt", &size);
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

  cudaFree(dev_res);
  cudaFree(dev_primes);
  cudaFree(dev_size);
  cudaFree(dev_n);
}

int* load_primes_from_file(const char* filename, int* size) {
  FILE* file = fopen(filename, "r");
  if (file == NULL) {
    perror("Error opening file");
    return NULL;
  }

  int capacity = 10;
  int* primes = (int*)malloc(sizeof(int) * capacity);
  if (primes == NULL) {
    perror("Memory allocation error");
    fclose(file);
    return NULL;
  }

  int count = 0;
  int num;

  while (fscanf(file, "%d", &num) == 1) {
    if (count == capacity) {
      capacity *= 2;
      int* new_primes = (int*)realloc(primes, sizeof(int) * capacity);
      if (new_primes == NULL) {
        perror("Memory reallocation error");
        free(primes);
        fclose(file);
        return NULL;
      }
      primes = new_primes;
    }
    primes[count++] = num;
  }

  fclose(file);

  int* resized_primes = (int*)realloc(primes, sizeof(int) * count);
  if (resized_primes == NULL) {
    perror("Memory reallocation error");
    free(primes);
    return NULL;
  }

  *size = count;
  return resized_primes;
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
