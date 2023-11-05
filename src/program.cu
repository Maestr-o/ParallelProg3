﻿#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

void calc(long n, int num_blocks, int num_threads);
bool is_prime_between_pairs(long x, long y, int num_blocks, int num_threads);
bool check(bool *flags, long current_prime);
__global__ void is_prime(long *num, bool *flags);

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

void calc(long n, int num_blocks, int num_threads) {
  long last_pair_x = 2, last_pair_y = 3;
  long prev_prime = 5, current_prime = 7;
  long middle = 6, last_middle = 2;

  while (1) {
    long *dev_current_prime;
    bool *flags = (bool *)calloc(current_prime, sizeof(bool));
    bool *dev_flags;
    cudaMalloc((void **)&dev_current_prime, sizeof(long));
    cudaMemcpy(dev_current_prime, &current_prime, sizeof(long),
               cudaMemcpyHostToDevice);
    cudaMalloc((void **)&dev_flags, sizeof(flags));
    is_prime<<<num_blocks, num_threads>>>(dev_current_prime, dev_flags);
    cudaMemcpy(flags, dev_flags, sizeof(bool) * current_prime, cudaMemcpyDeviceToHost);
    printf("%d: ", current_prime);
    for (int i = 0; i < current_prime; i++) {
      printf("%d-", flags[i]);
    }
    printf("\n");
    char c;
    scanf("%c", &c);

    if (check(flags, current_prime)) {
      if (current_prime - prev_prime == 2) {
        if (prev_prime != last_pair_y) {
          middle = (current_prime + prev_prime) / 2;
          if (middle - last_middle > n &&
              !is_prime_between_pairs(last_pair_y, prev_prime, num_blocks, num_threads)) {
            printf(
                "Pair 1: %ld, %ld (mid: %ld)\nPair 2: %ld, %ld (mid: %ld)\n%ld "
                "> %ld\n",
                last_pair_x, last_pair_y, last_middle, prev_prime,
                current_prime, middle, middle - last_middle, n);
            break;
          }
          last_middle = middle;
          last_pair_x = prev_prime;
          last_pair_y = current_prime;
        }
      }
      prev_prime = current_prime;
    }
    current_prime++;

    cudaFree(dev_current_prime);
    cudaFree(dev_flags);
    free(flags);
  }
}

bool check(bool *flags, long current_prime) {
  for (int i = 2; i < current_prime; i++) {
    if (flags[i]) return false;
  }
  return true;
}

__global__ void is_prime(long *num, bool *flags) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < 2) return;
  if (*num % tid != 0) return;
  flags[tid] = true;
}

bool is_prime_between_pairs(long x, long y, int num_blocks, int num_threads) {
  for (long current_prime = x + 1; current_prime < y; current_prime++) {
    long *dev_current_prime;
    bool *flags = (bool *)calloc(current_prime, sizeof(bool));
    bool *dev_flags;
    cudaMalloc((void **)&dev_current_prime, sizeof(long));
    cudaMemcpy(dev_current_prime, &current_prime, sizeof(long),
               cudaMemcpyHostToDevice);
    cudaMalloc((void **)&dev_flags, sizeof(flags));
    is_prime<<<num_blocks, num_threads>>>(dev_current_prime, dev_flags);
    cudaMemcpy(flags, dev_flags, sizeof(bool) * current_prime,
               cudaMemcpyDeviceToHost);
    if (check(flags, current_prime)) return true;
  }
  return false;
}
