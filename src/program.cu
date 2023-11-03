#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>

bool is_prime_host(long num);
__device__ bool is_prime_device(long num);
__global__ void check_primes_kernel(long *d_results, long start, long end);
void calc(long n);
bool is_prime_between_pairs(long x, long y);
void run();

int main() {
    cudaSetDevice(0);
    run();
    cudaDeviceReset();
    return 0;
}

void run() {
    long n;
    printf("Enter N: ");
    scanf("%ld", &n);
    double start, end;
    start = omp_get_wtime();
    calc(n);
    end = omp_get_wtime();
    printf("Parallel time = %.3lf sec\n", (end - start));
}

// Функция проверки простоты для хоста
bool is_prime_host(long num) {
    if (num < 2)
        return false;
    long i;
    for (i = 2; i * 2 <= num; i++) {
        if (num % i == 0)
            return false;
    }
    return true;
}

// Функция проверки простоты для устройства
__device__ bool is_prime_device(long num) {
    if (num < 2)
        return false;
    long i;
    for (i = 2; i * 2 <= num; i++) {
        if (num % i == 0)
            return false;
    }
    return true;
}

__global__ void check_primes_kernel(long *d_results, long start, long end) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < end - start && is_prime_device(tid + start)) {
        d_results[tid] = tid + start;
    } else {
        d_results[tid] = -1;
    }
}

void calc(long n) {
    const int block_size = 256;
    long last_pair_x = 2, last_pair_y = 3;
    long prev_prime = 5, current_prime = 7;
    long middle = 6, last_middle = 2;

    while (1) {
        if (is_prime_host(current_prime)) {
            if (current_prime - prev_prime == 2) {
                if (prev_prime != last_pair_y) {
                    middle = (current_prime + prev_prime) / 2;
                    if (middle - last_middle > n && !is_prime_between_pairs(last_pair_y, prev_prime)) {
                        printf("Pair 1: %ld, %ld (mid: %ld)\nPair 2: %ld, %ld (mid: %ld)\n%ld > %ld\n",
                               last_pair_x, last_pair_y, last_middle, prev_prime, current_prime, middle,
                               middle - last_middle, n);
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
    }

    // Выделяем память на GPU и копируем результаты проверки простоты
    long *d_results;
    cudaMalloc(&d_results, (current_prime - 2) * sizeof(long));

    int num_blocks = (current_prime - 2 + block_size - 1) / block_size;
    check_primes_kernel<<<num_blocks, block_size>>>(d_results, 2, current_prime);

    // Копируем результаты обратно на CPU
    long *h_results = (long *)malloc((current_prime - 2) * sizeof(long));
    cudaMemcpy(h_results, d_results, (current_prime - 2) * sizeof(long), cudaMemcpyDeviceToHost);

    // Освобождаем память на GPU и CPU
    cudaFree(d_results);
    free(h_results);
}

bool is_prime_between_pairs(long x, long y) {
    for (int i = x + 1; i < y; i++) {
        if (is_prime_host(i)) // Используем функцию для хоста
            return true;
    }
    return false;
}