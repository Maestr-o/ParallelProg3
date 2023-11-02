#include <stdio.h>

__global__ void calc_cuda(long n, long *result) {
    long last_pair_x = 2, last_pair_y = 3;
    long prev_prime = threadIdx.x * blockDim.x * gridDim.x + 5;
    long current_prime = threadIdx.x * blockDim.x * gridDim.x + 7;
    long middle = 6, last_middle = 2;
    while (1) {
        if (current_prime - prev_prime == 2) {
            if (prev_prime != last_pair_y) {
                middle = (current_prime + prev_prime) / 2;
                if (middle - last_middle > n) {
                    result[0] = last_pair_x;
                    result[1] = last_pair_y;
                    result[2] = prev_prime;
                    result[3] = current_prime;
                    result[4] = middle;
                    result[5] = last_middle;
                    result[6] = n;
                    break;
                }
                last_middle = middle;
                last_pair_x = prev_prime;
                last_pair_y = current_prime;
            }
        }
        prev_prime = current_prime;
        current_prime += blockDim.x * gridDim.x * 2;
    }
}

int main() {
    long n;
    printf("Enter N: ");
    scanf("%ld", &n);

    long *d_result;
    long result[7];
    cudaMalloc(&d_result, 7 * sizeof(long));

    calc_cuda<<<1, 1>>>(n, d_result);

    cudaMemcpy(result, d_result, 7 * sizeof(long), cudaMemcpyDeviceToHost);

    printf("Pair 1: %ld, %ld (mid: %ld)\nPair 2: %ld, %ld (mid: %ld)\n%ld > %ld\n",
           result[0], result[1], result[5], result[2], result[3], result[4], result[4] - result[5], result[6]);

    cudaFree(d_result);

    return 0;
}