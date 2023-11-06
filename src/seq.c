#include <omp.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void run();
bool is_prime(long num);
bool is_prime_between_pairs(long x, long y);
void calc(long n);

int main() {
    run();
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
    printf("Seq time = %.3lf sec\n", (end - start));
}

void calc(long n) {
    long last_pair_x = 2, last_pair_y = 3;
    long prev_prime = 5, current_prime = 7;
    long middle = 6, last_middle = 2;
    while (1) {
        if (is_prime(current_prime)) {
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
}

bool is_prime(long num) {
    if (num < 2)
        return false;
    for (long i = 2; i <= sqrt(num); i++) {
        if (num % i == 0)
            return false;
    }
    return true;
}

bool is_prime_between_pairs(long x, long y) {
    for (int i = x + 1; i < y; i++) {
        if (is_prime(i))
            return true;
    }
    return false;
}