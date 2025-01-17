#include <stdio.h>
#include <cs50.h>

const int N = 5;
float avg (int length, int array[]);

int main (void) {
    int scores[N]; // Array of len 3!
    for (int i = 0; i < N; i++) {
        scores[i] = get_int("Score: " );
    }
    printf("Average: %.2f\n", avg(N, scores));
}

float avg (int length, int array[]) {
    int sum = 0;
    for (int i = 0; i < length; i++) {
        sum += array[i];
    }
    return sum / (float) length;
}
