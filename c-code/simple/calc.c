#include <stdio.h>
#include <cs50.h>


float divide (int a, int b);

int main (void) {
    int a = get_int("a: ");
    int b = get_int("b: ");
    printf("%.3f\n", divide(a,b));
}

float divide (int a, int b) {
    return (float) a / (float) b;
}
