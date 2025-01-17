#include <stdio.h>
#include <cs50.h>

void meow(int i);
int get_positive_int(void);

int main(void) {
    printf("\n");
    int n = get_positive_int();
    meow(n);
}



void meow(int i) { // Function definition
    const int x = i; // Constant!
    while (i > 0) {
        printf("meow %i\n", i);
        i--; // or i -= 1
    }
    for (int j = 0; j<x; j++) {
        printf("\n%i. Same as JS!",j+1);
    }
    printf("\n");
}
int get_positive_int(void) {
    int n;
    do {
        n = get_int("Meow? ");
    } while (n < 0); // do while loop to repeat question until input makes sense
    return n;
}
