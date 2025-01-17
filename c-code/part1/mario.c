#include <stdio.h>
#include <stdlib.h>
#include <cs50.h>

int main (void) {
    int h;
    do {
        h = get_int("height: ");
    } while (h <= 0);
    if (h > 0) {
        for (int i = 1; i <= h; i++) {
            for (int j = 0; j < h-i; j++) {
                printf(" ");
            }
            for (int j=0; j < i; j++) {
                printf("#");
            }
            printf("  ");
            for (int j=0; j < i; j++) {
                printf("#");
            }
            printf("\n");
        }
    }
}
