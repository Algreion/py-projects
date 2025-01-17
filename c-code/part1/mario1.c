#include <stdio.h>
#include <stdlib.h>

int main (void) {
    int *h = malloc(sizeof(int));
    printf("h: ");
    scanf("%i",h);
    if (h > 0) {
        for (int i = 0; i <= *h; i++) {
            for (int j = 0; j < *h-i; j++) {
                printf(" ");
            }
            for (int j=0; j < i; j++) {
                printf("#");
            }
            printf("\n");
        }
    }
    free(h);
}
