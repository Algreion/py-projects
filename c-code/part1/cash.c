#include <stdio.h>
#include <stdlib.h>

int main (void) {
    int c;
    printf("Change owed: ");
    scanf("%i",&c);
    int coins[4] = {25,10,5,1};
    int res = 0;
    while (c > 0) {
        for (int i=0;i<sizeof(coins)/sizeof(coins[0]);i++) {
            if (coins[i] <= c) {
                c -= coins[i];
                res++;
                break;
            }
        }
    }
    printf("%i\n",res);
}
