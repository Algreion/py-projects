#include <stdio.h>
#include <cs50.h>

int main (void) {
    long c = get_long("Number: ");
    if (c < 1000000000000) {
        printf("INVALID\n");
        return 0;
    }
    int check = 0;
    long d = c/10;
    int digit;
    do {
        digit = 2*(d%10);
        if (digit>9) {
            check += digit%10;
            digit /= 10;
        }
        check += digit;
        d /= 100;
    } while (d > 0);
    d = c;
    do {
        check += (d%10);
        d /= 100;
    } while (d > 0);
    if ((check % 10) != 0) {
        printf("INVALID\n");
        return 0;
    }
    if (c/10000000000000 == 34 || c/10000000000000 == 37) {
        printf("AMEX\n");
    } else if (c/100000000000000 >= 51 && c/100000000000000 <= 55) {
        printf("MASTERCARD\n");
    } else if (c/1000000000000 == 4 || c/1000000000000000 == 4) {
        printf("VISA\n");
    } else {
        printf("INVALID\n");
    }
}

