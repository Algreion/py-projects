#include <cs50.h>
#include <stdio.h>
#include <ctype.h>
#include <string.h>

void uppercase(char* str);

int main (void) {
    int c;
    char* p1 = get_string("Player 1: ");
    char* p2 = get_string("Player 2: ");
    uppercase(p1);
    uppercase(p2);
    int s1 = 0;
    int s2 = 0;
    int abc[] = {1,3,3,2,1,4,2,4,1,8,5,1,3,1,1,3,10,1,1,1,1,4,4,8,4,10};
    for (int i = 0; i < strlen(p1); i++) {
        c = p1[i]-65;
        if (c>=0 && c < 26) {s1 += abc[c];}
    }
    for (int i = 0; i < strlen(p2); i++) {
        c = p2[i]-65;
        if (c>=0 && c < 26) {s2 += abc[c];}
    }
    if (s2 > s1) {
        printf("Player 2 wins!\n");
    } else if (s1 > s2) {
        printf("Player 1 wins!\n");
    } else {
        printf("Tie!\n");
    }
}

void uppercase (char* str) {
    for (int i = 0, len = strlen(str); i < len; i++) {
        str[i] = toupper(str[i]);
    }
}
