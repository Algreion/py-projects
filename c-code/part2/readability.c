#include <stdio.h>
#include <cs50.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

void uppercase(char* str);

int main (void) {
    // index = 0.0588 * L - 0.296 * S - 15.8
    char* txt = get_string("Text: ");
    uppercase(txt);
    float L = 0;
    float S = 0;
    float s = 0;
    float l = 0;
    int words = 0;
    for (int i = 0; i < strlen(txt); i++) {
        if (txt[i] == '.' || txt[i] == '!' || txt[i] == '?') {s += 1;}
        else if (txt[i] == ' ') {words += 1;}
        else if (txt[i]-65 >= 0 && txt[i]-65 < 26) {
            int x = txt[i]-65;
            l += 1;
        }
    }
    words += 1;
    L = (l*100)/words;
    S = (s*100)/words;
    int g = round(0.0588*L - 0.296*S - 15.8);
    if (g < 1) {
        printf("Before Grade 1\n");
    } else if (g >= 16) {
        printf("Grade 16+\n");
    } else {
        printf("Grade %i\n", g);
    }
}

void uppercase (char* str) {
    for (int i = 0, len = strlen(str); i < len; i++) {
        str[i] = toupper(str[i]);
    }
}
