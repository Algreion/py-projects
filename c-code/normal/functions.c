#include <stdio.h>
#include <cs50.h>
#include <string.h>
#include <ctype.h>

int length(const string str);
void uppercase(string str);
void lowercase(string str);
int get_integer (const char *str);

int main (void) {
    int x = 1;
}


typedef struct {
    string name;
    string number;
} person;


int length (const string str) {
    int i = 0;
    while (str[i] != '\0') {
        i ++;
    }
    return i;
} // or <string> and strlen(str) | Note that special chars (eg. è) are +2 length here

void uppercase (string str) {
    for (int i = 0, len = strlen(str); i < len; i++) {
        if ('a' <= str[i] && str[i] <= 'z') {
            printf("%c", str[i] - 32);
        } else {
            printf("%c",str[i]);
        }
    }
    printf("\n");
} // toupper(str)

void lowercase (string str) {
    for (int i = 0, len = strlen(str); i < len; i++) {
        if ('a' <= str[i] && str[i] <= 'z') {
            printf("%c", str[i]);
        } else {
            printf("%c",str[i] + 32);
        }
    }
    printf("\n");
} // tolower(str)

int get_integer (const char * str) {
    int n;
    printf("%s",str);
    scanf("%i", &n);
    return n;
}
