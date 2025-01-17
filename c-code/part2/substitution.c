#include <cs50.h>
#include <stdio.h>
#include <ctype.h>
#include <string.h>

void uppercase(char* str);
int check(char* str);

int main (int argc, char *argv[]) {
    if (argc!=2 || strlen(argv[1]) != 26) {
        printf("Usage: ./substitution key\n");
        return 1;
    }
    char *key = argv[1];
    uppercase(key);
    if (check(key) == 1) {
        printf("Usage: ./substitution key\n");
        return 1;
    }
    char *txt = get_string("plaintext: ");
    int c;
    for (int i = 0; i<strlen(txt); i++) {
        c = txt[i];
        if (c >= 65 && c < 65+26) {
            txt[i] = key[c-65];
        } else if (c >= 97 && c < 97+26) {
            txt[i] = key[c-97]+32;
        }
    }
    printf("ciphertext: %s\n",txt);
}

void uppercase (char* str) {
    for (int i = 0, len = strlen(str); i < len; i++) {
        str[i] = toupper(str[i]);
    }
}

int check (char* str) {
    int chars[26];
    for (int i=0; i < 26; i++) {
        chars[i] = 0;
    }
    for (int i=0, len = strlen(str); i < len; i++) {
        if (str[i]-65 < 0 || str[i]-65 >= 26 || chars[str[i]-65] != 0) {
            return 1;
        }
        chars[str[i]-65] = 1;
    }
    return 0;
}

