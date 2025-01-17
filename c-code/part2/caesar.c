#include <cs50.h>
#include <stdio.h>
#include <ctype.h>
#include <string.h>

int to_int (char* n);

int main (int argc, char *argv[]) {
    if (argc!=2) {
        printf("Usage: ./caesar key\n");
        return 1;
    }
    int key = to_int(argv[1]);
    if (key==0) {
        printf("Usage: ./caesar key\n");
        return 1;
    }
    char *txt = key > 0 ? get_string("plaintext: ") : get_string("ciphertext: ");
    key = key>0 ? key % 26 : key % (-26);
    int c;
    for (int i = 0; i<strlen(txt); i++) {
        c = txt[i];
        if (c >= 'A' && c < 'Z') {
            txt[i] = ((c-'A'+key)%26)+'A';
        } else if (c >= 'a' && c < 'z') {
            txt[i] = ((c-'a'+key)%26)+'a';
        }
    }
    if (key > 0) {
        printf("ciphertext: %s\n",txt);
    } else {printf("plaintext: %s\n", txt);}
}

int to_int (char* n) {
    int res = 0;
    int d;
    int t = 1;
    bool cipher = true;
    if (n[0] == '-') {
        n ++;
        cipher = false;
    }
    for (int i = strlen(n)-1; i>=0;i--) {
        d = n[i]-'0';
        if (d<0 || d>9) {return 0;}
        res += d*t;
        t *= 10;
    }
    return cipher ? res : -res;
}
