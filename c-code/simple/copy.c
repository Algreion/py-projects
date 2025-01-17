#include <stdio.h>
#include <cs50.h>
#include <string.h>
#include <ctype.h> // toupper
#include <stdlib.h>

int main (void) {
    char *s = get_string("s: ");
    char *t = malloc(strlen(s)+1); // Allocate enough memory for s
    if (t == NULL) {
        return 1; // Error, NULL means there is no memory left to allocate.
    }
    for (int i = 0, n = strlen(s); i <= n; i++) {
        t[i] = s[i]; // Copies all chars from s to t, including '\o'
    }
    if (strlen(t)>0) {
        t[0] = toupper(t[0]);
    }
    printf("%s\n",s);
    printf("%s\n",t);
    free(t); // Make sure no memory leaks occur
}
// strcopy(destination, source) does the same thing
