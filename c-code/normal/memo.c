#include <stdio.h>
#include <cs50.h>

typedef char* str;

int main (void) {
    // Basics
    int n = 50;
    printf("%p\n", &n); // & is the hex address of n in memory

    int *p = &n; // Create a variable p that points to the address of n. Int is the type of n, * here means we declared a pointer(!)
    printf("%i\n", *p); // Equivalently, printf("%p\n", &n); | * here means 'go to this location in memory, dereference'

    printf("\n");

    // Strings (don't exist in C!)
    string s = "HI!";
    printf("%s\n",s);
    printf("%p\n",s); // They are actually pointers to the first character! (printf then loops over bytes until it finds '\0')
    printf("%p %p %p %p\n",&s[0], &s[1], &s[2], &s[3]); // Contiguous chars in memory
    printf("\n");
    // Hence we can say that a string is just a char *s:
    str t = "Hello, World!";
    printf("%s\n",t);
    printf("%c%c%c%c%c%c\n",*t,*(t+1),*(t+2),*(t+3),*(t+4),*(t+12)); // Pointer arithmetic, we can go to specific points in memory!
                            // for (int i = 0; i<100; i++) {
                            //     printf("%b ",*(t+i));
                            // }
}

