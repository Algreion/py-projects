#include <stdio.h>
#include <cs50.h>

int main (void) {
    char* name = get_string("Name: ");
    printf("hello, %s\n", name);
}
