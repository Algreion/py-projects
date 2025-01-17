#include <cs50.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>

int phonebook_add(void);
int copy_file(void);

int main (void) {

}

int phonebook_add(void) {
    FILE *file = fopen("phonebook.csv","a");
    if (file == NULL) {
        return 1;
    }
    char *name = get_string("Name: ");
    char *number = get_string("Number: ");
    fprintf(file, "%s,%s\n", name, number);
    fclose(file);
    return 0;
}
