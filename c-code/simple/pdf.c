#include <stdio.h>
#include <stdint.h>


int main (int argc, char* argv[]) {
    if (argc != 2) {
        printf("Provide only one file name.\n");
        return 1;
    }
    FILE *inp = fopen(argv[1], "r");
    uint8_t buffer[4];
    uint8_t signature[] = {0x25, 0x50, 0x44, 0x46}; // Signature of PDF files
    fread(buffer, sizeof(uint8_t), 4, inp);
    for (int i = 0; i <4; i++) {
        if (signature[i] != buffer[i]) {
            printf("Not a PDF.\n");
            return 0;
        }
    }
    printf("PDF!\n");
    fclose(inp);
    return 0;
}
