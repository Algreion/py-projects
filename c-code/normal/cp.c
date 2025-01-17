#include <stdio.h>
#include <stdint.h>

typedef uint8_t BYTE;

int main (int argc, char *argv[]) {
    // COMMAND: cp file -> destination
    if (argc != 3) {
        printf("Usage: ./c src.file dst.file\n");
        return 1;
        };
    FILE *src = fopen(argv[1], "rb"); // Read and write in binary
    if (src == NULL) {
        printf("Could not open file.\n");
        return 1;
    }
    FILE *dst = fopen(argv[2], "wb");
    BYTE b;
    while (fread(&b, sizeof(BYTE), 1, src) != 0) {
        fwrite(&b, sizeof(BYTE), 1, dst);
    }
    fclose(dst);
    fclose(src);
    return 0;
}
