#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

// Forensically recovers JPEG images from deleted stream

int main(int argc, char *argv[])
{
    if (argc != 2) {
        printf("Usage: %s file.raw\n", argv[0]);
        return 1;
    }
    FILE *raw = fopen(argv[1], "rb");
    if (raw == NULL) {
        printf("Unable to open the file %s.\n",argv[1]);
        return 1;
    }
    FILE *out = NULL;
    uint8_t block[512];
    int jpeg = 0;
    char fin[8];
    while (fread(block, sizeof(uint8_t)*512, 1, raw)==1) {
         if (block[0] == 0xff && block[1] == 0xd8 && block[2] == 0xff && (block[3] & 0xf0) == 0xe0) { // & used as bitmask here to get first 4 bits
                if (out != NULL) {
                    fclose(out);
                }
                snprintf(fin, sizeof(fin), "%03d.jpg", jpeg++);
                out = fopen(fin, "wb");
            }
        if (out != NULL) {
            fwrite(&block, sizeof(uint8_t)*512, 1, out);
        }
        }
    if (out != NULL) {
        fclose(out);
    }
    fclose(raw);
    return 0;
}
