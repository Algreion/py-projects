#include "helpers.h"
#include <math.h>

// Convert image to grayscale
void grayscale(int height, int width, RGBTRIPLE image[height][width])
{
    // rgb(x,x,x) where x is avg of R, G and B.
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int r = image[i][j].rgbtRed;
            int g = image[i][j].rgbtGreen;
            int b = image[i][j].rgbtBlue;
            int avg = round((r+g+b)/3.0);
            image[i][j].rgbtRed = avg;
            image[i][j].rgbtGreen = avg;
            image[i][j].rgbtBlue = avg;
        }
    }
    return;
}

// Convert image to sepia
void sepia(int height, int width, RGBTRIPLE image[height][width])
{
    /*  sepiaRed = .393 * originalRed + .769 * originalGreen + .189 * originalBlue
      sepiaGreen = .349 * originalRed + .686 * originalGreen + .168 * originalBlue
       sepiaBlue = .272 * originalRed + .534 * originalGreen + .131 * originalBlue */
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int r = image[i][j].rgbtRed;
            int g = image[i][j].rgbtGreen;
            int b = image[i][j].rgbtBlue;
            int sr = round(0.393*r + 0.769*g + 0.189*b);
            int sg = round(0.349*r + 0.686*g + 0.168*b);
            int sb = round(0.272*r + 0.534*g + 0.131*b);
            sr = (sr > 255) ? 255 : sr;
            sg = (sg > 255) ? 255 : sg;
            sb = (sb > 255) ? 255 : sb;
            image[i][j].rgbtRed = sr;
            image[i][j].rgbtGreen = sg;
            image[i][j].rgbtBlue = sb;
        }
    }
    return;
}

// Reflect image horizontally
void reflect(int height, int width, RGBTRIPLE image[height][width])
{
    RGBTRIPLE copy[height][width];
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            copy[i][j] = image[i][j];
    }}
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            image[i][j] = copy[i][width-j];
        }
    }
    return;
}

// Flip image vertically
void flip(int height, int width, RGBTRIPLE image[height][width])
{
    RGBTRIPLE copy[height][width];
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            copy[i][j] = image[i][j];
    }}
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            image[i][j] = copy[height-i][j];
        }
    }
    return;
}

// Blur image
void blur(int height, int width, RGBTRIPLE image[height][width])
{
    // Box method, aka. average the color values of a 3x3
    int dirs[18] = {1,0,0,1,-1,0,0,-1,1,1,-1,-1,1,-1,-1,1,0,0};
    RGBTRIPLE copy[height][width];
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            copy[i][j] = image[i][j];
        }
    }
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int r = 0;
            int g = 0;
            int b = 0;
            float n = 0;
            for (int x = 0; x < 18; x+=2) {
                int dw = dirs[x];
                int dh = dirs[x+1];
                int w = j+dw;
                int h = i+dh;
                if (h<0 || w<0 || h>=height || w >= width) {
                    continue;
                } else {
                    r += copy[h][w].rgbtRed;
                    g += copy[h][w].rgbtGreen;
                    b += copy[h][w].rgbtBlue;
                    n++;
                }
            }
            image[i][j].rgbtRed = round(r/n);
            image[i][j].rgbtGreen = round(g/n);
            image[i][j].rgbtBlue = round(b/n);
        }
    }
    return;
}

// Detect edges
void edges(int height, int width, RGBTRIPLE image[height][width])
{
    // Sobel operator
    int dirs[18] = {1,0,0,1,-1,0,0,-1,1,1,-1,-1,1,-1,-1,1,0,0};
    int Gx[9] = {2,0,-2,0,1,-1,1,-1,0};
    int Gy[9] = {0,-2,0,2,-1,1,1,-1,0};
    RGBTRIPLE copy[height][width];
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            copy[i][j] = image[i][j];
        }
    }
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float rX = 0;
            float gX = 0;
            float bX = 0;
            float rY = 0;
            float gY = 0;
            float bY = 0;
            int r;
            int g;
            int b;
            for (int x = 0; x < 18; x+=2) {
                int dw = dirs[x];
                int dh = dirs[x+1];
                int w = j+dw;
                int h = i+dh;
                if (h<0 || w<0 || h>=height || w >= width) {
                    continue;
                } else {
                    rX += copy[h][w].rgbtRed * Gx[x/2];
                    gX += copy[h][w].rgbtGreen * Gx[x/2];
                    bX += copy[h][w].rgbtBlue * Gx[x/2];
                    rY += copy[h][w].rgbtRed * Gy[x/2];
                    gY += copy[h][w].rgbtGreen * Gy[x/2];
                    bY += copy[h][w].rgbtBlue * Gy[x/2];
                }
            }
            r = round(sqrt(rX*rX+rY*rY));
            g = round(sqrt(gX*gX+gY*gY));
            b = round(sqrt(bX*bX+bY*bY));
            r = (r > 255) ? 255 : (r < 0 ? 0 : r);
            g = (g > 255) ? 255 : (g < 0 ? 0 : g);
            b = (b > 255) ? 255 : (b < 0 ? 0 : b);
            image[i][j].rgbtRed = r;
            image[i][j].rgbtGreen = g;
            image[i][j].rgbtBlue = b;
        }
    }
    return;
}

// Invert image
void invert(int height, int width, RGBTRIPLE image[height][width])
{
    // Inverted values for each (255-color)
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int r = image[i][j].rgbtRed;
            int g = image[i][j].rgbtGreen;
            int b = image[i][j].rgbtBlue;
            image[i][j].rgbtRed = 255-r;
            image[i][j].rgbtGreen = 255-g;
            image[i][j].rgbtBlue = 255-b;
        }
    }
    return;
}

// Cipher image
// problems/week4/filter/backup $ for file in *; do ../filter -c ../images/$file $file
void cipher(int height, int width, RGBTRIPLE image[height][width])
{
    // Trust the process
    int key = 421697+height;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int r = image[i][j].rgbtRed;
            int g = image[i][j].rgbtGreen;
            int b = image[i][j].rgbtBlue;
            image[i][j].rgbtRed = (r^key) & 0xff;
            key += i*j*19;
            image[i][j].rgbtGreen = (g^key) & 0xff;
            key -= i*j*17;
            image[i][j].rgbtBlue = (b^key) & 0xff;
            key += i*j*29;
        }
    }
    return;
}

// problems/week4/filter/backup $ mkdir ../new
// problems/week4/filter/backup $ for file in *; do ../filter -d $file ../new/$file
void decipher(int height, int width, RGBTRIPLE image[height][width])
{
    // Trust the process
    int key = 421697+height;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int r = image[i][j].rgbtRed;
            int g = image[i][j].rgbtGreen;
            int b = image[i][j].rgbtBlue;
            image[i][j].rgbtRed = (r^key) & 0xff;
            key += i*j*19;
            image[i][j].rgbtGreen = (g^key) & 0xff;
            key -= i*j*17;
            image[i][j].rgbtBlue = (b^key) & 0xff;
            key += i*j*29;
        }
    }
    return;
}
