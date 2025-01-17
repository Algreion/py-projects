#include "bmp.h"

// Convert image to grayscale
void grayscale(int height, int width, RGBTRIPLE image[height][width]);

// Reflect image horizontally
void reflect(int height, int width, RGBTRIPLE image[height][width]);

// Flip image vertically
void flip(int height, int width, RGBTRIPLE image[height][width]);

// Detect edges
void edges(int height, int width, RGBTRIPLE image[height][width]);

// Blur image
void blur(int height, int width, RGBTRIPLE image[height][width]);

// Convert image to sepia
void sepia(int height, int width, RGBTRIPLE image[height][width]);

// Invert image
void invert(int height, int width, RGBTRIPLE image[height][width]);

void cipher(int height, int width, RGBTRIPLE image[height][width]);

void decipher(int height, int width, RGBTRIPLE image[height][width]);
