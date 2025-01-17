#include <stdio.h>

void selection_sort(int t[], int n);
void bubble_sort(int t[], int n);

int main() {
    int t[] = {5,4,3,2,1,0};
    int t2[] = {30,50,40,20,10};
    int t3[] = {3,7,4,2,1,9,5,8,99,27}
    int n = sizeof(t) / sizeof(t[0]);
    int n2 = sizeof(t2) / sizeof(t2[0]);
    selection_sort(t, n);
    bubble_sort(t2, n2);
    printf("Selection sort: \n");
    for (int i = 0; i < n; i++) {
        printf("%d ", t[i]);
    }
    printf("\n");
    for (int i = 0; i < n2; i++) {
        printf("%d ", t2[i]);
    }
    printf("\n");
    return 0;
}

void selection_sort(int t[], int n) {
    for (int i = 0; i < n - 1; i++) {
        int index = i;
        for (int j = i + 1; j < n; j++) {
            if (t[index] > t[j]) {
                index = j;
            }
        }
        int temp = t[i];
        t[i] = t[index];
        t[index] = temp;
    }
}

void bubble_sort (int t[], int n) {
    int iters = 1;
    int swapped = 1;
    while (swapped == 1) {
        swapped = 0;
        for (int i=0; i < n-iters; i++) {
            if (t[i] > t[i+1]) {
                swapped = 1;
                int temp = t[i];
                t[i] = t[i + 1];
                t[i + 1] = temp;
            }
        }
        iters ++;
    }
}

