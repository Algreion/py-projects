#include <stdio.h>

void selection_sort(int t[], int n);

int main() {
    int t[] = {5,4,3,2,1};
    int n = sizeof(t) / sizeof(t[0]);
    selection_sort(t, n);
    printf("Sorted array: \n");
    for (int i = 0; i < n; i++) {
        printf("%d ", t[i]);
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
