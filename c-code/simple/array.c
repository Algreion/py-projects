#include <stdio.h>
#include <stdlib.h>

// Doubly linked list

typedef struct node {
    struct node *nxt; // More verbose because C reads top to bottom, and it needs to know what 'node' is
    struct node *prv;
    int data;
} node;

int N = 9;

int main (void) {
    node *tail = NULL;
    for (int i = 0; i < N; i++) {
        node *n = malloc(sizeof(node));
        if (n == NULL) {
            return 1;
        }
        n -> data = i;
        n -> prv = tail;
        if (tail != NULL) {
            tail -> nxt = n;
        }
        n -> nxt = NULL;
        tail = n;
    }
    for (node *ptr = tail; ptr != NULL; ptr = ptr -> prv) {
        printf("%i\n",ptr -> data);
    }

    // Memory freeing
    while (tail != NULL) {
        node *ptr = tail->prv;
        free(tail);
        tail = ptr;
    }
    return 0;
}

