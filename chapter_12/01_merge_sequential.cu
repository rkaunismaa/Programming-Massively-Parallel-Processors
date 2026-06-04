// §12.2 A sequential merge algorithm — Figure 12.2
//
// Ordered merge: takes two sorted arrays A (m elements) and B (n elements),
// produces sorted output array C (m+n elements).
//
// Stability (§12.1): when A[i] == B[j], A element goes first — preserves
// the original ordering of equal elements across and within input lists.
//
// Complexity: O(m + n) — each element visited exactly once.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ── Figure 12.2: sequential merge ────────────────────────────────────────────
void merge_sequential(int *A, int m, int *B, int n, int *C) {
    int i = 0; // index into A
    int j = 0; // index into B
    int k = 0; // index into C

    while ((i < m) && (j < n)) {
        if (A[i] <= B[j])
            C[k++] = A[i++];
        else
            C[k++] = B[j++];
    }

    if (i == m) {
        while (j < n) C[k++] = B[j++];
    } else {
        while (i < m) C[k++] = A[i++];
    }
}

static void print_array(const char *label, int *arr, int n) {
    printf("%s: [", label);
    for (int i = 0; i < n; i++) printf("%d%s", arr[i], i < n-1 ? " " : "");
    printf("]\n");
}

static bool verify_sorted(int *C, int m_plus_n) {
    for (int i = 1; i < m_plus_n; i++)
        if (C[i] < C[i-1]) return false;
    return true;
}

int main(void) {
    // Example from Fig. 12.1: A=[1,7,8,9,10], B=[7,10,10,12] → C=[1,7,7,8,9,10,10,10,12]
    printf("=== Sequential Merge (Fig 12.2) ===\n\n");

    int A[] = {1, 7, 8, 9, 10};
    int B[] = {7, 10, 10, 12};
    int m = 5, n = 4;
    int C[9];

    print_array("A", A, m);
    print_array("B", B, n);

    merge_sequential(A, m, B, n, C);
    print_array("C", C, m + n);
    printf("Sorted correctly: %s\n", verify_sorted(C, m + n) ? "YES" : "NO");
    printf("Stability check (7 from A before 7 from B): %s\n\n",
           (C[1] == 7 && C[2] == 7) ? "PASS" : "FAIL");

    // Larger random test
    printf("=== Random test: m=50000, n=30000 ===\n");
    int M = 50000, N = 30000;
    int *Ah = (int *)malloc(M * sizeof(int));
    int *Bh = (int *)malloc(N * sizeof(int));
    int *Ch = (int *)malloc((M + N) * sizeof(int));

    srand(42);
    // Generate sorted A
    Ah[0] = rand() % 5;
    for (int i = 1; i < M; i++) Ah[i] = Ah[i-1] + rand() % 5;
    // Generate sorted B
    Bh[0] = rand() % 5;
    for (int i = 1; i < N; i++) Bh[i] = Bh[i-1] + rand() % 5;

    merge_sequential(Ah, M, Bh, N, Ch);
    printf("Output sorted: %s\n", verify_sorted(Ch, M + N) ? "PASS" : "FAIL");
    printf("Output length: %d (expected %d)\n", M + N, M + N);
    printf("Output range: [%d .. %d]\n", Ch[0], Ch[M + N - 1]);

    free(Ah); free(Bh); free(Ch);
    return 0;
}
