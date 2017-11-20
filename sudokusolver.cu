#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <curand.h>

#define N 9
#define n 3

void load(char* filename, int* board) {
	FILE * f = fopen(filename, "r");

	if(f == NULL) {
		printf("Could not open file\n"); return;
	}

	char tmpBuff;

	for(int i=0; i < N; i++) {
		for(int j=0; j < N; j++) {
			if(!fscanf(f, "%c", &tmpBuff)) {
				printf("Error reading char\n");
				return;
			}

			if(tmpBuff >= '1' && tmpBuff <= '9') {
				board[i*N + j] = (int) (tmpBuff - '0');
			} else {
				board[i*N + j] = 0;
			}
		}
	}

}

void printBoard(int *board) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", board[i*N + j]);
        }
    }
}


int main(int argc, char* argv[]) {
	if (argc < 2){
        printf("Usage: sudokusolver (filename.in)\n");
        exit(-1);
    }

    char* filename = argv[1];
    
    //store board as flattened 9*9 int array
    int* board = new int[N*N];
    load(filename, board);

    printBoard(board);
	return 0;
}