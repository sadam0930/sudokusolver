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
			if(!fscanf(f, "%c\n", &tmpBuff)) {
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
        printf("\n");
    }
}

__global__
void genChildBoards(int* frontierBoards,
					int* childBoards,
					int total_boards,
					int* emptySpaces,
					int* emptySpacesCount) {
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
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

    //ToDo: optimize block sizes
    int numBlocks = 512;
    int threadsPerBlock = 256; 

    dim3 dimGrid(numBlocks, 1, 1);
    dim3 dimBlock(threadsPerBlock, 1, 1);

    //space to allocate for boards generated from BFS
    // const int maxBoardsGen = pow(2,26);

    genChildBoards<<<dimGrid, dimBlock>>>
    	();

    // printBoard(board);
	return 0;
}