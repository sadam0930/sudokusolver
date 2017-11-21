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
					int numFrontierBoards,
					int* emptySpaces,
					int* emptySpacesCount) {
	
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = idx * N*N;

	while(idx < numFrontierBoards) {
		//each thread searches one N*N board for next empty space
		bool found = false;

		for(int i = offset; (found == false) && (i < offset + N*N); i++) {
			if(frontierBoards[i] == 0) {
				found = true;

				//now generate a child board with valid input within the constraints
				int row = (i - offset) / N;
				int col = (i - offset) % N;

				//guess integer between [1-N]				
				for(int guess = 1; guess <= N; guess++){
					bool valid = true;
					//check if guess exists in same row
					for(int c=0; c < N; c++) {
						if(frontierBoards[row*N + c + offset] == guess) {
							valid = false;
						}
					}
					//check if guess exists in same column
					for(int r=0; r < N; r++) {
						if(frontierBoards[r*N + col + offset] == guess) {
							valid = false;
						}
					}
					//check if guess exists in same 3x3 box
					for(int r = n*(row/n); r < n; r++) {
						for(int c = n*(col/n); c < n; c++) {
							if(frontierBoards[r*N + c + offset] == guess) {
								valid = false;
							}
						}
					}

					//persist the new child board
					if(valid == true) {
						int childBoardIdx = atomicAdd(boardIdx, 1); //multiple threads updating this index
						int emptyIdx = 0;

						for(int r=0; r < N; r++) {
							for(int c=0; c < N; c++) {
								childBoards[childBoardIdx*N*N + r*N + c] = frontierBoards[idx*N*N + r*N + c];
								if(frontierBoards[idx*N*N + r*N + c] == 0 && (r != row || c != col)) {
									emptySpaces[emptyIdx + N*N*childBoardIdx] = r*9 + c;
									emptyIdx++;
								}
							}
							emptySpacesCount[childBoardIdx] = emptyIdx; //num empty spaces on board
							childBoards[childBoardIdx*N*N + row*N + col] = guess;
						}
					}
				}
			}
		}

		idx += blockDim.x * gridDim.x;
	}
}

// magic starts here
int main(int argc, char* argv[]) {
	if (argc < 2){
        printf("Usage: sudokusolver (filename.in)\n");
        exit(-1);
    }

    char* filename = argv[1];
    
    //store board as flattened 9*9 int array
    int* board = new int[N*N];
    load(filename, board);

    int* frontierBoards; //start BFS from these boards
    int* childBoards; //boards generated from iteration of BFS
    int* emptySpaces; //location of empty spaces in the boards
    int* emptySpacesCount; //number of empty spaces in each board
    int* boardIdx; //index within child boards

    // amount of space to allocate for boards generated from BFS
    const int maxBoardsGen = pow(2,26);

    cudaMalloc(&emptySpaces, maxBoardsGen*sizeof(int));
    cudaMalloc(&emptySpacesCount, (maxBoardsGen/81 + 1) * sizeof(int));
    cudaMalloc(&frontierBoards, maxBoardsGen*sizeof(int));
    cudaMalloc(&childBoards, maxBoardsGen*sizeof(int));
    cudaMalloc(&boardIdx, sizeof(int));

    cudaMemset(boardIdx, 0, sizeof(int));
    cudaMemset(frontierBoards, 0, maxBoardsGen*sizeof(int));
    cudaMemset(childBoards, 0, maxBoardsGen*sizeof(int));

    //copy given board to frontier
    cudaMemcpy(frontierBoards, board, N*N*sizeof(int), cudaMemcpyHostToDevice);
    int numFrontierBoards = 1;

    //ToDo: optimize block sizes
    int numBlocks = 512;
    int threadsPerBlock = 256; 
    dim3 dimGrid(numBlocks, 1, 1);
    dim3 dimBlock(threadsPerBlock, 1, 1);

    //call once as set up for the loop
    genChildBoards<<<dimGrid, dimBlock>>>
    	(frontierBoards, childBoards, numFrontierBoards, boardIdx, emptySpaces, emptySpacesCount);

    cudaMemcpy(&numFrontierBoards, boardIdx, sizeof(int), cudaMemcpyDeviceToHost);

    printf("total boards after an iteration %d: %d\n", i, numFrontierBoards);

    // printBoard(board);
	return 0;
}