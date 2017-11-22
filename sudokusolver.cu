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

#define cudaCheckError() {                                          \
	cudaError_t error = cudaGetLastError();	\
	if(error != cudaSuccess) {	\
		printf("CUDA error: %s\n", cudaGetErrorString(error));	\
		exit(-1);	\
	}                                                             \
}

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
					int* boardIdx,
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

__device__
void resetBitmap(bool* map, int mapsize) {
	for(int i=0; i < mapsize; i++) {
		map[i] = false;
	}
}

//checks if value has been seen earlier
//if it has the board is invalid
__device__
bool seenIt(int val, bool* seen) {
	if (val != 0) {
        if (seen[val-1]) {
            return true;
        }
        seen[val-1] = true;
        return false;
    }
    return false;
}

//check entire board for value
__device__
bool valid(const int* board) {
	bool seen[N];
	resetBitmap(seen, N);

	//check row for repetitions
	for(int i=0; i<N; i++) {
		resetBitmap(seen, N);

		for(int j=0; j<N; j++) {
			int v = board[i*N + j];

			if(seenIt(v, seen)) {
				return false;
			}
		}
	}

	//check col for repetitions
	for(int j=0; j<N; j++) {
		resetBitmap(seen, N);

		for(int i=0; i<N; i++) {
			int v = board[i*N + j];

			if(seenIt(v, seen)) {
				return false;
			}
		}
	}

	//check 3x3 for repetitions
	for(int ridx=0; ridx < n; ridx++) {
		for(int cidx=0; cidx < n; cidx++) {
			resetBitmap(seen, N);

			for(int i=0; i<n; i++) {
				for(int j=0; j<n; j++) {
					int v = board[(ridx*n + i)*N + (cidx*n + j)];

					if(seenIt(v, seen)) {
						return false;
					}
				}
			}
		}
	}

	return true;
}

//chech if change is valud
__device__
bool valid(const int* board, int changedIdx) {
	int r = changedIdx / 9;
	int c = changedIdx % 9;

	bool seen[N];
	resetBitmap(seen, N);

	if(changedIdx < 0) {
		return valid(board);
	}
	if((board[changedIdx] < 1) || (board[changedIdx] > 9)) {
		return false;
	}

    //check for repetitions in row
    for(int i=0; i < N; i++) {
    	int v = board[r*N + i];

    	if(seenIt(v, seen)) {
    		return false;
    	}
    }

    //check for repetitions in col
    resetBitmap(seen, N);
    for(int i=0; i < N; i++) {
    	int v = board[i*N + c];

    	if(seenIt(v, seen)) {
    		return false;
    	}
    }

    //check 3x3 for repetitions
    int ridx = r / n;
    int cidx = c / n;

    resetBitmap(seen, N);
    for (int i=0; i < n; i++) {
    	for (int j=0; j < n; j++) {
    		int v = board[(ridx*n + i)*N + (cidx*n + j)];

    		if(seenIt(v, seen)) {
    			return false;
    		}
    	}
    }

    return true;
}

__global__
void findSolution(int* boards,
				const int numBoards,
				int* emptySpaces,
				int* emptySpacesCount,
				int* found,
				int* solution) {
	int idx = blockDim.x * gridDim.x + threadIdx.x;

	int* curBoard;
	int* curEmptySpace;
	int curEmptySpaceCount;

	while((*found == 0) && (idx < numBoards)) {
		int emptyIdx = 0;

		# if __CUDA_ARCH__>=200
			printf("hello world\n");
		#endif

		curBoard = boards + idx*N*N;
		curEmptySpace = emptySpaces + idx*N*N;
		curEmptySpaceCount = emptySpacesCount[idx];

		while((emptyIdx >= 0) && (emptyIdx < curEmptySpaceCount)) {
			curBoard[curEmptySpace[emptyIdx]]++;

			if(!valid(curBoard, curEmptySpace[emptyIdx])) {
				//backtrack
				if(curBoard[curEmptySpace[emptyIdx]] >= 9) {
					//reset
					curBoard[curEmptySpace[emptyIdx]] = 0;
					emptyIdx--;
				}
			} else {
				emptyIdx++;
			}
		}

		// # if __CUDA_ARCH__>=200
		// 	printf("emptyIdx = %d \n", emptyIdx);
		// 	printf("curEmptySpaceCount = %d \n", curEmptySpaceCount);
		// #endif  

		if(emptyIdx == curEmptySpaceCount) {
			# if __CUDA_ARCH__>=200
				printf("found solution \n");
			#endif  
			*found = 1;

			for(int i=0; i < N*N; i++) {
				solution[i] = curBoard[i];
			}
		}

		idx += gridDim.x * blockDim.x;
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

    int* d_frontierBoards; //start BFS from these boards
    int* d_childBoards; //boards generated from iteration of BFS
    int* d_emptySpaces; //location of empty spaces in the boards
    int* d_emptySpacesCount; //number of empty spaces in each board
    int* d_boardIdx; //index within child boards

    // amount of space to allocate for boards generated from BFS
    const int maxBoardsGen = pow(2,26);

    cudaMalloc(&d_emptySpaces, maxBoardsGen*sizeof(int));
    cudaMalloc(&d_emptySpacesCount, (maxBoardsGen/81 + 1) * sizeof(int));
    cudaMalloc(&d_frontierBoards, maxBoardsGen*sizeof(int));
    cudaMalloc(&d_childBoards, maxBoardsGen*sizeof(int));
    cudaMalloc(&d_boardIdx, sizeof(int));

    cudaMemset(d_boardIdx, 0, sizeof(int));
    cudaMemset(d_frontierBoards, 0, maxBoardsGen*sizeof(int));
    cudaMemset(d_childBoards, 0, maxBoardsGen*sizeof(int));

    //ToDo: optimize block sizes
    int numBlocks = 512;
    int threadsPerBlock = 256; 
    dim3 dimGrid(numBlocks, 1, 1);
    dim3 dimBlock(threadsPerBlock, 1, 1);

    /***
    * Generate BFS of guesses, 
    * each guess generates a child board
    ***/
    //copy given board to frontier
    cudaMemcpy(d_frontierBoards, board, N*N*sizeof(int), cudaMemcpyHostToDevice);
    int totalBoards = 1;
    //call once as set up for the loop
    genChildBoards<<<dimGrid, dimBlock>>>
    	(d_frontierBoards, d_childBoards, totalBoards, d_boardIdx, d_emptySpaces, d_emptySpacesCount);

    int BFSiterations = 18;

    for(int i=0; i < BFSiterations; i++) {
    	cudaMemcpy(&totalBoards, d_boardIdx, sizeof(int), cudaMemcpyDeviceToHost);
    	// printf("total boards after an iteration %d: %d\n", i, totalBoards);
    	cudaMemset(d_boardIdx, 0, sizeof(int));

    	if(i%2 == 0) {
    		genChildBoards<<<dimGrid, dimBlock>>>
    			(d_childBoards, d_frontierBoards, totalBoards, d_boardIdx, d_emptySpaces, d_emptySpacesCount);
    	} else {
    		genChildBoards<<<dimGrid, dimBlock>>>
    			(d_frontierBoards, d_childBoards, totalBoards, d_boardIdx, d_emptySpaces, d_emptySpacesCount);
    	}
    }

    cudaMemcpy(&totalBoards, d_boardIdx, sizeof(int), cudaMemcpyDeviceToHost);
    printf("total boards: %d\n", totalBoards);

    int* d_found;
    int* d_solution; //solved board

    cudaMalloc(&d_found, sizeof(int));
    cudaMalloc(&d_solution, N*N*sizeof(int));

    cudaMemset(d_found, false, sizeof(int));
    cudaMemcpy(d_solution, board, N*N*sizeof(int), cudaMemcpyHostToDevice);

    if(BFSiterations % 2 == 1) {
    	d_childBoards = d_frontierBoards;
    }

    /***
	* Check generated boards for a solution 
	* in separate threads using backtracking 
    ***/
	findSolution<<<dimGrid, dimBlock>>>
		(d_childBoards, totalBoards, d_emptySpaces, d_emptySpacesCount, d_found, d_solution);

	cudaDeviceSynchronize();
	cudaCheckError();

	//copy solution back to host
    int* h_solution = new int[N*N];
    cudaMemset(h_solution, 0, N*N*sizeof(int));
    cudaMemcpy(h_solution, d_solution, N*N*sizeof(int), cudaMemcpyDeviceToHost);
    printBoard(h_solution);

    delete[] board;
    delete[] h_solution;
    cudaFree(d_emptySpaces);
    cudaFree(d_emptySpacesCount);
    cudaFree(d_childBoards);
    cudaFree(d_frontierBoards);
    cudaFree(d_boardIdx);
    cudaFree(d_found);
    cudaFree(d_solution);

	return 0;
}