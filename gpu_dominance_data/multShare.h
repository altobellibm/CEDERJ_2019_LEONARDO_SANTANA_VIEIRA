/*
* multShare.h
*
* Robert Hochberg
* January 24, 2012
*
* Based nearly entirely on the code from the CUDA C Programming Guide
*/

#include <stdio.h>

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
	int width;
	int height;
	float* elements;
	int stride;
} Matrix;

// Thread block size
#define BLOCK_SIZE 16



// Get a matrix element
__device__ float GetElement(const Matrix A, int row, int col) {
	return A.elements[row * A.stride + col];
}
// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col, float value) {
	A.elements[row * A.stride + col] = value;
}
// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is

// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
__device__ Matrix GetSubMatrix(Matrix A, int row, int col) {
	Matrix Asub;
	Asub.width = BLOCK_SIZE;
	Asub.height = BLOCK_SIZE;
	Asub.stride = A.stride;
	Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
	return Asub;
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {
	// Block row and column
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;
	// Each thread block computes one sub-matrix Csub of C
	Matrix Csub = GetSubMatrix(C, blockRow, blockCol);
	// Each thread computes one element of Csub
	// by accumulating results into Cvalue
	float Cvalue = 0.0;
	// Thread row and column within Csub
	int row = threadIdx.y;
	int col = threadIdx.x;
	// Loop over all the sub-matrices of A and B that are
	// required to compute Csub
	// Multiply each pair of sub-matrices together
	// and accumulate the results
	for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
		// Get sub-matrix Asub of A
		Matrix Asub = GetSubMatrix(A, blockRow, m);
		// Get sub-matrix Bsub of B
	
		Matrix Bsub = GetSubMatrix(B, m, blockCol);
		// Shared memory used to store Asub and Bsub respectively
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
		// Load Asub and Bsub from device memory to shared memory
		// Each thread loads one element of each sub-matrix
		As[row][col] = GetElement(Asub, row, col);
		Bs[row][col] = GetElement(Bsub, row, col);
		
		// Synchronize to make sure the sub-matrices are loaded
		// before starting the computation
		__syncthreads();
	
		// Multiply Asub and Bsub together
		for (int e = 0; e < BLOCK_SIZE; ++e)
			Cvalue += As[row][e] * Bs[e][col];
			// Synchronize to make sure that the preceding
			// computation is done before loading two new
			// sub-matrices of A and B in the next iteration
		__syncthreads();
	}
	// Write Csub to device memory
	// Each thread writes one element
	SetElement(Csub, row, col, Cvalue);
}

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(Matrix& d_A, Matrix& d_B, Matrix& d_C) {
	
	// Allocate C in device memory
	// Matrix d_C;
	// d_C.width = d_C.stride = C.width;
	// d_C.height = C.height;
	// auto size = C.width * C.height * sizeof(float);
	// auto err = cudaMalloc(&d_C.elements, size);
	// printf("CUDA malloc C: %s\n",cudaGetErrorString(err));

	// Invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(d_B.width / dimBlock.x, d_A.height / dimBlock.y);
	
	MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
	
	auto err = cudaThreadSynchronize();
	
	printf("Run kernel: %s\n", cudaGetErrorString(err));
	
	// Read C from device memory
	//err = cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
	
	//printf("Copy C off of device: %s\n",cudaGetErrorString(err));
	
	//Free device memory
	//cudaFree(d_A.elements);
	//cudaFree(d_B.elements);

	// thrust::device_ptr<type> dev_ptr_C(d_C.elements);
	// thrust::copy(thrust::device, dev_ptr_C, dev_ptr_C + (d_C.height*d_C.width), S.begin());

	// std::cout << "C  = " << C.elements[0] << " " << C.elements[1] << std::endl;

	//thrust::device_ptr<float> dev_ptr_Result(d_C.elements);
	//thrust::copy(thrust::device, dev_ptr_Result, 
	//		    dev_ptr_Result + (C.num_rows * C.num_cols),
	//			C.values.begin());

	// float * resul = (float*)malloc(C.width * C.height * sizeof(float));
	// err = cudaMemcpy(resul, d_C.elements, size, cudaMemcpyDeviceToHost);
	// printf("Copy C off of device: %s\n",cudaGetErrorString(err));

	// std::cout << "C  = " << resul[0] << " " << resul[1] << std::endl;

	// cudaFree(d_C.elements);
}
