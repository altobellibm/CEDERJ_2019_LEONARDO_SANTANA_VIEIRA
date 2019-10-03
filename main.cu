#include <fstream>
#include <iostream>
#include <chrono>

#include <cusolverDn.h>

#include <cusp/array1d.h>
#include <cusp/array2d.h>
#include <cusp/convert.h>
#include <cusp/elementwise.h>
#include <cusp/functional.h>
#include <cusp/multiply.h>
#include <cusp/print.h>
#include <cusp/transpose.h>

#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/tabulate.h>
#include <thrust/execution_policy.h>

#include "multShare.h"

using namespace std;

typedef float type;
typedef cusp::array1d<type, cusp::device_memory> Array1d;
typedef cusp::array2d<type, cusp::device_memory> Array2d;


// convert a linear index to a row index
template <typename T>
struct linear_index_to_row_index : public thrust::unary_function<T,T>
{
  T C; // number of columns
  
  __host__ __device__
  linear_index_to_row_index(T C) : C(C) {}

  __host__ __device__
  T operator()(T i)
  {
    return i / C;
  }
};

template <typename T>
	struct reciprocal_my : public thrust::unary_function<T,T>
	{
	  T value;
	  reciprocal_my(T thr) : value(thr) {};
	  __host__ __device__ 
	  T operator()(const T& v) const {
		   return sqrt(T(value) / v);
	  }
	};

template<typename T>
struct is_true: thrust::unary_function<T, T> {
	T col;

	is_true(T _c) :
			col(_c) {
	}
	;

	__host__ __device__
	bool operator()(const T &x)
	{
		return (x % col) != 0;
	}
};

template<typename T>
struct sub_matrix: thrust::unary_function<T, T> {
	T col;
	T row;
	T pitch;

	sub_matrix(T _c, T _r, T _p) :
			col(_c), row(_r), pitch(_p) {
	};

	__host__ __device__
	bool operator()(const T &x)
	{
		return  (x % (int)pitch) < (int) col && (x / (int)pitch) < (int)row;
	}
};

template<typename T>
struct is_diagonal: thrust::unary_function<T, T> {
	T col;

	is_diagonal(T _c) :
			col(_c) {
	}
	;

	__host__ __device__
	bool operator()(const T &x)
	{
		return (x % (col + 1)) == 0;
	}
};

template <typename T>
	struct column_by_vector : public thrust::unary_function<T,T>
	{
	  T* data;
	  T col;
	  column_by_vector(T *_data, T _col) : data(_data), col(_col) {};
	  __host__ __device__ 
	  T operator()(const thrust::tuple<int,type>& v) {
		   return data[thrust::get<0>(v) % (int)col] * thrust::get<1>(v);
	 }
	};


bool svd(int M, cusp::array2d<type, cusp::device_memory>& M_denseHM,
		cusp::array2d<type, cusp::device_memory>& U,
		cusp::array1d<type, cusp::device_memory>& S) {

	thrust::device_ptr<type> dev_ptr = &(M_denseHM.values[0]);
	type *M_raw_ptr = thrust::raw_pointer_cast(dev_ptr);

	int work_size = 0;

	int *devInfo;
	cudaMalloc(&devInfo, sizeof(int));
	type *d_U;
	cudaMalloc(&d_U, M * M * sizeof(type));
	type *d_V;
	cudaMalloc(&d_V, M * M * sizeof(type));
	type *d_S;
	cudaMalloc(&d_S, M * sizeof(type));

	//cusolverStatus_t stat;

	cusolverDnHandle_t solver_handle;
	cusolverDnCreate(&solver_handle);

	cusolverDnSgesvd_bufferSize(solver_handle, M, M, &work_size);

	type *work;
	cudaMalloc(&work, work_size * sizeof(type));

	cusolverDnSgesvd(solver_handle, 'A', 'A', M, M, M_raw_ptr, M, d_S, d_U, M,
			d_V, M, work, work_size, NULL, devInfo);

	int devInfo_h = 0;
	cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);

	thrust::device_ptr<type> dev_ptr_U(d_U);
	thrust::copy(thrust::device, dev_ptr_U, dev_ptr_U + (M * M),
			U.values.begin());

	thrust::device_ptr<type> dev_ptr_S(d_S);
	thrust::copy(thrust::device, dev_ptr_S, dev_ptr_S + M, S.begin());

	cusolverDnDestroy(solver_handle);

	return 1;
}

int getmultiple16sizeMatriz(int n){
	if (n % 16)
    	n = n + (16 - n % 16);
    return n;
}

void multiply(Array2d & A, Array2d& B, Array2d& C){
	
	// Load A and B to device memory
	Array2d A_new(getmultiple16sizeMatriz(A.num_rows), getmultiple16sizeMatriz(A.num_cols));
	cusp::array1d<int,cusp::device_memory> index(A_new.num_rows*A_new.num_cols);
  	thrust::sequence(index.begin(), index.end(),0);	
  	A_new(0,0) = 0; A_new(0,1) = 1; A_new(0,2) = 2;
    A_new(1,0) = 3; A_new(1,1) = 4; A_new(1,2) = 5;
    A_new(2,0) = 6; A_new(2,1) = 7; A_new(2,2) = 8;

	cusp::print(A_new);

	Matrix d_A;
	d_A.width = d_A.stride = A_new.num_cols;
	d_A.height = A_new.num_rows;
	thrust::device_ptr<float> dev_ptr_A = &(A_new.values[0]);
	d_A.elements = thrust::raw_pointer_cast(dev_ptr_A);

	Array2d B_new(getmultiple16sizeMatriz(B.num_rows), getmultiple16sizeMatriz(B.num_cols));
	B_new(0,0) = 0; B_new(0,1) = 1; B_new(0,2) = 2;
    B_new(1,0) = 3; B_new(1,1) = 4; B_new(1,2) = 5;
    B_new(2,0) = 6; B_new(2,1) = 7; B_new(2,2) = 8;

	cusp::print(B_new);

	Matrix d_B;
	d_B.width = d_B.stride = B_new.num_cols;
	d_B.height = B_new.num_rows;
	thrust::device_ptr<float> dev_ptr_B = &(B_new.values[0]);
	d_B.elements = thrust::raw_pointer_cast(dev_ptr_B);

	Array2d C_new(getmultiple16sizeMatriz(C.num_rows), getmultiple16sizeMatriz(C.num_cols));
	// Allocate C in device memory
	Matrix d_C;
	d_C.width = d_C.stride = C_new.num_cols;
	d_C.height = C_new.num_rows;
	auto size = d_C.width * d_C.height * sizeof(float);
	auto err = cudaMalloc(&d_C.elements, size);
	printf("CUDA malloc C: %s\n",cudaGetErrorString(err));

	MatMul(d_A,d_B,d_C);

	thrust::device_ptr<type> dev_ptr_S(d_C.elements);
	thrust::copy(thrust::device, dev_ptr_S, dev_ptr_S + (d_C.width*d_C.height), C_new.values.begin());
	
	cusp::array1d<int,cusp::device_memory> index_(C_new.num_rows*C_new.num_cols);
  	thrust::sequence(index_.begin(), index_.end(),0);
  	thrust::copy_if( dev_ptr_S,  
  		             dev_ptr_S + (d_C.width*d_C.height), 
  		             index_.begin(),  
  		             C.values.begin(), 
  		             sub_matrix<int>(C.num_rows,C.num_cols,C_new.num_cols));
	
  	cudaFree(d_C.elements);
	cusp::print(C);

}

int main(int argc, char *argv[]) {

	if (argc != 2) {
		cout << "Argumento incorreto. " << endl;
		cout << "parâmetro entrada: ./main <base de dados> " << endl;
		return 1;
	}

	//auto start = std::chrono::high_resolution_clock::now();
	//******************** ler base de dados *************************
	std::string strFile = argv[1];
    std::string strPrefixFile = strFile.substr(strFile.find_last_of("/")+1,strFile.find(".dat")-4);

	ifstream file;
	file.open(strFile);

	if(!file.is_open()){
		std::cout << "Error opening file" << std::endl;
		return 1;
	}

	int intNrLinhas = 0;
	int intNrColunas = 0;
	std::string strLinha;
	std::string strNumero;
	std::vector<type> database;

	while (getline(file, strLinha)) {
		stringstream ssLinha(strLinha);
		while (getline(ssLinha, strNumero, ' ')) {
			database.push_back(type(std::stoi(strNumero)));
			if ( intNrLinhas == 0 ){
				intNrColunas++;
			}
		}
		intNrLinhas++;
	}

	file.close();

	std::cout << "linha " << intNrLinhas << " colunas " << intNrColunas << std::endl;
	
	cudaSetDevice(1); // selecionar placa de video Tesla K40c

	Array2d E(intNrLinhas, intNrColunas);
	thrust::copy(database.begin(), database.end(), E.values.begin());

	//cusp::print(E);

	//******************** E*-2 *************************

	thrust::transform(E.values.begin(), E.values.end(),
			E.values.begin(), cusp::multiplies_value<int>(-2));

	//cusp::print(E);

	//******************** E*col+1 *************************
	thrust::transform(E.values.begin(), E.values.end(),
			E.values.begin(), cusp::plus_value<int>(E.num_cols + 1));

	//cusp::print(E);

	//*********************************************
	Array2d matrizTransposta;

	cusp::transpose(E, matrizTransposta);

	cusp::print(matrizTransposta);
	
	Array2d C(E.num_cols, E.num_cols);
	multiply(matrizTransposta, E, C);

	//cusp::print(C);

	// thrust::transform(C.values.begin(), C.values.end(), C.values.begin(),
	// 		cusp::divide_value<type>(
	// 				E.num_cols * E.num_rows * (E.num_cols - 1)
	// 						* (E.num_cols - 1)));

	//cusp::print(C);
	
	//******************** Inicio da Decomposicao *************************

	cusp::array2d<type, cusp::device_memory> U(C.num_cols, C.num_cols);
	cusp::array1d<type, cusp::device_memory> S(C.num_cols);

	//std::cout << "Inicio da Decomposicao" << std::endl;
	svd(C.num_cols, C, U, S);

	//cusp::print(U);

	//cusp::print(S);

	// Teste
	// U(0, 0) =  -0.486383 ;U(0, 1) =    0.19879; U(0, 2) =   -0.161157; U(0, 3) =   -0.104751; U(0, 4) =    -0.30292; U(0, 5) =    0.617952; U(0, 6) =  0.00336875; U(0, 7) =     0.12139;
	// U(1, 0) =  -0.257823 ;U(1, 1) =  -0.470035; U(1, 2) =  0.00482367; U(1, 3) =  -0.0783436; U(1, 4) =  -0.0339571; U(1, 5) =    0.441138; U(1, 6) =  -0.0552392; U(1, 7) =  -0.0423734;
	// U(2, 0) =   0.117446 ;U(2, 1) =  -0.515145; U(2, 2) =   -0.103892; U(2, 3) =    -0.76635; U(2, 4) =    0.100349; U(2, 5) =  -0.0886266; U(2, 6) =   0.0471155; U(2, 7) =   0.0603901;
	// U(3, 0) =   0.370876 ;U(3, 1) =  -0.299143; U(3, 2) =    0.210852; U(3, 3) =    0.262026; U(3, 4) =  -0.0698713; U(3, 5) =    0.367443; U(3, 6) =    0.263212; U(3, 7) =   -0.608804;
	// U(4, 0) = -0.0843047 ;U(4, 1) =   0.423009; U(4, 2) =   -0.175306; U(4, 3) =   -0.328237; U(4, 4) =    0.212571; U(4, 5) =   0.0279127; U(4, 6) =     0.70117; U(4, 7) =   -0.279298;
	// U(5, 0) =  0.0904064 ;U(5, 1) =   0.380869; U(5, 2) =    0.623452; U(5, 3) =   -0.452362; U(5, 4) =   -0.170505; U(5, 5) =   0.0908042; U(5, 6) =    -0.37022; U(5, 7) =   -0.253274;
	// U(6, 0) =  -0.490776 ;U(6, 1) = -0.0552435; U(6, 2) =    0.158411; U(6, 3) =   0.0819397; U(6, 4) =    0.744001; U(6, 5) =  -0.0265378; U(6, 6) =   -0.222269; U(6, 7) =   -0.291328;
	// U(7, 0) =   0.457635 ;U(7, 1) =   0.147379; U(7, 2) =    0.120787; U(7, 3) = 0.000417037; U(7, 4) =    0.506722; U(7, 5) =    0.500389; U(7, 6) =   0.0182501; U(7, 7) =    0.496706;
	// U(8, 0) =   0.282924 ;U(8, 1) =   0.189519; U(8, 2) =   -0.677971; U(8, 3) =  -0.0886237; U(8, 4) =   0.0870132; U(8, 5) =    0.143339; U(8, 6) =   -0.496989; U(8, 7) =   -0.368016;
	   
	   

	// 	S[7] =-8.22593e-018;
	// 	S[6] = -4.1709e-018;
	// 	S[5] = 1.18595e-018;
	// 	S[4] = 7.85299e-018;
	// 	S[3] = 1.23348e-017;
	// 	S[2] =    0.0396248;
	// 	S[1] =     0.152448;
	// 	S[0] =     0.224594;

	//cusp::print(U);

	//cusp::print(S);

	//******************** Inicio da Decomposicao *************************

	Array1d rho(E.num_cols - 1);

	thrust::transform(S.begin(), S.end()-1, rho.begin(),
			cusp::sqrt_functor<type>());

	//std::cout << "Rho" << std::endl;
	//cusp::print(rho);

	//*********************************************
	Array2d x(E.num_cols, E.num_cols - 1);

  	cusp::array1d<int,cusp::device_memory> index(intNrColunas*intNrColunas);
  	thrust::sequence(index.begin(), index.end(),1);
  	//cusp::print(index);	
  	thrust::copy_if( U.values.begin(),  U.values.end(), index.begin(),  x.values.begin(), is_true<int>(intNrColunas));
	
	//std::cout << "X" << std::endl;
	//cusp::print(x);

	//*********************************************
	Array2d x_sqr(x.num_rows, x.num_cols);

	thrust::transform(x.values.begin(), x.values.end(), x_sqr.values.begin(),
			cusp::square_functor<type>());

	//std::cout << "x_sqr" << std::endl;
	//cusp::print(x_sqr);

	int ft = intNrLinhas * intNrColunas * (intNrColunas - 1);

	//std::cout << "ft value = " <<  ft << std::endl;
	//*********************************************

	Array2d T = x_sqr;
	thrust::transform(T.values.begin(), T.values.end(),
			T.values.begin(), cusp::multiplies_value<type>(intNrLinhas * (intNrColunas - 1)));

	//std::cout << "T" << std::endl;
	//cusp::print(T);

    //*********************************************
  	cusp::array1d<int,cusp::device_memory> index_sum_t(intNrColunas);
  	cusp::array1d<type,cusp::device_memory> marginal_sum_t(intNrColunas-1);
	
  	cusp::array2d<type,cusp::device_memory> T_T(intNrColunas,intNrLinhas);
    cusp::transpose(T, T_T);
	//cusp::print(T_T);

	Array1d index_similar(intNrColunas*(intNrColunas - 1));
	thrust::tabulate(thrust::device, index_similar.begin(), index_similar.end(), linear_index_to_row_index<int>(intNrColunas));
	//cusp::print(index_similar);

  	thrust::reduce_by_key(index_similar.begin(), 
  						  index_similar.end(), 
  						  T_T.values.begin(), 
  						  index_sum_t.begin(), 
  						  marginal_sum_t.begin(),
  						  thrust::equal_to<int>(), 
  						  thrust::plus<type>());
	
  	//std::cout << "cc" << std::endl;
  	//cusp::print(marginal_sum_t);
	
  	cusp::array1d<type,cusp::device_memory> cc(intNrColunas-1);
	
  	thrust::transform(marginal_sum_t.begin(), marginal_sum_t.end(), cc.begin(), reciprocal_my<type>(type(ft)));
	//std::cout << "cc" << std::endl;
  	//cusp::print(cc);
	
	//*********************************************

	Array1d index_X(intNrColunas*(intNrColunas - 1));
	thrust::sequence(index_X.begin(), index_X.end(),0);

  	cusp::array2d<type,cusp::device_memory> x_normed(x.num_rows,x.num_cols);
		
	thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(index_X.begin(), x.values.begin())), thrust::make_zip_iterator(thrust::make_tuple(index_X.end(), x.values.end())), x_normed.values.begin(), column_by_vector<type>(thrust::raw_pointer_cast(cc.data()),(type)x.num_cols));

    //std::cout << "x_normed" << std::endl;
  	//cusp::print(x_normed);
	
  	//**************************************
  	cusp::array2d<type,cusp::device_memory> x_project(x.num_rows,x.num_cols);
	
  	thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(index_X.begin(), x_normed.values.begin())), thrust::make_zip_iterator(thrust::make_tuple(index_X.end(), x_normed.values.end())), x_project.values.begin(), column_by_vector<type>(thrust::raw_pointer_cast(rho.data()),(type)x.num_cols));
	//std::cout << "x_project" << std::endl;
  	//cusp::print(x_project);
  	
  	//*************************** GPU para Memória principal **********************
 	cusp::array2d<type,cusp::host_memory> out(x_project);

	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::ratio<1> > elapsed_seconds = finish - start;
	auto time = elapsed_seconds.count();

	//cusp::print(x_project);

	std::ofstream myfile;
      myfile.open ("./output/" + strPrefixFile + "_dominance_tempoProcessamentoGPU.txt");
      myfile << "Nome arquivo: " << strPrefixFile << std::endl;
      myfile << "Matriz formato: [" << E.num_rows << "," << E.num_cols << "]" << std::endl;
      myfile << "Tempo calcular em GPU (segundo):" <<  time << std::endl;
      myfile << "Matriz Xproject: [" << x_project.num_rows << " , " << x_project.num_cols << " ]" << std::endl;
    myfile.close();

    std::cout << "Finished with success!" << std::endl;
*/
	return 0;
}
