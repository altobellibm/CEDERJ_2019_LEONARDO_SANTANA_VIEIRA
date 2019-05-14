#include <fstream>
#include <iostream>

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

using namespace std;

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

bool svd(int M, cusp::array2d<float, cusp::device_memory>& M_denseHM,
		cusp::array2d<float, cusp::device_memory>& U,
		cusp::array1d<float, cusp::device_memory>& S) {

	thrust::device_ptr<float> dev_ptr = &(M_denseHM.values[0]);
	float *M_raw_ptr = thrust::raw_pointer_cast(dev_ptr);

	int work_size = 0;

	int *devInfo;
	cudaMalloc(&devInfo, sizeof(int));
	float *d_U;
	cudaMalloc(&d_U, M * M * sizeof(float));
	float *d_V;
	cudaMalloc(&d_V, M * M * sizeof(float));
	float *d_S;
	cudaMalloc(&d_S, M * sizeof(float));

	//cusolverStatus_t stat;

	cusolverDnHandle_t solver_handle;
	cusolverDnCreate(&solver_handle);

	cusolverDnSgesvd_bufferSize(solver_handle, M, M, &work_size);

	float *work;
	cudaMalloc(&work, work_size * sizeof(float));

	cusolverDnSgesvd(solver_handle, 'A', 'A', M, M, M_raw_ptr, M, d_S, d_U, M,
			d_V, M, work, work_size, NULL, devInfo);

	int devInfo_h = 0;
	cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);

	thrust::device_ptr<float> dev_ptr_U(d_U);
	thrust::copy(thrust::device, dev_ptr_U, dev_ptr_U + (M * M),
			U.values.begin());

	thrust::device_ptr<float> dev_ptr_S(d_S);
	thrust::copy(thrust::device, dev_ptr_S, dev_ptr_S + M, S.begin());

	cusolverDnDestroy(solver_handle);

	return 1;
}

int main(int argc, char *argv[]) {

	typedef cusp::array1d<float, cusp::device_memory> Array1d;
	typedef cusp::array2d<float, cusp::device_memory> Array2d;
	typedef cusp::array2d<float, cusp::device_memory>::column_view column_view;

	if (argc != 2) {
		cout << "Argumento incorreto." << endl;
		return 1;
	}

	string strArquivo = argv[1];
	cout << strArquivo << endl;

	ifstream ifsArquivo;
	string strLinha;
	ifsArquivo.open(strArquivo);

	int intNrLinhas = 0;
	int intNrColunas = 0;
	int intLinha = 0;
	int intColuna = 0;

	while (getline(ifsArquivo, strLinha)) {
		intNrLinhas++;
		stringstream ssLinha(strLinha);
		string strNumero;
		intColuna = 0;

		while (getline(ssLinha, strNumero, ' ') && (intLinha == 0)) {
			intNrColunas++;
			intColuna++;
		}

		intLinha++;
	}

	Array2d matriz(intNrLinhas, intNrColunas);

	intLinha = 0;
	intColuna = 0;

	ifsArquivo.clear();
	ifsArquivo.seekg(0);

	while (getline(ifsArquivo, strLinha)) {
		stringstream ssLinha(strLinha);
		string strNumero;
		intColuna = 0;

		while (getline(ssLinha, strNumero, ' ')) {
			matriz(intLinha, intColuna) = stoi(strNumero);
			intColuna++;
		}

		intLinha++;
	}

	ifsArquivo.close();

	cusp::print(matriz);

	Array2d E(intNrLinhas, intNrColunas);

	E = matriz;

	thrust::transform(matriz.values.begin(), matriz.values.end(),
			matriz.values.begin(), cusp::multiplies_value<int>(-2));

	cusp::print(matriz);

	thrust::transform(matriz.values.begin(), matriz.values.end(),
			matriz.values.begin(), cusp::plus_value<int>(matriz.num_cols + 1));

	cusp::print(matriz);

	Array2d matrizTransposta;

	cusp::transpose(matriz, matrizTransposta);

	cusp::print(matrizTransposta);

	Array2d C;

	cusp::multiply(matrizTransposta, matriz, C);

	cusp::print(C);

	thrust::transform(C.values.begin(), C.values.end(), C.values.begin(),
			cusp::divide_value<float>(
					matriz.num_cols * matriz.num_rows * (matriz.num_cols - 1)
							* (matriz.num_cols - 1)));

	cusp::print(C);

	cusp::array2d<float, cusp::device_memory> U(C.num_cols, C.num_cols);
	cusp::array1d<float, cusp::device_memory> S(C.num_cols);

	//std::cout << "Inicio da Decomposicao" << std::endl;
	svd(C.num_cols, C, U, S);

	cusp::print(U);

	cusp::print(S);

	// Teste

	U(0, 0) = 0.445644;
	U(0, 1) = -0.12139;
	U(0, 2) = -0.00336875;
	U(0, 3) = -0.617952;
	U(0, 4) = 0.30292;
	U(0, 5) = 0.104751;
	U(0, 6) = 0.161157;
	U(0, 7) = -0.19879;
	U(0, 8) = 0.486383;
	U(1, 0) = -0.711218;
	U(1, 1) = 0.0423734;
	U(1, 2) = 0.0552392;
	U(1, 3) = -0.441138;
	U(1, 4) = 0.0339571;
	U(1, 5) = 0.0783436;
	U(1, 6) = -0.00482367;
	U(1, 7) = 0.470035;
	U(1, 8) = 0.257823;
	U(2, 0) = 0.31457;
	U(2, 1) = -0.0603901;
	U(2, 2) = -0.0471155;
	U(2, 3) = 0.0886266;
	U(2, 4) = -0.100349;
	U(2, 5) = 0.76635;
	U(2, 6) = 0.103892;
	U(2, 7) = 0.515145;
	U(2, 8) = -0.117446;
	U(3, 0) = 0.282895;
	U(3, 1) = 0.608804;
	U(3, 2) = -0.263212;
	U(3, 3) = -0.367443;
	U(3, 4) = 0.0698713;
	U(3, 5) = -0.262026;
	U(3, 6) = -0.210852;
	U(3, 7) = 0.299143;
	U(3, 8) = -0.370876;
	U(4, 0) = -0.244688;
	U(4, 1) = 0.279298;
	U(4, 2) = -0.70117;
	U(4, 3) = -0.0279127;
	U(4, 4) = -0.212571;
	U(4, 5) = 0.328237;
	U(4, 6) = 0.175306;
	U(4, 7) = -0.423009;
	U(4, 8) = 0.0843047;
	U(5, 0) = -0.12212;
	U(5, 1) = 0.253274;
	U(5, 2) = 0.37022;
	U(5, 3) = -0.0908042;
	U(5, 4) = 0.170505;
	U(5, 5) = 0.452362;
	U(5, 6) = -0.623452;
	U(5, 7) = -0.380869;
	U(5, 8) = -0.0904064;
	U(6, 0) = 0.189107;
	U(6, 1) = 0.291328;
	U(6, 2) = 0.222269;
	U(6, 3) = 0.0265378;
	U(6, 4) = -0.744001;
	U(6, 5) = -0.0819397;
	U(6, 6) = -0.158411;
	U(6, 7) = 0.0552435;
	U(6, 8) = 0.490776;
	U(7, 0) = -0.00735859;
	U(7, 1) = -0.496706;
	U(7, 2) = -0.0182501;
	U(7, 3) = -0.500389;
	U(7, 4) = -0.506722;
	U(7, 5) = -0.000417037;
	U(7, 6) = -0.120787;
	U(7, 7) = -0.147379;
	U(7, 8) = -0.457635;
	U(8, 0) = -0.0773703;
	U(8, 1) = 0.368016;
	U(8, 2) = 0.496989;
	U(8, 3) = -0.143339;
	U(8, 4) = -0.0870132;
	U(8, 5) = 0.0886237;
	U(8, 6) = 0.677971;
	U(8, 7) = -0.189519;
	U(8, 8) = -0.282924;

	S[0] = -5.05182e-017;
	S[1] = -8.22593e-018;
	S[2] = -4.1709e-018;
	S[3] = 1.18595e-018;
	S[4] = 7.85299e-018;
	S[5] = 1.23348e-017;
	S[6] = 0.0396248;
	S[7] = 0.152448;
	S[8] = 0.224594;

	cusp::print(U);

	cusp::print(S);

	// FimTeste

	Array1d rho(intNrColunas - 1);

	thrust::transform(S.rbegin(), S.rend(), rho.begin(),
			cusp::sqrt_functor<float>());

	cusp::print(rho);

	Array2d x(intNrColunas, intNrColunas - 1);

	for (int i = 0; i < intNrColunas; ++i) {
		for (int j = 0; j < (intNrColunas - 1); ++j) {
			x(i, j) = -U(i, intNrColunas - j - 1);
		}
	}

	cusp::print(x);

	Array2d x_sqr(intNrColunas, intNrColunas - 1);

	thrust::transform(x.values.begin(), x.values.end(), x_sqr.values.begin(),
			cusp::square_functor<float>());

	cusp::print(x_sqr);

	int ft = intNrLinhas * intNrColunas * (intNrColunas - 1);

	cout << "ft" << endl << ft << endl;

	Array2d Dc(intNrColunas, intNrColunas);
	Array1d index(intNrColunas * intNrColunas);
	thrust::sequence(index.begin(), index.end(), 0);
	thrust::transform_if(index.begin(), index.end(), Dc.values.begin(),
			cusp::constant_functor<int>(intNrLinhas * (intNrColunas - 1)),
			is_diagonal<int>(intNrColunas));

	cusp::print(Dc);

	Array2d T;
	cusp::multiply(Dc, x_sqr, T);

	cusp::print(T);

	Array1d cc(intNrColunas - 1);

	for (int i = 0; i < intNrColunas - 1; i++) {
		column_view v = T.column(i);
		Array1d a;
		cusp::convert(v, a);
		float resultado = thrust::reduce(a.begin(), a.end());
		cc[i] = ft / resultado;
	}

	thrust::transform(cc.begin(), cc.end(), cc.begin(),
			cusp::sqrt_functor<float>());

	cusp::print(cc);

	Array2d x_normed(intNrColunas, intNrColunas - 1);

	for (int i = 0; i < intNrColunas; i++) {
		int j = i * 8;
		thrust::transform(x.values.begin() + j, x.values.end() + 8 + j,
				cc.begin(), x_normed.values.begin() + j,
				thrust::multiplies<float>());
	}

	cusp::print(x_normed);

	int fr = intNrColunas * (intNrColunas - 1);

	cout << "fr" << endl << fr << endl;

	Array2d y_normed;

	cusp::multiply(E, x_normed, y_normed);

	cusp::print(y_normed);

	return 0;
}
