#include <cusp/array2d.h>
#include <cusp/print.h>
#include <fstream>
#include <iostream>

using namespace std;

int main(int argc, char *argv[]) {

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

	cusp::array2d<int, cusp::device_memory> matriz(intNrLinhas, intNrColunas);

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
	cusp::print(matriz.values);
	return 0;
}
