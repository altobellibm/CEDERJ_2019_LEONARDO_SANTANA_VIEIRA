#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>

using namespace std;

int main(int argc, char *argv[]) {
	unsigned int nrPerguntas, nrParticipantes;
	string nomeArquivo;

	if (argc == 4) {
		nrPerguntas = stoi(argv[1]);
		nrParticipantes = stoi(argv[2]);
		nomeArquivo = argv[3];
	} else {
		cout << "Informe o número de perguntas:" << endl;
		cin >> nrPerguntas;
		cout << "Informe o número de participantes:" << endl;
		cin >> nrParticipantes;
		cout << "Informe o nome do arquivo:" << endl;
		cin >> nomeArquivo;
	}

	unsigned int controle = 0;
	ofstream arquivo;
	arquivo.open(nomeArquivo);

	while (controle++ < nrParticipantes) {
		vector<int> vetor;

		for (unsigned int i = 1; i <= nrPerguntas; i++) {
			vetor.push_back(i);
		}

		unsigned seed = chrono::system_clock::now().time_since_epoch().count();
		shuffle(vetor.begin(), vetor.end(), default_random_engine(seed));

		for (vector<int>::iterator i = vetor.begin(); i != vetor.end(); ++i) {
			arquivo << *i;

			if (vetor.end() - i != 1) {
				arquivo << " ";
			}
		}

		arquivo << endl;

	}

	arquivo.close();
	return 0;

}
