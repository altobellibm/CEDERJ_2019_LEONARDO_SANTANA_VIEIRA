#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <algorithm>
#include <Eigen/Core>

using namespace std;

class BASE_NUM{
        friend class LeitorBaseNumerico;

public:
    BASE_NUM(){}

     Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> getMatrix(){

        if (m_transacoes.empty()) return Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>(0,0);

        int subjects = m_transacoes.size();
        int stimuli = m_cabecalho.size();

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> dynamic_size_matrix(subjects, stimuli);

        int i = 0;
        for (auto transation : m_transacoes){
            for (auto index = 0; index  < transation.size(); index++)
                dynamic_size_matrix(i, index) = transation[index];
            i++;
        }

        return dynamic_size_matrix;
    }

    unsigned int getSizeTransation() const { return m_transacoes.size();}
    unsigned int getSizeColumn() const {return m_cabecalho.size();}

    vector<vector<int>>	getTransation() const {return m_transacoes;}

    map<int, int>  getCabecalho() const {return m_cabecalho;}
    map<int, int>  getCabecalhoIdReal() const {return m_cabecalhoIdReal;}

public:

    map<int, int> 		m_cabecalho;
    map<int, int> 		m_cabecalhoIdReal;
    vector<vector<int>>	m_transacoes;

};

class LeitorBaseNumerico
{
public:
    LeitorBaseNumerico() {}

    static bool obterDadoArquivo(const std::string& _strArquivo, BASE_NUM& _dado){
        std::ifstream myfile(_strArquivo);

        if (! myfile.is_open()) {
            std::cout << "Nao abriu o arquivo!" << std::endl;
            return false;
        }

        map<int, int> frequenceId;
        std::string line;
        while (getline(myfile, line)){
            auto transation = splitInt(line);
            _dado.m_transacoes.push_back(transation);

            for(auto id: transation)
                frequenceId[id]++;
        }

        myfile.close();

        if (_dado.m_transacoes.size() == 0){
            cout << "Arquivo vazio!" << endl;
            return false;
        }


        int indexMatrix = 0;
        for(auto pair : frequenceId){
            _dado.m_cabecalho[pair.first] = indexMatrix;
            _dado.m_cabecalhoIdReal[indexMatrix] = pair.first;
            indexMatrix++;
        }


        return true;
    }

private:


    static vector<string> split( string s, char c = ' '){
        vector<string> v;

        istringstream iss(s);
        string token;

        while (getline(iss, token, c))
            v.push_back(token);

        return v;
    }

    static vector<int> splitInt( string s, char c = ' '){
        vector<int> v;
        vector<string> resul = split(s,c);

        for(auto item : resul)
            v.push_back(atoi(item.c_str()));

        return v;
    }
};
