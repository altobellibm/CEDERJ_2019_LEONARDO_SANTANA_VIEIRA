#include <iostream>
#include <chrono>

#undef NDEBUG
#include <assert.h>

#include <fstream>

#define EIGEN_RUNTIME_NO_MALLOC // Define this symbol to enable runtime tests for allocations

#include "dual_scaling.h"
#include "leitorbasenumerica.h"

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::milliseconds ms;
typedef std::chrono::duration<float> fsec;

const double LIMITE_INFERIOR_DELTA = 1e-8;
typedef double real_type;
typedef double index;

typedef Eigen::Matrix<real_type, Eigen::Dynamic, Eigen::Dynamic>	dynamic_size_matrix;
typedef Eigen::Matrix<real_type, Eigen::Dynamic, 1>					column_vector_type;
typedef Eigen::Matrix<real_type, 1, Eigen::Dynamic>					row_vector_type;

typedef ds::rank_order_data<dynamic_size_matrix>                    ro;


BASE_NUM getMatrix(std::string& _strFile){

   BASE_NUM base = BASE_NUM();
   if(!LeitorBaseNumerico::obterDadoArquivo(_strFile, base)){
       std::cout << "Erro read file!" << std::endl;
       exit (EXIT_FAILURE);
   }
   return base;
}

int main(int argc, char* argv[]){

    if(argc != 2)
        return 1;

    std::string strFile = argv[1];
    std::string strPrefixFile = strFile.substr(strFile.find_last_of("/")+1,strFile.find("."));

    // dominance_data
    dynamic_size_matrix x_normedR, x_projectedR;
    dynamic_size_matrix y_normedR, y_projectedR;
    row_vector_type rhoR, deltaR;

    auto t0 = Time::now();
      auto baseDominance = getMatrix(strFile);
      auto matrix = ro(baseDominance.getMatrix());
    auto t1 = Time::now();

    fsec fs = t1 -t0;

    std::cout << "Started Dual Scaling " << endl;
    t0 = Time::now();
      ds::dual_scaling(matrix, x_normedR, y_normedR, x_projectedR, y_projectedR, rhoR, deltaR);
    t1 = Time::now();

    fsec calculo = t1 -t0;

    std::ofstream myfile;
      myfile.open (strPrefixFile + "_dominance_tempoProcessamentoCPU.txt");
      myfile << "Nome arquivo: " << strFile << std::endl;
      myfile << "Matriz formato: [" << baseDominance.getSizeTransation() << "," << baseDominance.getSizeColumn() << "]" << std::endl;
      myfile << "Tempo criar matriz entrada (s):" << fs.count() << std::endl;
      myfile << "Tempo calcular em CPU (s):" << calculo.count() << std::endl;
      myfile << "Matriz Xproject: [" << x_projectedR.rows() << " , " << x_projectedR.cols() << " ]" << std::endl;
      myfile << "Matriz Yproject: [" << y_projectedR.rows() << " , " << y_projectedR.cols() << " ]" << std::endl;
    myfile.close();

    std::cout << "Finished with success!" << std::endl;

    return 0;
}
