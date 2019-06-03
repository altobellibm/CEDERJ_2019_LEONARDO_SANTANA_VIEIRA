#include <iostream>
#include <chrono>

#undef NDEBUG
#include <assert.h>

#include <fstream>

#define EIGEN_RUNTIME_NO_MALLOC // Define this symbol to enable runtime tests for allocations

#include "dual_scaling.h"
#include "leitorbasenumerica.h"

// --- Timing includes
#include "TimingCPU.h"

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::milliseconds ms;
typedef std::chrono::duration<float> fsec;

const double LIMITE_INFERIOR_DELTA = 1e-8;
typedef double real_type;

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

    TimingCPU timer_CPU;
    timer_CPU.StartCounter();

    std::string strFile = argv[1];
    std::string strPrefixFile = strFile.substr(strFile.find_last_of("/")+1,strFile.find("."));

    // dominance_data
    dynamic_size_matrix x_normedR, x_projectedR;
    dynamic_size_matrix y_normedR, y_projectedR;
    row_vector_type rhoR, deltaR;

    auto baseDominance = getMatrix(strFile);
    auto matrix = ro(baseDominance.getMatrix());

    std::cout << "Started Dual Scaling " << endl;
    ds::dual_scaling(matrix, x_normedR, y_normedR, x_projectedR, y_projectedR, rhoR, deltaR);

    std::ofstream myfile;
      myfile.open ("./output/" + strPrefixFile + "_dominance_tempoProcessamentoCPU.txt");
      myfile << "Nome arquivo: " << strFile << std::endl;
      myfile << "Matriz formato: [" << baseDominance.getSizeTransation() << "," << baseDominance.getSizeColumn() << "]" << std::endl;
      myfile << "Tempo calcular em CPU (ms):" <<  timer_CPU.GetCounter() << std::endl;
      myfile << "Matriz Xproject: [" << x_projectedR.rows() << " , " << x_projectedR.cols() << " ]" << std::endl;
    myfile.close();

    std::cout << "Finished with success!" << std::endl;

    return 0;
}
