#include <igl/cotmatrix.h>
#include <igl/readOFF.h>
#include <igl/readSTL.h>
#include <igl/readOBJ.h>

#include <iostream>
#include <chrono>
#include <filesystem>
#include <string>

//#include <tbb/parallel_for.h>

#include "cotmatrix_timed.h"
#include "cotmatrix_dense.h"

Eigen::MatrixXd V;
Eigen::MatrixXi F;
Eigen::MatrixXd N;
Eigen::SparseMatrix<double> L;
Eigen::MatrixXd Ld;

int main(int argc, char *argv[])
{
  using namespace Eigen;
  using namespace std;

  // Load a mesh in OFF format
  const filesystem::path models_folder{"/mnt/hdd1/michael/small-models"};
  for(auto const& dir_entry : filesystem::directory_iterator{models_folder}){
    const char* filename = dir_entry.path().string().c_str();
    const char* extension = filesystem::path(dir_entry).extension().string().c_str();
    FILE* fileptr = fopen(dir_entry.path().string().c_str(), "r");
    //std::cout << dir_entry.path().string() << " " << std::flush;
    //std::cout << extension << " ";

    if(strcmp(extension, ".stl") == 0){
      igl::readSTL(fileptr, V, F, N);
    }else if(strcmp(extension, ".off") == 0){
      igl::readOFF(filename, V, F);
    }else if(strcmp(extension, ".obj") == 0){
      igl::readOBJ(filename, V, F);
    }
    
    if(V.rows() == 0){
      continue;
    }
    //std::cout << "results for " << filename << ".off:" << endl;
    std::cout << V.rows() << " ";
    
    // Compute Laplace-Beltrami operator: #V by #V
    chrono::steady_clock::time_point begins = chrono::steady_clock::now();
    cotmatrix_timed(V,F,L);
    chrono::steady_clock::time_point ends = chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> time_s = ends-begins;
    //std::cout << "time to compute sparse (total): " << time_s.count() << " ms" << endl << endl;
    std::cout << time_s.count() << " ";

    Ld.resize(V.rows(),V.rows());
    chrono::steady_clock::time_point begind = chrono::steady_clock::now();
    cotmatrix_dense(V,F,Ld);
    chrono::steady_clock::time_point endd = chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> time_d = endd-begind;
    //std::cout << "time to compute dense (total): " << time_d.count() << " ms" << endl;
    std::cout << time_d.count() << " ";
    //std::cout << "----" << endl;
    std::cout << endl;

    //std::cout << L.block(0,0,5,5) << endl;
    //std::cout << Ld.block(0,0,5,5) << endl;
  }
}
