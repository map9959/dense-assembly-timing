#include <igl/cotmatrix.h>
#include <igl/readOFF.h>
#include <igl/readSTL.h>
#include <igl/readOBJ.h>

#include <iostream>
#include <chrono>
#include <filesystem>
#include <string>

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

  if(argc != 2){
    cout << "usage: ./dense_timing [path to folder of models]" << endl;
    return 1;
  }

  // Load meshes in various formats
  const filesystem::path models_folder{argv[1]};
  for(auto const& dir_entry : filesystem::directory_iterator{models_folder}){
    const char* filename = dir_entry.path().string().c_str();
    const char* extension = filesystem::path(dir_entry).extension().string().c_str();
    FILE* fileptr = fopen(dir_entry.path().string().c_str(), "r");

    if(strcmp(extension, ".stl") == 0){
      igl::readSTL(fileptr, V, F, N);
    }else if(strcmp(extension, ".off") == 0){
      igl::readOFF(filename, V, F);
    }else if(strcmp(extension, ".obj") == 0){
      igl::readOBJ(filename, V, F);
    }
    
    // Only test small models, ignore models that can't be read
    if(V.rows() == 0 || V.rows() > 10000){
      continue;
    }

    std::cout << V.rows() << " " << std::flush;
    
    // Compute Laplace-Beltrami operator: #V by #V
    cotmatrix_timed(V,F,L);

    Ld.resize(V.rows(),V.rows());
    cotmatrix_dense(V,F,Ld);
    
    // Make sure sparse and dense computations are the same
    double diff_error = (L-Ld).norm();
    if(diff_error > pow(10.0, -10.0)){
      std::cout << "WRONG! " << diff_error;
    }

    std::cout << endl;
  }
}
