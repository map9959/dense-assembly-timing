#include <igl/cotmatrix.h>
#include <igl/readOFF.h>

#include <iostream>
#include <chrono>

#include <tbb/parallel_for.h>

#include "cotmatrix_timed.h"
#include "cotmatrix_dense.h"

Eigen::MatrixXd V;
Eigen::MatrixXi F;
Eigen::SparseMatrix<double> L;
Eigen::MatrixXd Ld;

int main(int argc, char *argv[])
{
  using namespace Eigen;
  using namespace std;

  // Load a mesh in OFF format
  //string filename = "cow.off";
  string files[] = {"cow", "cube", "lion", "screwdriver", "beetle"};
  for(string filename : files){
    igl::readOFF("../data/" + filename + ".off", V, F);
    std::cout << "results for " << filename << ".off:" << endl;
    
    // Compute Laplace-Beltrami operator: #V by #V
    chrono::steady_clock::time_point begins = chrono::steady_clock::now();
    cotmatrix_timed(V,F,L);
    chrono::steady_clock::time_point ends = chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> time_s = ends-begins;
    std::cout << "time to compute sparse (total): " << time_s.count() << " ms" << endl << endl;

    Ld.resize(V.rows(),V.rows());
    chrono::steady_clock::time_point begind = chrono::steady_clock::now();
    cotmatrix_dense(V,F,Ld);
    chrono::steady_clock::time_point endd = chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> time_d = endd-begind;
    std::cout << "time to compute dense (total): " << time_d.count() << " ms" << endl;
    std::cout << "----" << endl;

    //std::cout << L.block(0,0,5,5) << endl;
    //std::cout << Ld.block(0,0,5,5) << endl;
  }
}
