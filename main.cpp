#include <igl/barycenter.h>
#include <igl/cotmatrix.h>
#include <igl/cotmatrix_entries.h>
#include <igl/doublearea.h>
#include <igl/grad.h>
#include <igl/jet.h>
#include <igl/massmatrix.h>
#include <igl/per_vertex_normals.h>
#include <igl/readDMAT.h>
#include <igl/readOFF.h>
#include <igl/repdiag.h>

#include <iostream>
#include <chrono>

#include "cotmatrix_timed.h"

Eigen::MatrixXd V,U;
Eigen::MatrixXi F;
Eigen::SparseMatrix<double> L;
Eigen::MatrixXd Ld;

void cotmatrix_dense(
  const Eigen::MatrixXd & V, 
  const Eigen::MatrixXi & F, 
  Eigen::MatrixXd& L)
{
  using namespace Eigen;
  using namespace std;

  //L.resize(V.rows(),V.rows());
  Matrix<int,Dynamic,2> edges;
  int simplex_size = F.cols();
  // 3 for triangles, 4 for tets
  assert(simplex_size == 3 || simplex_size == 4);
  if(simplex_size == 3)
  {
    // This is important! it could decrease the comptuation time by a factor of 2
    // Laplacian for a closed 2d manifold mesh will have on average 7 entries per
    // row
    //L.reserve(10*V.rows());
    edges.resize(3,2);
    edges << 
      1,2,
      2,0,
      0,1;
  }else if(simplex_size == 4)
  {
    //L.reserve(17*V.rows());
    edges.resize(6,2);
    edges << 
      1,2,
      2,0,
      0,1,
      3,0,
      3,1,
      3,2;
  }else
  {
    return;
  }
  // Gather cotangents
  MatrixXd C;
  chrono::steady_clock::time_point begin_cot = chrono::steady_clock::now();
  igl::cotmatrix_entries(V,F,C);
  chrono::steady_clock::time_point end_cot = chrono::steady_clock::now();
  std::chrono::duration<double, std::milli> time_cot = end_cot-begin_cot;
  cout << "time to compute dense cotangents: " << time_cot.count() << " ms" << endl;
  
  //vector<Triplet<Scalar> > IJV;
  //IJV.reserve(F.rows()*edges.rows()*4);
  // Loop over triangles
  //#pragma openmp parallel for num_threads(16)
  chrono::steady_clock::time_point begin_assembly = chrono::steady_clock::now();
  for(int i = 0; i < F.rows(); i++)
  {
    // loop over edges of element
    for(int e = 0;e<edges.rows();e++)
    {
      int source = F(i,edges(e,0));
      int dest = F(i,edges(e,1));
      L(source, dest) += C(i,e);
      L(dest, source) += C(i,e);
      L(source, source) -= C(i,e);
      L(dest, dest) -= C(i,e);
    }
  }
  chrono::steady_clock::time_point end_assembly = chrono::steady_clock::now();
  std::chrono::duration<double, std::milli> time_assembly = end_assembly-begin_assembly;
  cout << "time to assemble dense entries: " << time_assembly.count() << " ms" << endl;

}

int main(int argc, char *argv[])
{
  using namespace Eigen;
  using namespace std;

  // Load a mesh in OFF format
  igl::readOFF("./data/cow.off", V, F);

  // Compute Laplace-Beltrami operator: #V by #V
  chrono::steady_clock::time_point begins = chrono::steady_clock::now();
  cotmatrix_timed(V,F,L);
  chrono::steady_clock::time_point ends = chrono::steady_clock::now();
  std::chrono::duration<double, std::milli> time_s = ends-begins;

  //cout << "time to compute sparse: " << chrono::duration_cast<chrono::nanoseconds>(ends-begins).count() << " ns" << endl;
  cout << "time to compute sparse (total): " << time_s.count() << " ms" << endl;

  Ld.resize(V.rows(),V.rows());
  chrono::steady_clock::time_point begind = chrono::steady_clock::now();
  cotmatrix_dense(V,F,Ld);
  chrono::steady_clock::time_point endd = chrono::steady_clock::now();
  std::chrono::duration<double, std::milli> time_d = endd-begind;
  cout << "time to compute dense (total): " << time_d.count() << " ms" << endl;
  //cout << "time to compute dense: " << chrono::duration_cast<chrono::nanoseconds>(endd-begind).count() << " ns" << endl;

}
