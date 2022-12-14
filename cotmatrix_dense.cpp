#include <vector>

// For error printing
#include <cstdio>
#include <igl/cotmatrix_entries.h>

// Bug in unsupported/Eigen/SparseExtra needs iostream first
#include <iostream>

#include <Eigen/Dense>
#include <Eigen/Sparse>

template <typename DerivedV, typename DerivedF>
IGL_INLINE void cotmatrix_dense(
  const Eigen::MatrixBase<DerivedV> & V, 
  const Eigen::MatrixBase<DerivedF> & F, 
  Eigen::MatrixBase<Eigen::MatrixXd> & L)
{
    using namespace Eigen;
    using namespace std;
    Matrix<int,Dynamic,2> edges;
    int simplex_size = F.cols();
    // 3 for triangles, 4 for tets
    assert(simplex_size == 3 || simplex_size == 4);
    if(simplex_size == 3)
    {
        // This is important! it could decrease the comptuation time by a factor of 2
        // Laplacian for a closed 2d manifold mesh will have on average 7 entries per
        // row
        edges.resize(3,2);
        edges << 
        1,2,
        2,0,
        0,1;
    }else if(simplex_size == 4)
    {
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
    std::cout << "time to compute dense cotangents: " << time_cot.count() << " ms" << endl;
  
    // Loop over triangles
    chrono::steady_clock::time_point begin_assembly = chrono::steady_clock::now();
    
    /*
    tbb::parallel_for(tbb::blocked_range<int>(0,F.rows()), [&](tbb::blocked_range<int> r){
    for(int i = r.begin(); i < r.end(); i++)
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
    });
    */
    
    
    //#pragma omp parallel for num_threads(128)
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
    std::cout << "time to assemble dense entries: " << time_assembly.count() << " ms" << endl;
}

#ifdef IGL_STATIC_LIBRARY
// Explicit template instantiation
// generated by autoexplicit.sh
//template void cotmatrix_dense<Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, double>(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> > const&, Eigen::SparseMatrix<double, 0, int>&, Eigen::SparseMatrix<double, 0, int>&, Eigen::MatrixXd&);
// generated by autoexplicit.sh
template void cotmatrix_dense<Eigen::Matrix<double, -1, -1, 1, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 1, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::MatrixXd>&);
template void cotmatrix_dense<Eigen::Matrix<double, -1, 3, 0, -1, 3>, Eigen::Matrix<int, -1, 4, 0, -1, 4> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 3, 0, -1, 3> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, 4, 0, -1, 4> > const&, Eigen::MatrixBase<Eigen::MatrixXd>&);
template void cotmatrix_dense<Eigen::Matrix<double, -1, 3, 0, -1, 3>, Eigen::Matrix<int, -1, 3, 0, -1, 3> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 3, 0, -1, 3> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, 3, 0, -1, 3> > const&, Eigen::MatrixBase<Eigen::MatrixXd>&);
template void cotmatrix_dense<Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::MatrixXd>&);
#endif