// The MIT License (MIT)
//
// Copyright (c) 2016 Northeastern University
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Kernel Dimension Alternative Clustering (KDAC)
// Please refer to the paper published in PAMI by Liu, Dy and Jordan at:
// http://people.eecs.berkeley.edu/~jordan/papers/niu-dy-jordan-pami.pdf
// We try to follow naming conventions in the paper as much as possible.
// The lower cased variable names is the same as in the paper, and the
// upper cased matrix variable names in the paper are converted to lower
// case suffixed with "_matrix". For example:
// matrix U in the paper is named u_matrix in this implementation.

#ifndef CPP_INCLUDE_KDAC_H
#define CPP_INCLUDE_KDAC_H

#include <functional>
#include <vector>
#include <cmath>
#include "include/matrix.h"
#include "include/vector.h"
#include "include/cpu_operations.h"
#include "include/svd_solver.h"
#include "include/kmeans.h"
#include "include/spectral_clustering.h"
#include "Eigen/Core"
#include "include/util.h"
#include "include/kernel_types.h"


namespace Nice {
template<typename T>
class KDAC {
 public:
  /// This is the default constructor for KDAC
  /// Number of clusters c and reduced dimension q will be both set to 2
  KDAC() {
    SetC(2);
    SetQ(2);
    SetKernel(kGaussianKernel, 1.0);
  }

  /// Set the number of clusters c
  void SetC(int c) {
    c_ = c;
    CheckCQ();
  }

  /// Set the reduced dimension q
  void SetQ(int q) {
    q_ = q;
    CheckCQ();
  }

  Matrix<T> GetU(void) {
    return u_matrix_;
  }

  Matrix<T> GetUNormalized(void) {
    return u_matrix_normalized_;
  }

  Matrix<T> GetL(void) {
    return l_matrix_;
  }

  Matrix<T> GetD(void) {
    return d_matrix_;
  }

  Matrix<T> GetDToTheMinusHalf(void) {
    return d_matrix_to_the_minus_half_;
  }

  Matrix<T> GetK(void) {
    return k_matrix_;
  }

  /// Set the kernel type: kGaussianKernel, kPolynomialKernel, kLinearKernel
  /// And set the constant associated the kernel
  void SetKernel(KernelType kernel_type, float constant) {
    kernel_type_ = kernel_type;
    constant_ = constant;
  }

  /// This function creates the first clustering result
  /// \param input_matrix
  /// The input matrix of n samples and d features where each row
  /// represents a sample
  /// \return
  /// It only generates the clustering result but does not returns it
  /// Users can use Predict() to get the clustering result returned
  void Fit(const Matrix<T> &input_matrix) {
    // Following the pseudo code in Algorithm 1 in the paper
    Init(input_matrix);
    OptimizeU();
    OptimizeW();
  }

  /// This function creates an alternative clustering result
  /// Must be called after \ref Fit(const Matrix<T> &input_matrix)
  /// when the first clustering result is generated
  void Fit(void) {

  }

  /// Running Predict() after Fit() returns
  /// the current clustering result as a Vector of T
  /// \return
  /// A NICE vector of T that specifies the clustering result
  Vector<T> Predict(void) {

  }


 private:
  int c_;  // cluster number c
  int q_;  // reduced dimension q
  int n_;  // number of samples in input data X
  int d_;  // input data X dimension d
  KernelType kernel_type_;  // The kernel type of the kernel matrix
  float constant_;  // In Gaussian kernel, this is sigma;
                    // In Polynomial kernel, this is the polynomial order
                    // In Linear kernel, this is c as well
  bool u_converge;  // If matrix U reaches convergence, false by default
  bool v_converge;  // If matrix V reaches convergence, false by default
  Matrix<T> x_matrix_;  // Input matrix X (n by d)
  Matrix<T> w_matrix_;  // Transformation matrix W (d by q). Initialized to I
  Matrix<bool> y_matrix_;  // Labeling matrix Y (n by (c0 + c1 + c2 + ..))
  Matrix<T> d_matrix_;  // Diagonal degree matrix D (n by n)
  Matrix<T> d_matrix_to_the_minus_half_;  // D^(-1/2) matrix
  Matrix<T> k_matrix_;  // Kernel matrix K (n by n)
  Matrix<T> u_matrix_;  // Embedding matrix U (n by c)
  Matrix<T> u_matrix_normalized_;  // Row-wise normalized U
  Matrix<T> l_matrix_;  // D^(-1/2) * K * D^(-1/2)
  Matrix<T> h_matrix_;  // Centering matrix (n by n)
  Vector<T> clustering_result;  // Current clustering result


  // Initialization
  void Init(const Matrix<T> &input_matrix) {
    x_matrix_ = input_matrix;
    n_ = input_matrix.rows();
    d_ = input_matrix.cols();
    w_matrix_ = Matrix<T>::Identity(d_, d_);
//    y_matrix_ = Matrix<bool>::Zero(n_, c_);
//    d_matrix_ = Matrix<T>::Zero(n_, n_, 0);
//    k_matrix_ = Matrix<T>::Zero(n_, n_, 0);
//    u_matrix_ = Matrix<T>::Zero(n_, c_, 0);
    h_matrix_ = Matrix<T>::Identity(n_, n_)
        - Matrix<T>::Constant(n_, n_, 1) / float(n_);
    clustering_result = Vector<T>::Zero(n_);
    u_converge = false;
    v_converge = false;
  }

  // Check if q is not bigger than c
  void CheckCQ() {
    if (q_ > c_) {
      std::cerr <<
          "Reduced dimension q cannot exceed cluster number c" << std::endl;
      exit(1);
    }
  }

  void OptimizeW(void) {

  }

  void OptimizeU(void) {
    // Projects X to subspace W (n * d to n * q)
    // If this is the first round, then projected X equals to X
    Matrix<T> projected_x_matrix = x_matrix_ * w_matrix_;
    // Generate the kernel matrix based on kernel type from projected X
    k_matrix_ = CpuOperations<T>::GenKernelMatrix(
        projected_x_matrix, kernel_type_, constant_);
    // Generate degree matrix from the kernel matrix
    // d_i is the diagonal vector of degree matrix D

    // This is a reference to how to directly generate D^(-1/2)
    // Vector<T> d_i = k_matrix_.rowwise().sum().array().sqrt().unaryExpr(
    //     std::ptr_fun(util::reciprocal<T>));
    // d_matrix_ = d_i.asDiagonal();

    // Generate D and D^(-1/2)
    CpuOperations<T>::GenDegreeMatrix(
        k_matrix_, d_matrix_, d_matrix_to_the_minus_half_);
    l_matrix_ = d_matrix_to_the_minus_half_ * k_matrix_ *
        d_matrix_to_the_minus_half_;
    SvdSolver<T> solver;
    solver.Compute(l_matrix_);
    // Generate a u matrix from SVD solver and then use Normalize to normalize
    // its rows
    u_matrix_ = solver.MatrixU().leftCols(c_);
    u_matrix_normalized_ = CpuOperations<T>::Normalize(u_matrix_, 2, 1);
  }

};
}  // namespace NICE

#endif  // CPP_INCLUDE_KDAC_H