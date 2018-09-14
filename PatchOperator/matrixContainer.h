#pragma once
#ifndef MATRIX_CONTAINER_H
#define MATRIX_CONTAINER_H
#include<Eigen/Dense>
#include<Eigen/StdVector>

// just a basic container for block matrices in eigen
template <typename T, int n, int m>
class block_matrix {
public :
	block_matrix(): rows_(0), cols_(0), v(NULL){}
	~block_matrix() {
		if (v != NULL) {
			delete[] v;
		}
	}
	/*void resize(int i, int j) {
		Mat.resize(i*n, j*m);
		rows_ = i;
		cols_ = j;
	}
	void setConstant(const Eigen::Matrix< T, n, m >& M) {
		for (int i = 0; i < M.rows(); i++) {
			for (int j = 0; j < M.cols(); j++) {
				Mat.block<n, m>(i*n, j*m) = M;
			}
		}
	}
	const Eigen::Matrix< T, n, m >& operator()(int i, int j) const {
		return Mat.block<n, m>(i*n, j*m);
	}
	Eigen::Matrix< T, n, m >& operator()(int i, int j) {
		return Mat.block<n, m>(i*n, j*m);
	}*/
	void resize(unsigned int i, unsigned int j) {
		if (v != NULL) {
			delete[] v;
		}
		v = new Eigen::Matrix<T, n, m>[i*j];
		rows_ = i;
		cols_ = j;
		//v.resize(i*j);
	}
	void setConstant(const Eigen::Matrix< T, n, m >& cst) {
		for (int i = 0; i < rows_*cols_; i++) {
			v[i] = cst;
		}
	}
	void setConstant(const T& cst) {
		for (int i = 0; i < rows_*cols_; i++) {
			v[i].setConstant(cst);
		}
	}

	Eigen::Matrix< T, n, m >& operator() (int i, int j) {
		return v[i * cols_ + j];
	}
	const Eigen::Matrix< T, n, m >& operator() (int i, int j) const {
		return v[i * cols_ + j];
	}
	unsigned int rows() const { return rows_; }
	unsigned int cols() const { return cols_; }
private:
	//Eigen::Matrix< T, -1, -1> Mat;
	Eigen::Matrix<T, n, m>* v;
	//std::vector < Eigen::Matrix< T, n, m >, Eigen::aligned_allocator<Eigen::Matrix< T, n, m > > v;
	unsigned int rows_;
	unsigned int cols_;
};




#endif