#pragma once
#ifndef LOCAL_PCA_H
#define LOCAL_PCA_H

#include <nanoflann.hpp>
#include <Eigen/Dense>
#include "utils.h"
#include <algorithm>    // std::sort
#include "visualize.h"
#include "kdtree3d.h"

using namespace std;
using namespace nanoflann;

// local PCA patterns
#define LINE 0
#define PLATE 1
#define SPHERE 2
#define STRIP 3

template <typename num_t>
inline void barycenter(const Eigen::Matrix<num_t, -1, -1>& V, const std::vector<int>& idx, Eigen::Matrix<num_t, 3, 1>& b) {
	b.setZero();
	int nv = idx.size();
	for (int i = 0; i < nv; i++) {
		b += V.row(idx[i]).transpose();
	}
	b /= nv;
}



template <typename num_t>
inline void PCA_3D(const Eigen::Matrix<num_t, -1, -1>& V, const Eigen::Matrix<num_t, 3, 1>& c, const std::vector<int>& idx,
	Eigen::Vector3d& eigs, Eigen::Matrix3d& eigvec) {
	// compute covariance matrix
	Eigen::Matrix3d M;
	M.setZero();
	int nv = idx.size();
	Eigen::Vector3d p;
	
	int j = 0;
	for (int i = 0; i < nv; i++) {
		j = idx[i];
		p(0) = V(j, 0) - c(0);
		p(1) = V(j, 1) - c(1);
		p(2) = V(j, 2) - c(2);	
		M += p*(p.transpose());
	}
	M / nv;
	Eigs3D(M, eigs, eigvec);
}

template <typename num_t>
inline void PCA_3D(const Eigen::Matrix<num_t, -1, -1>& V, const std::vector<int>& idx,
	Eigen::Vector3d& eigs, Eigen::Matrix3d& eigvec) {
	Eigen::Matrix<num_t, 3, 1> c;
	barycenter(V, idx, c);
	PCA_3D(V, c, idx, eigs, eigvec);
}


// compute a "probability" for each local pattern
inline void PcaPattern(const Eigen::Vector3d& eigs, Eigen::Vector4d& p) {

	std::vector<double> l(3);

	l[0] = fabs(eigs(0));
	l[1] = fabs(eigs(1));
	l[2] = fabs(eigs(2));
	
	// sort in descending order
	std::sort(l.begin(), l.end());
	std::swap(l[0], l[2]);

	double mu_l = (l[0] - l[1]) / (l[0] + l[1] + l[2]);
	double mu_p = 2 * (l[1] - l[2]) / (l[0] + l[1] + l[2]);
	double mu_s = 3 * l[2] / (l[0] + l[1] + l[2]);

	mu_l = std::min(1.0, mu_l);
	mu_p = std::min(1.0, mu_p);
	mu_s = std::min(1.0, mu_s);
	double mu_st = (1. - mu_l)*(1. - mu_p)*(1. - mu_s);

	p(LINE) = mu_l;
	p(PLATE) = mu_p;
	p(SPHERE) = mu_s;
	p(STRIP) = mu_st;
}






template <typename num_t>
inline void localDir3d(const Eigen::Matrix<num_t, -1, -1>& V, const Eigen::Matrix<num_t, 3, 1>& x, Eigen::Matrix<num_t, 3, 1>& v) {
	barycenter(V, v);
	v -= x;
	v.normalize();
}

template <typename num_t>
inline void localDir3d(const Eigen::Matrix<num_t, 3, 1>& barycenter, const Eigen::Matrix<num_t, 3, 1>& x, Eigen::Matrix<num_t, 3, 1>& v) {
	barycenter -= x;
	v.normalize();
}




void computeLocalPcaPaterns(const Eigen::MatrixXd& V, double radius, Eigen::MatrixXd& p) {
	p.resize(V.rows(), 4);
	kdtree3d<double> kd(V);
	int nb_nn = 0;
	std::vector<int> idx;
	Eigen::Vector3d eigs;
	Eigen::Matrix3d eigvec;
	Eigen::Vector4d p_;
	for (int i = 0; i < V.rows(); i++) {
		nb_nn = kd.radiusSearch(V.row(i).transpose(), radius, idx);
		PCA_3D(V, idx, eigs, eigvec);
		PcaPattern(eigs, p_);
		p.row(i) = p_.transpose();
	}
}

void test_local_pca_paterns(const std::string& path, int patern, double radius) {
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	if (!igl::readOFF(path, V, F))
	{
		cout << "failed to load mesh" << endl;
		exit(666);
	}
	std::vector<double> f(V.rows());
	Eigen::MatrixXd p;
	computeLocalPcaPaterns(V, radius, p);
	for (int i = 0; i < V.rows(); i++) {
		f[i] = p(i, patern);
	}
	visualize(V, F, f);
}




/*
template <typename num_t>
class LocalPCA {
public:
	LocalPCA(const Eigen::Matrix<num_t, -1, -1>& V_) {
		// copy point cloud
		V.resize(V_.rows(), V_.cols());
		for (int i = 0; i < V.rows(); i++) {
			V.row(i) = V_.row(i);
		}

		// build kd-tree
		mat_index = new kd_tree_d(V, 10); // max leaf = 10
		mat_index->index->buildIndex();
	}
	LocalPCA() {
		delete mat_index;
	}

	






	void getNN(double theta, double r, std::vector<std::size_t>& index, std::vector<double>& dists_sqr) const {
		// do a knn search
		int nb_neighs = index.size();

		std::vector<double> query_pt(2);
		query_pt[0] = r*cos(theta);
		query_pt[1] = r*sin(theta);

		//std::vector<double> out_dists_sqr(nb_neighs);

		nanoflann::KNNResultSet<double> resultSet(nb_neighs);

		resultSet.init(&index[0], &dists_sqr[0]);
		mat_index->index->findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));

		std::cout << "knnSearch(nn=" << num_results << "): \n";
		for (size_t i = 0; i<num_results; i++)
		std::cout << "ret_index[" << i << "]=" << ret_indexes[i] << " out_dist_sqr=" << out_dists_sqr[i] << endl;

	}
private:
	Eigen::Matrix<num_t, -1, -1> V;
	kd_tree_d* mat_index;
};
*/

#endif // !LOCAL_PCA_H
