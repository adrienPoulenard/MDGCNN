#pragma once
#ifndef KDTREE_H
#define KDTREE_H

#include <nanoflann.hpp>
#include <Eigen/Dense>
#include <algorithm>    // std::sort



using namespace std;
using namespace nanoflann;

template <typename num_t>
void kdtree_demo(const size_t nSamples, const size_t dim)
{
	Eigen::Matrix<num_t, Dynamic, Dynamic>  mat(nSamples, dim);

	const num_t max_range = 20;

	// Generate points:
	generateRandomPointCloud(mat, nSamples, dim, max_range);

	//	cout << mat << endl;

	// Query point:
	std::vector<num_t> query_pt(dim);
	for (size_t d = 0; d < dim; d++)
		query_pt[d] = max_range * (rand() % 1000) / num_t(1000);


	// ------------------------------------------------------------
	// construct a kd-tree index:
	//    Some of the different possibilities (uncomment just one)
	// ------------------------------------------------------------
	// Dimensionality set at run-time (default: L2)
	typedef KDTreeEigenMatrixAdaptor< Eigen::Matrix<num_t, Dynamic, Dynamic> >  my_kd_tree_t;

	// Dimensionality set at compile-time
	//	typedef KDTreeEigenMatrixAdaptor< Eigen::Matrix<num_t,Dynamic,Dynamic> >  my_kd_tree_t;

	// Dimensionality set at compile-time: Explicit selection of the distance metric: L2
	//	typedef KDTreeEigenMatrixAdaptor< Eigen::Matrix<num_t,Dynamic,Dynamic>,nanoflann::metric_L2>  my_kd_tree_t;

	// Dimensionality set at compile-time: Explicit selection of the distance metric: L2_simple
	//	typedef KDTreeEigenMatrixAdaptor< Eigen::Matrix<num_t,Dynamic,Dynamic>,nanoflann::metric_L2_Simple>  my_kd_tree_t;

	// Dimensionality set at compile-time: Explicit selection of the distance metric: L1
	//	typedef KDTreeEigenMatrixAdaptor< Eigen::Matrix<num_t,Dynamic,Dynamic>,nanoflann::metric_L1>  my_kd_tree_t;

	my_kd_tree_t   mat_index(mat, 10 /* max leaf */);
	mat_index.index->buildIndex();

	// do a knn search
	const size_t num_results = 3;
	vector<size_t>   ret_indexes(num_results);
	vector<num_t> out_dists_sqr(num_results);

	nanoflann::KNNResultSet<num_t> resultSet(num_results);

	resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);
	mat_index.index->findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));

	std::cout << "knnSearch(nn=" << num_results << "): \n";
	for (size_t i = 0; i<num_results; i++)
		std::cout << "ret_index[" << i << "]=" << ret_indexes[i] << " out_dist_sqr=" << out_dists_sqr[i] << endl;

}

template <typename num_t>
class kdtree3d {
public:
	typedef nanoflann::KDTreeEigenMatrixAdaptor< Eigen::Matrix<num_t, -1, -1> > kd_tree_d;
	kdtree3d(const Eigen::Matrix<num_t, -1, -1>& V_) {
		// copy point cloud
		V.resize(V_.rows(), V_.cols());
		for (int i = 0; i < V.rows(); i++) {
			V.row(i) = V_.row(i);
		}

		// build kd-tree
		mat_index = new kd_tree_d(V, 10); // max leaf = 10
		mat_index->index->buildIndex();
		query_pt.resize(3);
	}
	kdtree3d() {
		delete mat_index;
	}

	/*size_t radiusSearch(const Eigen::Matrix<num_t, 3, 1>& v, num_t radius, std::vector<std::pair<size_t, num_t> >& ret_matches) {
	//const num_t search_radius = static_cast<num_t>(0.1);

	nanoflann::SearchParams params;
	//params.sorted = false;

	return mat_index->index.radiusSearch(&query_pt[0], radius, ret_matches, params);
	}*/

	void knnSearch(const Eigen::Matrix<num_t, 3, 1>& v, int k,
		std::vector<size_t>& ret_indexes,
		std::vector<num_t>& out_dists_sqr) {

		query_pt[0] = v(0);
		query_pt[1] = v(1);
		query_pt[2] = v(2);

		ret_indexes.resize(k);
		out_dists_sqr.resize(k);

		nanoflann::KNNResultSet<num_t> resultSet(k);

		resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);
		mat_index->index->findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));
	}

	num_t distToNN(const Eigen::Matrix<num_t, 3, 1>& v) {
		std::vector<size_t> ret_indexes;
		std::vector<num_t> out_dists_sqr;
		this->knnSearch(v, 1, ret_indexes, out_dists_sqr);
		return sqrt(out_dists_sqr[0]);
	}


	size_t radiusSearch(const Eigen::Matrix<num_t, 3, 1>& v, num_t radius, std::vector<int>& idx, std::vector<num_t>& dists) {
		//const num_t search_radius = static_cast<num_t>(0.1);

		nanoflann::SearchParams params;
		//params.sorted = false;
		std::vector<std::pair<int, num_t> > ret_matches;
		//std::vector<num_t> query_pt(3);
		query_pt[0] = v(0);
		query_pt[1] = v(1);
		query_pt[2] = v(2);

		
		size_t nb_nn = mat_index->index->radiusSearch(&query_pt[0], radius*radius, ret_matches, params);
		

		idx.resize(nb_nn);
		dists.resize(nb_nn);
		for (int i = 0; i < nb_nn; i++) {
			idx[i] = ret_matches[i].first;
			dists[i] = sqrt(ret_matches[i].second);
		}
		return nb_nn;
	}

	size_t radiusSearch(const Eigen::Matrix<num_t, 3, 1>& v, num_t radius, std::vector<int>& idx) {
		std::vector<num_t> dists;
		this->radiusSearch(v, radius, idx, dists);
	}

	void radiusSearch(int i, num_t radius) {
		Eigen::Vector<num_t, 3> v = V.row(i).transpose();
		this->radiusSearch(v, radius);
	}

private:
	Eigen::Matrix<num_t, -1, -1> V;
	kd_tree_d* mat_index;
	std::vector< num_t > query_pt;
};

/*void test_kd_tree(const std::string& path, int v_id, double radius) {
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	if (!igl::readOFF(path, V, F))
	{
		cout << "failed to load mesh" << endl;
		exit(666);
	}
	std::vector<double> f(V.rows());
	std::fill(f.begin(), f.end(), 0.);
	kdtree3d<double> kd(V);
	std::vector<int> idx;
	kd.radiusSearch(V.row(v_id).transpose(), radius, idx);

	for (int i = 0; i < idx.size(); i++) {
		f[idx[i]] = 1;
	}
	visualize(V, F, f);
}*/

#endif // !KDTREE_H
