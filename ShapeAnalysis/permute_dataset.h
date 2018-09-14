#pragma once
#ifndef PERMUTE_DATASET_H
#define PERMUTE_DATASET_H

#include "utils.h"
#include <igl/readOFF.h>
#include <igl/writeOFF.h>


void reversePermuatation(const std::vector<int>& permutation, std::vector<int>& reverse) {
	int n = permutation.size();
	reverse.resize(n);
	std::vector<pair<int, int> >a(n);
	
	for (int i = 0; i < n; i++) {
		// filling the original array
		a[i] = std::make_pair(permutation[i], i);
	}

	std::sort(a.begin(), a.end());

	for (int i = 0; i < n; i++) {
		reverse[i] = a[i].second;
	}
}

void randPermute0Fixed(int n, std::vector<int>& permutation, std::vector<int>& reverse) {
	permutation.resize(n);
	for (int i = 0; i < n; i++) {
		permutation[i] = i;
	}
	// fix 0
	std::random_shuffle(permutation.begin()+1, permutation.end());
	reversePermuatation(permutation, reverse);
}

void randPermute(int n, std::vector<int>& permutation, std::vector<int>& reverse) {
	permutation.resize(n);
	for (int i = 0; i < n; i++) {
		permutation[i] = i;
	}
	// fix 0
	std::random_shuffle(permutation.begin(), permutation.end());
	reversePermuatation(permutation, reverse);
}

template < typename T >
void permuteVector(const std::vector<int>& permutation, const std::vector< T >& f, std::vector< T >& f_, bool reverse=false) {
	f_.resize(f.size());
	if (reverse) {
		for (int i = 0; i < f.size(); i++) {
			f_[permutation[i]] = f[i];
		}
	}
	else {
		for (int i = 0; i < f.size(); i++) {
			f_[i] = f[permutation[i]];
		}
	}
}

void permuteMatrix(const std::vector<int>& permutation, const Eigen::MatrixXd& M, Eigen::MatrixXd& M_) {
	M_.resize(M.rows(), M.cols());
	for (int i = 0; i < M.rows(); i++) {
		M_.row(permutation[i]) = M.row(i);
	}
}

void permuteMesh(const std::vector<int>& permutation, 
	const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
	Eigen::MatrixXd& V_, Eigen::MatrixXi& F_) {
	V_.resize(V.rows(), V.cols());
	F_.resize(F.rows(), F.cols());
	for (int i = 0; i < V.rows(); i++) {
		V_.row(permutation[i]) = V.row(i);
	}
	for (int i = 0; i < F.rows(); i++) {
		F_(i, 0) = permutation[F(i, 0)];
		F_(i, 1) = permutation[F(i, 1)];
		F_(i, 2) = permutation[F(i, 2)];
	}
}

void permuteDataset(const std::string& dataset_path, 
					const std::string& permuted_dataset_path,
					const std::string& permuatations_path,
					const std::string& reverse_permutations_path) {
	// list directory
	std::vector< std::string > shapes;
	getFilesList_(dataset_path, ".off", shapes);
	std::string name;
	cout << shapes.size() << endl;

	Eigen::MatrixXd V;
	Eigen::MatrixXi F;

	for (int i = 0; i < shapes.size(); i++) {
		int nv = 0;
		if (!igl::readOFF(dataset_path + "/" + shapes[i] + ".off", V, F))
		{
			cout << "failed to load mesh" << endl;
			exit(666);
		}
		// permute the mesh and save reverse permutation
		nv = V.rows();
		std::vector<int> permutation;
		std::vector<int> reverse;
		Eigen::MatrixXd V_perm;
		Eigen::MatrixXi F_perm;
		randPermute(nv, permutation, reverse);
		save_vector_<int>(permuatations_path + "/" + shapes[i] + ".txt", permutation);
		save_vector_<int>(reverse_permutations_path + "/" + shapes[i] + ".txt", reverse);
		permuteMesh(permutation, V, F, V_perm, F_perm);
		igl::writeOFF(permuted_dataset_path + "/" + shapes[i] + ".off", V_perm, F_perm);
	}
}


void permuteLabels(const std::string& labels_path, const std::string& permuted_labels_path,
				   const std::string& permutations_path) {
	// list permutations dir
	// list directory
	std::vector< std::string > names;
	getFilesList_(permutations_path, ".txt", names);
	std::vector<int> labels;
	std::vector<int> permuted_labels;
	std::vector<int> permutation;
	for (int i = 0; i < names.size(); i++) {
		// permute labels 
		load_vector_<int>(labels_path + "/" + names[i] + ".txt", labels);
		load_vector_<int>(permutations_path + "/" + names[i] + ".txt", permutation);
		permuteVector<int>(permutation, labels, permuted_labels, true);
		save_vector_<int>(permuted_labels_path + "/" + names[i] + ".txt", permuted_labels);
	}
}

void permuteFunc(const Eigen::MatrixXd& f, const std::string& permuted_funcs_path, 
				 const std::string& permutations_path) {
	// list permutations dir
	// list directory
	std::vector< std::string > names;
	getFilesList_(permutations_path, ".txt", names);
	Eigen::MatrixXd f_perm;
	std::vector<int> permutation;
	for (int i = 0; i < names.size(); i++) {
		// permute labels 
		load_vector_<int>(permutations_path + "/" + names[i] + ".txt", permutation);
		permuteMatrix(permutation, f, f_perm);
		save_matrix_(permuted_funcs_path + "/" + names[i] + ".txt", f_perm);
	}
}

void refShapePermuteXYZ(const std::string& path, const std::string& permuted_xyz_path,
	const std::string& permutations_path) {
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	if (!igl::readOFF(path, V, F))
	{
		cout << "failed to load mesh" << endl;
		exit(666);
	}
	permuteFunc(V, permuted_xyz_path, permutations_path);
}


#endif // !PERMUTE_DATASET_H
