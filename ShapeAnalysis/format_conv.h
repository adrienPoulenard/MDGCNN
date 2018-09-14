#pragma once
#ifndef FORMAT_CONV_H
#define FORMAT_CONV_H

#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <Eigen/Dense>
#include <igl\per_vertex_normals.h>
#include <igl\readOFF.h>
#include "utils.h"
#include <algorithm>
#include <random>




using namespace std;

void convert_to_labelized_pointcloud_normals(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, 
	const std::vector<double>& labels, Eigen::MatrixXd& LPCN, const std::vector<int>& perm) {
	int nv = V.rows();
	LPCN.resize(nv, 7);
	Eigen::MatrixXd N;
	//N.setZero(nv, 3);
	igl::per_vertex_normals(V, F, N);
	int j = 0;
	for (int i = 0; i < nv; i++) {
		j = perm[i];
		LPCN(i, 0) = V(j, 0);
		LPCN(i, 1) = V(j, 1);
		LPCN(i, 2) = V(j, 2);

		LPCN(i, 3) = N(j, 0);
		LPCN(i, 4) = N(j, 1);
		LPCN(i, 5) = N(j, 2);

		LPCN(i, 6) = labels[j];
	}
}

void random_perm(int nv, std::vector<int>& perm) {
	perm.resize(nv);
	for (int i = 0; i < nv; i++) {
		perm[i] = i;
	}
	auto rng = std::default_random_engine{};
	std::shuffle(std::begin(perm), std::end(perm), rng);
}
void convert_to_labelized_pointcloud_normals(const std::string& src, const std::string& labels_path,
	const std::string& tar, bool shuffle = true) {

	std::vector< std::string > names;
	getFilesList_(src, ".off", names);
	std::vector<double> labels;
	Eigen::MatrixXd LPCN;
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	std::vector<int> perm;

	int nv = 0;

	for (int i = 0; i < names.size(); i++) {
		display_progress(float(i) / float(names.size()));
		// permute labels 
		load_vector_<double>(labels_path + "/" + names[i] + ".txt", labels);
		if (!igl::readOFF(src + "/" + names[i] + ".off", V, F))
		{
			cout << "failed to load mesh" << endl;
			exit(666);
		}
		nv = V.rows();
		//if (shuffle) {
			random_perm(nv, perm);
		//}
		convert_to_labelized_pointcloud_normals(V, F, labels, LPCN, perm);
		save_matrix_(tar + "/" + names[i] + ".txt", LPCN);
	}
}

void convert_to_labelized_desc(const Eigen::MatrixXd& desc, const std::vector<double>& labels, 
	Eigen::MatrixXd& D) {
	int nv = labels.size();
	D.resize(desc.rows(), desc.cols() + 1);
	for (int j = 0; j < desc.cols(); j++) {
		for (int i = 0; i < desc.rows(); i++) {
			D(i, j) = desc(i, j);
		}
	}
	for (int i = 0; i < desc.rows(); i++) {
		D(i, desc.cols()) = labels[i];
	}
}

void convert_to_labelized_desc(const std::string& descs_path, int nv, int n_descs, const std::string& labels_path, const std::string& tar ) {
	std::vector< std::string > names;
	getFilesList_(descs_path, ".txt", names);
	std::vector<double> labels;
	Eigen::MatrixXd descs(nv, n_descs);
	descs.setZero();
	Eigen::MatrixXd D;


	for (int i = 0; i < names.size(); i++) {
		display_progress(float(i) / float(names.size()));
		// permute labels 
		load_vector_<double>(labels_path + "/" + names[i] + ".txt", labels);
		load_matrix_(descs_path + "/" + names[i] + ".txt", descs);
		//cout << descs_path + "/" + names[i] + ".txt" << endl;
		convert_to_labelized_desc(descs, labels, D);
		save_matrix_(tar + "/" + names[i] + ".txt", D);
	}
}

void pointnet_dataset_split(const std::string& shapes_list_path, const std::string& out_file, 
	const std::string& shapes_folder) {
	std::ifstream shapes_list(shapes_list_path);
	std::ofstream output(out_file);
	std::string shape_name;
	output << "[" ;
	while (std::getline(shapes_list, shape_name))
	{
		output << "\"" << "shape_data/" << shapes_folder << "/" << shape_name << "\"" << ", ";
	}
	output << "]";
	shapes_list.close();
	output.close();
}

#endif