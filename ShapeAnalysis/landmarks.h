#pragma once
#ifndef LANDMARKS_H
#define LANDMARKS_H
#include <vector>
#include <Eigen/Dense>
#include "utils.h"
#include <igl/readOFF.h>
#include "visualize.h"

void landmarks(const std::vector<int>& idx, double radius, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::MatrixXd& L) {
	L.resize(V.rows(), idx.size());
	L.setZero();
	int np = 0;
	double d = 0.0;
	for (int i = 0; i < idx.size(); i++) {
		np = 0;
		for (int j = 0; j < V.rows(); j++) {
			d = (V.row(j) - V.row(idx[i])).norm();
			if (d < radius) {
				np++;
				L(j, i) = 1.0;
			}
		}
		//cout << "np: " << np << endl;

	}
}

void landmarks(const std::vector<int>& idx, double radius, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const std::string& path) {
	Eigen::MatrixXd L;
	landmarks(idx, radius, V, F, L);
	save_matrix_(path, L);
}

bool landmarks(const std::vector<int>& idx, double radius, const std::string& shape, const std::string& path) {
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	if (!igl::readOFF(shape, V, F))
	{
		cout << "failed to load mesh" << endl;
		return false;
	}
	landmarks(idx, radius, V, F, path);

	return true;
}

bool visualize_landmarks(std::vector<int> idx, double radius, const std::string& shape) {
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	if (!igl::readOFF(shape, V, F))
	{
		cout << "failed to load mesh" << endl;
		return false;
	}
	Eigen::MatrixXd L;
	landmarks(idx, radius, V, F, L);
	Eigen::MatrixXd C(L.rows(), 1);
	C.setZero();
	for (int i = 0; i < C.rows(); i++) {
		for (int j = 0; j < L.cols(); j++) {
			if (L(i, j) > 0.5) {
				C(i, 0) = 1.0;
			}
		}
	}
	visualize(V, F, C);
	return true;
}

void landmarks_on_dataset(const std::vector<int>& idx, double radius, const std::string& src_path, const std::string& tar_path) {
	std::vector< std::string > names;
	getFilesList_(src_path, ".off", names);
	for (int i = 0; i < names.size(); i++) {
		// permute labels
		display_progress(float(i) / names.size());
		landmarks(idx, radius, src_path + "/" + names[i] + ".off", tar_path + "/" + names[i] + ".txt");
	}
}

#endif