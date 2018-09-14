#pragma once
#ifndef SIGNED_SPHERE_H
#define SIGNED_SPHERE_H

#include <iostream>
#include <fstream>

//#include <igl/viewer/Viewer.h>
//#include <igl/parula.h>
//#include <igl/read_triangle_mesh.h>
#include <igl/readOFF.h>
#include <Eigen/Dense>
#include "create_dataset.h"
#include "utils.h"

void dotedSphere(const std::string& sphere_path, const std::string& dot_path) {
	// load sphere
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	if (!igl::readOFF(sphere_path, V, F))
	{
		//cout << "failed to load mesh" << endl;
		exit(666);
	}
	// center and normalize
	centerObject(V);
	int nv = V.rows();
	std::vector<double> v;
	v.resize(nv);
	for (int i = 0; i < nv; i++) {
		v[i] = 0.0;
	}
	double margin = sin(M_PI/100.0);
	for (int i = 0; i < nv; i++) {
		if (V(i, 2) > 0.999) {
			v[i] = 1.0;
		}
	}
	save_vector_(dot_path, v);
}

template <typename T>
void signedSphere_(const std::string& sphere_path, std::vector<T>& v) {
	// load sphere
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	if (!igl::readOFF(sphere_path, V, F))
	{
		//cout << "failed to load mesh" << endl;
		exit(666);
	}
	// center and normalize
	centerObject(V);
	// computing labels

	// regular
	int nv = V.rows();
	v.resize(nv);
	for (int i = 0; i < nv; i++) {
		v[i] = 0.0;
	}
	double margin = 0.02;
	for (int j = 0; j < 3; j++) {
		for (int i = 0; i < nv; i++) {
			if (V(i, j) < -1.0 + margin) {
				v[i] = -1.0;
			}
			if (V(i, j) > 1.0 - margin) {
				v[i] = 1.0;
			}
		}
	}
}

void signedSphere__(const std::string& sphere_path, Eigen::MatrixXd& M) {
	// load sphere
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	if (!igl::readOFF(sphere_path, V, F))
	{
		//cout << "failed to load mesh" << endl;
		exit(666);
	}
	// center and normalize
	centerObject(V);
	// computing labels

	// regular
	int nv = V.rows();
	M.resize(nv, 6);
	M.setZero();
	double margin = 0.02;
	for (int j = 0; j < 3; j++) {
		for (int i = 0; i < nv; i++) {
			if (V(i, j) < -1.0 + margin) {
				M(i, 2 * j) = 1.0;
			}
			if (V(i, j) > 1.0 - margin) {
				M(i, 2 * j + 1) = 1.0;
			}
		}
	}
}



void signedSphere(double radius,
	int nrings,
	int ndirs,
	const std::string& shapes_path,
	const std::string& dataset_path,
	double margin = 0.02) {
	// create permuted spheres
	

	computeFilterMaper(radius,
		nrings,
		ndirs,
		shapes_path,
		dataset_path);

	std::vector< std::string > names;
	getFilesList_(shapes_path, ".off", names);
	Eigen::MatrixXd M;
	
	for (int i = 0; i < names.size(); i++) {
		// permute labels 
		signedSphere__(shapes_path + "/" + names[i] + ".off", M);
		save_matrix_(dataset_path + "/" + "signals/" +  names[i] + ".txt", M);
	}
}



#endif // !SIGNED_SPHERE_H
