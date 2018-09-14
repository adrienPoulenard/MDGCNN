#pragma once
#ifndef CROSS_SPHERE_H
#define CROSS_SPHERE_H

#include <iostream>
#include <fstream>

//#include <igl/viewer/Viewer.h>
//#include <igl/parula.h>
//#include <igl/read_triangle_mesh.h>
#include <igl/readOFF.h>
#include <Eigen/Dense>
#include "create_dataset.h"
#include "utils.h"


template <typename T>
void crossSphere_(const std::string& sphere_path, std::vector<T>& v) {
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
	double margin = sin(M_PI/28.0);
	for (int i = 0; i < nv; i++) {
		if (V(i, 2) < -1.0 + margin) {
			if (V(i, 1) < 0.0) {
				v[i] = -1.0;
			}
			else {
				v[i] = 1.0;
			}
		}
	}
}

void crossSphere__(const std::string& sphere_path, Eigen::MatrixXd& M) {
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
	M.resize(nv, 4);
	M.setZero();
	
	double margin = sin(M_PI / 28.0);
	for (int i = 0; i < nv; i++) {
		if (V(i, 2) < -1.0 + margin) {
			if (V(i, 1) < 0.0) {
				M(i, 2) = -1.0;
			}
			else {
				M(i, 3) = 1.0;
			}
			if (V(i, 0) < 0.0) {
				M(i, 0) = -1.0;
			}
			else {
				M(i, 1) = 1.0;
			}
		}
	}
}



void crossSphere(const std::string& shapes_path,
	const std::string& dataset_path) {
	// create permuted spheres


	/*computeFilterMaper(radius,
		nrings,
		ndirs,
		shapes_path,
		dataset_path);*/

	std::vector< std::string > names;
	getFilesList_(shapes_path, ".off", names);
	Eigen::MatrixXd M;

	for (int i = 0; i < names.size(); i++) {
		// permute labels 
		crossSphere__(shapes_path + "/" + names[i] + ".off", M);
		save_matrix_(dataset_path + "/" + "signals/" + names[i] + ".txt", M);
	}
}

#endif // !CROSS_SPHERE_H
