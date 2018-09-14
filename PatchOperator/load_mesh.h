#pragma once
#ifndef LOAD_MESH_H
#define LOAD_MESH_H

#include "GPC.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <windows.h>
#include <geodesic_algorithm_exact.h>
#include <igl/readOFF.h>
#include <map>
#include <list>
#include <math.h>

void convert(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, std::vector<double>& v, std::vector<unsigned>& f) {
	v.resize(3 * V.rows());
	for (unsigned i = 0; i < V.rows(); i++) {
		v[3 * i] = V(i, 0);
		v[3 * i + 1] = V(i, 1);
		v[3 * i + 2] = V(i, 2);
	}
	f.resize(3 * F.rows());
	for (unsigned i = 0; i < F.rows(); i++) {
		f[3 * i] = F(i, 0);
		f[3 * i + 1] = F(i, 1);
		f[3 * i + 2] = F(i, 2);
	}
}

bool loadMesh(const std::string& path, geodesic::Mesh& mesh) {
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	std::vector<int> idx;
	std::vector<double> points;
	std::vector<unsigned> faces;
	if (!igl::readOFF(path, V, F))
	{
		cout << "failed to load mesh" << endl;
		return false;
	}

	convert(V, F, points, faces);
	mesh.initialize_mesh_data(points, faces);

	return true;
}


#endif // !LOAD_MESH
