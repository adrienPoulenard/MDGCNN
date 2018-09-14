#pragma once
#ifndef VISUALIZE_PATCH_H
#define VISUALIZE_PATCH_H
#include <iostream>
#include <fstream>

//#include <igl/viewer/Viewer.h>
//#include <igl/parula.h>
//#include <igl/read_triangle_mesh.h>
#include <igl/readOFF.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <patch_op.h>
#include <GPC.h>
#include <load_mesh.h>
#include "visualize.h"

void visualize_patch(int bp, double radius, const std::string& path) {
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	if (!igl::readOFF(path, V, F))
	{
		//cout << "failed to load mesh" << endl;
		exit(666);
	}
	geodesic::Mesh mesh;
	GPC* gpc = NULL;
	if (loadMesh(path, mesh)) {
		gpc = new GPC(mesh);
	}
	else {
		exit(666);
		// throw error
	}
	gpc->compute(bp, radius);
	int nv = gpc->getNbVertices();
	std::vector<double> v(nv);
	//v.resize(nv);
	for (int i = 0; i < nv; i++) {
		v[i] = gpc->getAngle()[i];
	}
	visualize(V, F, v);
}




#endif // !VISUALIZE_PATCH_H
