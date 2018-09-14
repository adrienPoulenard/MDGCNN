#pragma once
#ifndef POOLING_OP_H
#define POOLING_OP_H

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
#include "load_mesh.h"

template <typename T>
bool saveMatrix(const std::string& path, const Eigen::Matrix<T, -1, -1> & M) {
	ofstream myfile(path.c_str());
	if (myfile.is_open())
	{
		for (int i = 0; i < M.rows(); i++) {
			for (int j = 0; j < M.cols(); j++) {
				myfile << M(i,j);
			}
			myfile << "\n";
		}
		myfile.close();
		return true;
	}
	else {
		cout << "Unable to open file: " << path << endl;
		return false;
	}
}






class PoolingOperator {
public:
	PoolingOperator(const std::string& path) {

		if (loadMesh(path, mesh)) {
			gpc = new GPC(mesh);
		}
		else {
			exit(666);
			// throw error
		}
	}
	PoolingOperator(geodesic::Mesh& myMesh) {
		gpc = new GPC(myMesh);
	}
	~PoolingOperator() {
		delete gpc;
	}


	void compute(double radius, double precision = 0.000001) {
		if (gpc == NULL) {
			exit(666);
		}
		int nv = gpc->getNbVertices();
		int min_nv = std::numeric_limits<int>::max();
		Patch patch;
		int percent = 0;
		// get minimal number of vertices 
		for (int i = 0; i < nv; i++) {
			gpc->compute(i, radius, 0, precision);
			gpc->getPatch(patch);
			min_nv = std::min(min_nv, int(patch.l().size()));
			if (nv*percent / 100. < i) {
				percent++;
				cout << (percent/2) << " percent of patch operator computed" << endl;
			}
		}
		percent = 0;
		std::vector< std::size_t > nn_idx(min_nv);
		std::vector< double > nn_sqr(min_nv);
		for (int i = 0; i < nv; i++) {
			gpc->compute(i, radius, 0, precision);
			gpc->getPatch(patch);
			patch.getNN(0., 0., nn_idx, nn_sqr);
			for (int j = 0; j < min_nv; j++) {

			}
			if (nv*percent / 100. < i) {
				percent++;
				cout << (percent/2 + 50) << " percent of patch operator computed" << endl;
			}
		}

	}
	
	bool savePoolMat(const std::string& path) const {
		return saveMatrix(path, pool_mat);
	}

protected:
	GPC* gpc;
	geodesic::Mesh mesh;
	Eigen::MatrixXi pool_mat;
	Eigen::MatrixXi 
};



#endif