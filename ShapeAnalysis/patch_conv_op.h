#pragma once
#ifndef PATCH_CONV_OP_H
#define PATCH_CONV_OP_H

#include <GPC.h>
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
#include <load_mesh.h>
#include "utils.h"



class PatchConvOperator {
public:
	PatchConvOperator(const std::string& path) {

		if (loadMesh(path, mesh)) {
			gpc = new GPC(mesh);
		}
		else {
			exit(666);
			// throw error
		}
	}

	PatchConvOperator(geodesic::Mesh& myMesh) : mesh(myMesh) {
		gpc = new GPC(myMesh);
	}
	PatchConvOperator(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F) {
		std::vector<int> idx;
		std::vector<double> points;
		std::vector<unsigned> faces;
		convert(V, F, points, faces);
		mesh.initialize_mesh_data(points, faces);
		gpc = new GPC(mesh);
	}
	~PatchConvOperator() {
		delete gpc;
	}


	void compute(int ndir, int nrings, double radius, double precision = 0.000001) {
		int nv = gpc->getNbVertices();
		
		transport_angles.resize(nv*nrings*ndir * 3);
		// transport_bins.resize(nv*nrings*ndir * 3);
		// transport_weights.resize(nv*nrings*ndir * 3 * 2);
		bin_contrib.resize(nv*nrings*ndir * 3);
		triangle_bar_coords.resize(nv*nrings*ndir * 3);

		double r;
		double theta;
		Eigen::Vector3i contrib_id;
		Eigen::Vector3d bar_coords;
		Eigen::Vector3d transport;
		//local_frame3d;
		//base_point;
		double t_x;
		for (int i = 0; i < nv; i++) {
			gpc->compute(i, (nrings+1.0)*radius/nrings, 0, precision);
			for (int k = 0; k < ndir; k++) {
				theta = (2.*k*M_PI) / ndir;
				for (int j = 0; j < nrings; j++) {
					r = (radius * (j + 1))/nrings;
					gpc->getPatchBin(r, theta, contrib_id, bar_coords, transport, t_x);
					for (int l = 0; l < 3; l++) {
						bin_contrib[3 * (ndir*(nrings*i + j) + k) + l] = contrib_id(l);
						triangle_bar_coords[3 * (ndir*(nrings*i + j) + k) + l] = bar_coords(l);
						transport_angles[3 * (ndir*(nrings*i + j) + k) + l] = transport(l);
					}
				}
			}
		}
	}


	bool savePatchOp(const std::string& path_c, const std::string& path_b, const std::string& path_t) const {
		return (save_vector_(path_c, bin_contrib) && 
			save_vector_(path_b, triangle_bar_coords) &&
			save_vector_(path_t, transport_angles));
	}

	/*bool saveLocalFrame3d(const std::string& path) const {
		return save_vector_(path, local_frame3d);
	}

	bool saveBasePoint(const std::string& path) const {
		return save_vector_(path, base_point);
	}*/

	GPC& Gpc() {
		return *gpc;
	}

protected:
	GPC* gpc;
	//std::map< std::pair<int, int>, float > patch_op;
	geodesic::Mesh mesh;
	// Eigen::MatrixXi bin_count;

	std::vector<int> bin_contrib;
	std::vector<float> transport_angles;
	std::vector<float> transport_bins;
	std::vector<float> transport_weights;
	std::vector<float> triangle_bar_coords;

	std::vector<float> local_frame3d;
	std::vector<float> base_point;
};
//*/

#endif // !PATCH_CONV_OP_H
