#pragma once
#ifndef SHOT_H
#define SHOT_H

#include <shot_descriptor.h>
#include <Eigen/Dense>
#include <igl/doublearea.h>
#include <igl/readOFF.h>
#include "utils.h"

double mesh_area(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F) {
	Eigen::MatrixXd A;
	igl::doublearea(V, F, A);
	double area = 0.0;
	for (int i = 0; i < V.rows(); i++) {
		area += fabs(A(i, 0));
	}
	area /= 2.;
	return area;
}

void compute_shot(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::MatrixXd& S, int bins, double scale=6.0) {
	unibo::SHOTParams params;
	params.bins = bins;
	//double surface_area = 6.*sqrt((compute surface area)) / 100.0;
	//params.radius = (6.*sqrt(mesh_area(V, F)) / 100.0);
	params.radius = (scale*sqrt(mesh_area(V, F)) / 100.0);
	params.minNeighbors = 3;
	params.localRFradius = params.radius;

	unibo::SHOTDescriptor sd(params);
	const size_t sz = sd.getDescriptorLength();
	// sz == m_descLength
	/*if (m_params.doubleVolumes) m_k = 32;
      else m_k=16; //number of onion husks
      m_descLength = m_k*(m_params.bins+1);*/
	
	mesh_t mesh;

	int nv = V.rows();
	int nt = F.rows();
	S.resize(nv, sz);
	std::vector< vec3d<double> > V_(nv);

	for (int i = 0; i < nv; ++i)
	{
		V_[i].x = V(i, 0);
		V_[i].y = V(i, 1);
		V_[i].z = V(i, 2);
	}

	mesh.put_vertices(V_);

	for (int i = 0; i < nt; ++i){
		mesh.add_triangle(F(i, 0), F(i, 1), F(i, 2));
	}

	//std::cout << "Computing SHOTs on " << nv << " points... " << std::flush;

	for (size_t i = 0; i<nv; ++i)
	{
		unibo::shot s;
		sd.describe(mesh, i, s);
		for (size_t j = 0; j < sz; ++j) {
			S(i, j) = s(j);
		}
	}
}

void compute_shot(const std::string& shapes_path,  const std::string& dst_path, int bins, double scale) {
	std::vector< std::string > shapes;
	getFilesList_(shapes_path, ".off", shapes);
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	Eigen::MatrixXd S;
	std::cout << "compute shot descriptor for " << shapes_path << endl;
	for (int i = 0; i < shapes.size(); i++) {
		display_progress(float(i)/shapes.size());
		if (!igl::readOFF(shapes_path + "/" + shapes[i] + ".off", V, F))
		{
			cout << "failed to load mesh " << shapes[i] << endl;
			//return false;
		}

		compute_shot(V, F, S, bins, scale);

		save_matrix_(dst_path + "/" + shapes[i] + ".txt", S);
	}
}

#endif
