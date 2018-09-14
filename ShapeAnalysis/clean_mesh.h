#pragma once
#ifndef CLEAN_MESH_H
#define CLEAN_MESH_H

#include <igl/is_edge_manifold.h>
#include <igl/is_vertex_manifold.h>
#include <igl/remove_unreferenced.h>
#include <igl/readOFF.h>

bool is_vertex_manifold(const std::string& path) {
	Eigen::Matrix<bool, -1, -1> B;
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	Eigen::MatrixXd NV;
	Eigen::MatrixXi NF;
	igl::readOFF(path + ".off", V, F);
	Eigen::VectorXi I;
	Eigen::VectorXi J;
	igl::remove_unreferenced(V, F, NV, NF, I, J);
	igl::writeOFF(path + "_.off", NV, NF);
	return igl::is_vertex_manifold(NF, B);
}

bool is_edge_manifold(const std::string& path) {
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	igl::readOFF(path, V, F);
	return igl::is_edge_manifold(F);
}

#endif // !CLEAN_MESH_H
