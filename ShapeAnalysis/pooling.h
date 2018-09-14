#pragma once
#ifndef POOLING_H
#define POOLING_H

#include <igl/decimate.h>
#include <patch_op.h>
#include <Eigen/Dense>
#include "utils.h"

inline void local_frame(int i, const GPC& gpc,
	Eigen::Matrix3d& local_frame) {

	if (gpc.isMeshBoundaryPoint(i)) {
		local_frame.setIdentity();
		return;
	}
	Eigen::Vector3d u;
	Eigen::Vector3d v;
	Eigen::Vector3d n;
	n = gpc.getNormals().row(i).transpose();
	Eigen::Vector3d bp = gpc.getVertexCoord(i);
	int u_id = gpc.getOneRingVertices(i)[0];
	u = gpc.getVertexCoord(u_id) - bp;
	u.normalize();
	n.normalize();
	u -= n.dot(u)*n;
	u.normalize();
	v = n.cross(u);
	local_frame.col(0) = u;
	local_frame.col(1) = v;
	local_frame.col(2) = n;
}


inline void cross_prod_mat(const Eigen::Vector3d& v, Eigen::Matrix3d& V) {
	V << 0., -v(2), v(1),
		v(2), 0., -v(0),
		-v(1), v(0), 0.;
}

inline void rotab(Eigen::Vector3d& a, Eigen::Vector3d& b, Eigen::Matrix3d& R) {
	R.setIdentity();
	Eigen::Matrix3d V;
	a.normalize();
	b.normalize();
	cross_prod_mat(a.cross(b), V);
	double c = a.dot(b);
	//cout << R << endl;
	R += V + (1./(1+c))*(V * V);
}



inline double angular_shift(Eigen::Matrix3d& frame_1, Eigen::Matrix3d& frame_2) {
	// 1 = original
	// 2 = downsampled
	Eigen::Vector3d n1 = frame_1.col(2);
	Eigen::Vector3d n2 = frame_2.col(2);
	//cout << "normals dot " << n1.dot(n2) << endl;
	if (n1.dot(n2) < 0.) {
		n2 = -n2;
		frame_2.col(2) = n2;
	}
	Eigen::Matrix3d R;
	rotab(n1, n2, R);
	//cout << "R" << endl;
	//cout << R << endl;
	//cout << " " << endl;
	Eigen::Matrix3d frame_12 = R * frame_1;
	Eigen::Vector3d u = frame_12.col(0);
	//cout << "cc " << frame_2.col(0).dot(frame_2.col(1)) << endl;
	double c = u.dot(frame_2.col(0));
	double s = u.dot(frame_2.col(1));
	double res = 2.*M_PI-angle_(c, s);
	// cout << "a: " << res / (2*M_PI) << endl;
	return res;
}

inline int index_shift(double a, int ndir) {
	double b = (ndir*a) / (2.*M_PI);
	int B = (int)(std::round(b));
	return (ndir + B) % ndir;
}

inline double angular_shift(int i, int j, const GPC& g1, const GPC& g2) {
	Eigen::Matrix3d frame_1;
	Eigen::Matrix3d frame_2;
	local_frame(i, g1, frame_1);
	local_frame(j, g2, frame_2);
	return angular_shift(frame_1, frame_2);
}


inline int index_shift(int i, int j, const GPC& g1, const GPC& g2, int ndir) {

	Eigen::Matrix3d frame_1;
	Eigen::Matrix3d frame_2;
	local_frame(i, g1, frame_1);
	local_frame(j, g2, frame_2);
	double a = angular_shift(frame_1, frame_2);
	return index_shift(a, ndir);
}

void index_shift(const GPC& g1, const GPC& g2,
				 const Eigen::VectorXi& I, int ndir,
	             std::vector<int>& shift) {
	if (shift.size() != I.rows()) {
		shift.resize(I.size());
	}
	for (int i = 0; i < shift.size(); i++) {
		shift[i] = index_shift(I(i), i, g1, g2, ndir);
	}
}

void angular_shift(const GPC& g1, const GPC& g2,
	const Eigen::VectorXi& I,
	std::vector<double>& shift) {
	if (shift.size() != I.rows()) {
		shift.resize(I.size());
	}
	for (int i = 0; i < shift.size(); i++) {
		shift[i] = angular_shift(I(i), i, g1, g2);
	}
}

void shift_test(const std::string& path, double ratio) {
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	Eigen::MatrixXd U;
	Eigen::MatrixXi G;
	Eigen::VectorXi J;
	Eigen::VectorXi I;

	if (!igl::readOFF(path, V, F))
	{
		cout << "failed to load mesh" << endl;
		exit(666);
	}

	PatchOperator* mapper1 = NULL;
	PatchOperator* mapper2 = NULL;

	mapper1 = new PatchOperator(path);
	igl::decimate(V, F, (size_t)(F.rows()*ratio), U, G, J, I);
	mapper2 = new PatchOperator(U, G);

	Eigen::Vector3d u1;
	Eigen::Vector3d u2;
	Eigen::Vector3d v;
	Eigen::Vector3d n1;
	Eigen::Vector3d n2;


	for (int i = 0; i < U.rows(); i++) {
		int u1_id = mapper1->Gpc().getOneRingVertices(I(i))[0];
		int u2_id = mapper2->Gpc().getOneRingVertices(i)[0];
		u1 = mapper1->Gpc().getVertexCoord(u1_id) - mapper1->Gpc().getVertexCoord(I(i));
		u1.normalize();
		u2 = mapper2->Gpc().getVertexCoord(u2_id) - mapper2->Gpc().getVertexCoord(i);
		u2.normalize();
		v = mapper2->Gpc().getVertexCoord(i) - mapper1->Gpc().getVertexCoord(I(i));
		n1 = mapper1->Gpc().getNormals().row(I(i)).transpose();
		n2 = mapper2->Gpc().getNormals().row(i).transpose();
		cout << "normals dot: " << n1.dot(n2) << endl;
		if (n1.dot(n2) < 0) {
			n2 = -n2;
		}
		
		cout << "normals angle: " << asin((n1.cross(n2)).norm())/(2.*M_PI) << endl;
		cout << i << " " << v.norm() << " " << (u1.cross(u2)).norm() << endl;
	}

}

/*void pooling_op(const GPC& g1, const GPC& g2,
	const Eigen::MatrixXi F1, const Eigen::MatrixXi F2,
	const std::vector<int>& I, int ndir,
	std::vector<int>& shift) {

	//   U  #U by dim list of output vertex posistions (can be same ref as V)
	//   G  #G by 3 list of output face indices into U (can be same ref as G)
	//   J  #G list of indices into F of birth face
	//   I  #U list of indices into V of birth vertices

	Eigen::MatrixXd U;
	Eigen::MatrixXi G;

}*/

bool reduce_shape(double ratio, const std::string& name,
	const std::string& shape, const std::string& dst_path) {
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	Eigen::MatrixXd U;
	Eigen::MatrixXi G;
	Eigen::VectorXi J;
	Eigen::VectorXi I;

	if (!igl::readOFF(shape, V, F))
	{
		cout << "failed to load mesh" << endl;
		return false;
	}

	igl::decimate(V, F, (size_t)(F.rows()*ratio), U, G, J, I);

	if (!igl::writeOFF(dst_path + "/" + name + "_" + std::to_string(U.rows()) + ".off", U, G))
	{
		cout << "failed to save mesh" << endl;
		return false;
	}

	return true;
}


#endif
