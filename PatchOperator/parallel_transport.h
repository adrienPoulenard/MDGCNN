#pragma once
#ifndef PARALLEL_TRANSPORT_H
#define PARALLEL_TRANSPORT_H

#include <geodesic_mesh.h>
#include <Eigen/Dense>
#include "local_coordinates.h"

#define PI 3.14159265359





inline double principalAngle(double t) {
	int k = int(floor(t / (2.*PI)));
	return t - 2.*k*PI;
}

/*inline double anglesDifference(double t1, double t2) {
	t1 = principalAngle(t1);
	t2 = principalAngle(t2);
	if (t2 - t1 > PI) {
		return t1 + 2.*PI - t2;
	}
	if (t1 - t2 > PI) {
		return t1 - 2.*PI - t2;
	}
	return principalAngle((1. - alpha)*t1 + alpha*t2);
}*/

inline double angleMix(double alpha, double t1, double t2) {

	if (t2 - t1 > PI) {
		return principalAngle((1. - alpha)*(t1 + 2.*PI) + alpha*t2);
	}
	if (t1 - t2 > PI) {
		return principalAngle((1. - alpha)*t1 + alpha*(t2 + 2.*PI));
	}
	return principalAngle((1. - alpha)*t1 + alpha*t2);
}

inline double Angle_(double c, double s) {
	double a = atan2(s, c);
	if (a < 0.) {
		a += 2.*M_PI;
	}
	return a;
}

inline double Angle_(const Eigen::Vector3d& u, const Eigen::Vector3d& v) {
	double c = u.dot(v);
	double s = (u.cross(v)).norm();
	double a = atan2(s, c);
	if (a < 0.) {
		a += 2.*M_PI;
	}
	return a;
}

inline int is_adj(int p, geodesic::face_pointer& f) {
	for (int i = 0; i < 3; i++) {
		if (p == f->adjacent_vertices()[i]->id()) {
			return i;
		}
	}
	return -1;
}

inline double incident_angle(const std::vector<double>& angles, int j) {
	if (j = angles.size() - 1) {
		return 2 * M_PI - angles[j];
	}
	else {
		return angles[j + 1] - angles[j];
	}
}
inline void transportToFace(
	const Eigen::Vector3d& x_bar,
	geodesic::face_pointer& f,
	Eigen::Vector3d& rho1,
	Eigen::Vector3d& rho2,
	double& r_1,
	double& r_2,
	double alpha,
	int mode,
	int idj,
	int idk,
	const std::vector< double >& Rho1,
	const std::vector< double >& Rho2,
	const std::vector< std::vector<double> >& angles,
	const std::vector< std::vector<int> >& neigh_id,
	const Eigen::MatrixXd& Ve) {


	std::vector<bool> flip(3);
	std::vector<int> p_id(3);
	std::vector<int> p_id_in_f(3);
	std::vector<double> angles_x(3);
	Eigen::Matrix3d f_vertices;
	f_vertices.setZero();
	Eigen::Vector3d x;
	x.setZero();
	std::vector<int> v_id(3);




	for (int i = 0; i < 3; i++) {
		v_id[i] = f->adjacent_vertices()[i]->id();
		f_vertices.col(i) = Ve.row(v_id[i]).transpose();
		x += x_bar(i)*f_vertices.col(i);
		int adj = 0;
		int nb_n = neigh_id[v_id[i]].size();
		for (int j = 0; j < nb_n; j++) {
			adj = is_adj(neigh_id[v_id[i]][j], f);
			if (adj != -1) {
				if (neigh_id[v_id[i]][(j + 1)%nb_n] == f->adjacent_vertices()[(adj + 1) % 3]->id()) {
					flip[i] = false;
					p_id[i] = j;
					p_id_in_f[i] = adj;
				}
				if (neigh_id[v_id[i]][(j + 1)%nb_n] == f->adjacent_vertices()[(adj + 2) % 3]->id()) {
					flip[i] = true;
					p_id[i] = j;
					p_id_in_f[i] = adj;
				}
			}
		}
	}
	
	Eigen::Matrix3d x_to_f;
	x_to_f.setZero();
	Eigen::Matrix3d edges;
	edges.setZero();

	
	for (int i = 0; i < 3; i++) {
		x_to_f.col(i) = f_vertices.col(i) - x;
		x_to_f.col(i).normalize();
		edges.col(i) = f_vertices.col((i + 1) % 3) - f_vertices.col(i);
		edges.col(i).normalize();
	}
	std::vector<double> angles_f(3);

	for (int i = 0; i < 3; i++) {
		angles_f[i] = Angle_(edges.col(i), -edges.col((i + 2)%3));
	}

	angles_x[0] = 0.;
	for (int i = 0; i < 2; i++) {
		angles_x[i + 1] = angles_x[i] + Angle_(x_to_f.col(i), x_to_f.col(i + 1));
	}

	std::vector<double> angles_xf(3);
	std::vector<double> angles_fx(3);

	for (int i = 0; i < 3; i++) {
		if (!flip[i]) {
			angles_xf[i] = Angle_(edges.col(i), -x_to_f.col(i));
		}
		else {
			angles_xf[i] = Angle_(-edges.col((i + 2) % 3), -x_to_f.col(i));
		}
		
		double da = 0.0;
		int nb_n = angles[v_id[i]].size();
		if (!flip[i]) {
			da = incident_angle(angles[v_id[i]], p_id[i]);
		}
		else {
			da = incident_angle(angles[v_id[i]], (p_id[i] + nb_n - 1) % nb_n);
		}
			
		
		//angles_xf[i] *= (da / angles_f[i]);
		angles_fx[i] = angles_xf[i] + angles[v_id[i]][p_id[i]];
	}

	// transfer angles to center



	// transfer angles to interior point
	Eigen::Vector3d r1;
	Eigen::Vector3d r2;

	
	for (int i = 0; i < 3; i++) {
		if (!flip[i]) {
			r1(i) = principalAngle(angles_x[i] + (Rho1[v_id[i]] - (angles_fx[i]) - PI));
			r2(i) = principalAngle(angles_x[i] + (Rho2[v_id[i]] - (angles_fx[i]) - PI));	
		}
		else {
			r1(i) = principalAngle(angles_x[i] - (Rho1[v_id[i]] - (angles_fx[i]) - PI));
			r2(i) = principalAngle(angles_x[i] - (Rho2[v_id[i]] - (angles_fx[i]) - PI));
		}
	}

	// blend 

	//r_1 = r1(0);
	//r_2 = r2(0);

	//r_1 = angles_x[0];

	//double r1_ = r1(0);
	//double r2_ = r1(0);

	//cout << "mode " << mode << endl;
	switch (mode) {
	case 0:
		r_1 = angleMix(alpha, r1(idk), r1(idj));
		r_2 = angleMix(alpha, r2(idk), r2(idj));
		break;
	case 1:
		r_1 = angleMix(alpha, r1(idj), r1(idk));
		r_2 = angleMix(alpha, r2(idj), r2(idk));
		break;
	case 2:
		r_1 = r1(idk);
		r_2 = r2(idk);
		break;
	case 3:
		r_1 = r1(idj);
		r_2 = r2(idj);
		break;
	};//*/


	//r_1 = r1_;
	//r_2 = r2(0);

	// transfer back to vertices.

	for (int i = 0; i < 3; i++) {
		if (x_bar(i) < 0.95) {
			if (!flip[i]) {
				rho1(i) = principalAngle(angles_fx[i] + (r_1 - (angles_x[i]) - PI));
				rho2(i) = principalAngle(angles_fx[i] + (r_2 - (angles_x[i]) - PI));
			}
			else {
				rho1(i) = principalAngle(angles_fx[i] - (r_1 - (angles_x[i]) - PI));
				rho2(i) = principalAngle(angles_fx[i] - (r_2 - (angles_x[i]) - PI));
			}
		}
		else {
			rho1(i) = Rho1[v_id[i]];
			rho2(i) = Rho2[v_id[i]];
		}
	}
	
}


// a tangent vector at a vertex can be represented as an angle with respect to the first edge
// and a norm
// since the norm is invariant under parallel transport we only tranfer angles
// transfer angle rho along edge joining first vertex to second 
inline double transferAngle(
	double rho, 
	geodesic::vertex_pointer v1,
	geodesic::vertex_pointer v2, 
	const std::vector< double >& angles_v1,
	const std::vector< double >& angles_v2,
	const std::vector<int>& neighId_v1,
	const std::vector<int>& neighId_v2) {

	
	// first find the id of v2 in vertices adjacent to v1 (same for v1)
	int v1_id_in_v2 = 0;
	int v2_id_in_v1 = 0;
	int nb_neigh1 = neighId_v1.size();
	int nb_neigh2 = neighId_v2.size();

	if ((nb_neigh1 == 0) || (nb_neigh2 == 0)) {
		return 0;
	}

	for (unsigned i = 0; i < nb_neigh1; i++) {
		if (v2->id() == neighId_v1[i]) {
			v2_id_in_v1 = i;
		}
	}
	for (unsigned i = 0; i < nb_neigh2; i++) {
		if (v1->id() == neighId_v2[i]) {
			v1_id_in_v2 = i;
		}
	}
	// find relative orientation of tangent planes to v1 and v2
	//surement un probleme ici la determination de l'orientation par propagation de front ne marche pas'
	if (neighId_v1[(v2_id_in_v1 + 1) % nb_neigh1] != neighId_v2[(v1_id_in_v2 + 1) % nb_neigh2]) {
		return principalAngle(angles_v2[v1_id_in_v2] + (rho- (angles_v1[v2_id_in_v1]) - PI));
	}
	else {
		//cout << "kkkkkkk" << endl;
		return principalAngle(angles_v2[v1_id_in_v2] - (rho - (angles_v1[v2_id_in_v1]) - PI));
	}
}

inline void orthonormalize(double& t1, double& t2) {
	double mid = angleMix(0.5, t1, t2);
	if (t1 - mid >= 0.0) {
		t1 = mid + PI / 4.0;
	}
	else {
		t1 = mid - PI / 4.0;
	}
	if (t2 - mid >= 0.0) {
		t2 = mid + PI / 4.0;
	}
	else {
		t2 = mid - PI / 4.0;
	}
}

void tangentSpaceEmbeddingAtVertex(Eigen::Vector3d& v, double angle, const std::vector<double>& angles) {
	int v_id = 0;
	int deg = angles.size();
	for (unsigned i = 0; i < deg; i++) {

	}
}

class meshLocalAlignment : public localAlignment {
public:
	meshLocalAlignment(unsigned int nb_angular_bins) {
		nb_bins = nb_angular_bins;
	}
	virtual void computeLocalAlignmentOperatorAt(
		unsigned int pt_id,
		const Eigen::MatrixXd& transport_mat,
		const std::vector<int>& valid_idx,
		const std::vector<int>& valid_idx_inverse,
		std::vector< Eigen::Triplet<int> >& triplets) {
		unsigned nv = valid_idx.size();
		unsigned nb_valid = transport_mat.rows();
		int offset = 0;
		for (unsigned i = 0; i < nb_valid; i++) {

		}
	}
	~meshLocalAlignment() {}
private:
	unsigned nb_bins;

	inline bool orientation_reversed(double rho1, double rho2) {
		return (cos(rho1)*sin(rho2) - sin(rho1)*cos(rho2) < 0.0);
	}
	inline int angular_bin(double theta) {
		floor((nb_bins*theta + M_PI) / (2.0*M_PI));
	}
	inline int transform_id(double rho1, double rho2) {
		return orientation_reversed(rho1, rho2)*nb_bins + angular_bin(rho1);
	}
};


#endif
