#pragma once
#ifndef GPC_H
#define GPC_H

#include <geodesic_mesh.h>
#include <Eigen/Dense>
#include <map>
#include <nanoflann.hpp>
#include "mesh_connectivity.h"
#include "parallel_transport.h"
#include <ctime>
#include <igl/per_vertex_normals.h>
#include "local_coordinates.h"
#include <cmath>
//#include "pull_back.h"

#define SELF_COLLISION_RADIAL 1
#define ORIENTATION_RADIAL 2
#define SELF_COLLISION_GLOBAL 3
#define ORIENTATION_GLOBAL 4


inline void barycentric_coords_(
	const Eigen::Vector2d& a,
	const Eigen::Vector2d& b,
	const Eigen::Vector2d& c,
	const Eigen::Vector2d& x, 
	Eigen::Vector3d& x_bar) {

	double t = ((b(1) - c(1))*(a(0) - c(0)) + (c(0) - b(0))*(a(1) - c(1)));
	x_bar(0) = ((b(1) - c(1))*(x(0) - c(0)) + (c(0) - b(0))*(x(1) - c(1))) / t;
	x_bar(1) = ((c(1) - a(1))*(x(0) - c(0)) + (a(0) - c(0))*(x(1) - c(1))) / t;
	x_bar(2) = 1. - x_bar(0) - x_bar(1);
}

template <typename T>
inline void reindex_vector(Eigen::Matrix<T, 3, 1>& v, const Eigen::Vector3i& idx) {
	Eigen::Vector3d tmp;
	for (int i = 0; i < 3; i++) {
		tmp(i) = v(idx(i));
	}
	for (int i = 0; i < 3; i++) {
		v(i) = tmp(i);
	}
}

inline bool is_in_angular_sector(double t, double t1, double t2) {
	Eigen::Vector2d v;
	Eigen::Vector2d v1;
	Eigen::Vector2d v2;
	v(0) = cos(t);
	v1(0) = -sin(t1);
	v2(0) = -sin(t2);
	v(1) = sin(t);
	v1(1) = cos(t1);
	v2(1) = cos(t2);
	return (v.dot(v1)*v.dot(v2) <= 0.);
}


// check 

//bad function we should separate angular and radial distorsion of triangles
inline bool isValidTraingle(
	double ratio,
	double ua, double ub, double uc,
	double ta, double tb, double tc,
	const Eigen::Vector3d& a,
	const Eigen::Vector3d& b,
	const Eigen::Vector3d& c) {

	double l1;
	double l2;
	Eigen::Vector2d u;
	u << cos(ta), sin(ta);
	u *= ua;
	Eigen::Vector2d v;
	v << cos(tb), sin(tb);
	v *= ub;
	Eigen::Vector2d w;
	w << cos(tc), sin(tc);
	w *= uc;

	l1 = (b - a).norm() + (c - b).norm() + (c - a).norm();
	l2 = (v - u).norm() + (w - v).norm() + (w - u).norm();
	


	return (l2 <= ratio*l1);
}

inline void gradientAtTriangle(const Eigen::Vector3d& a,
							const Eigen::Vector3d& b,
							const Eigen::Vector3d& c, 
							double ua,
							double ub,
							double uc,
							Eigen::Vector3d& Grad) {
	Eigen::Vector3d e1 = b - a;
	Eigen::Vector3d e2 = c - a;
	Eigen::Matrix2d T;
	double n1 = e1.dot(e1);
	double n2 = e2.dot(e2);
	double n12 = e1.dot(e2);
	T << n1, n12,
		n12, n2;
	Eigen::Vector2d coeffs;
	coeffs << ub - ua, uc - ua;
	coeffs = T.inverse()*coeffs;
	Grad = coeffs(0)*e1+ coeffs(1)*e2;
}
/*
inline bool isValidTraingle(
	double ratio,
	const Eigen::Vector2d& u,
	const Eigen::Vector2d& v,
	const Eigen::Vector2d& w,
	const Eigen::Vector3d& a,
	const Eigen::Vector3d& b,
	const Eigen::Vector3d& c) {

	double l1;
	double l2;

	l1 = (b - a).norm() + (c - b).norm() + (c - a).norm();
	l2 = (v - u).norm() + (w - v).norm() + (w - u).norm();

	return (l2 <= ratio*l1);
}*/

class Patch {
public:
	typedef nanoflann::KDTreeEigenMatrixAdaptor< Eigen::Matrix<double, -1, -1> > kd_tree_d;
	Patch(){
		mat_index = NULL;
	}
	void setPatch(int nb_valid,
		const std::vector<int>& id,
		const std::vector<double>& Mu,
		const std::vector<double>& R,
		const std::vector<double>& Theta,
		const std::vector<double>& U1,
		const std::vector<double>& U2,
		const Eigen::Matrix3d& local_frame,
		const Eigen::Vector3d& base_point,
		double Radius) {
		int n = nb_valid;
		idx.resize(n);
		r_.resize(n);
		theta_.resize(n);
		u1_.resize(n);
		u2_.resize(n);
		mu_.resize(n);
		
		for (int i = 0; i < 3; i++) {
			bp(i) = base_point(i);
			for (int j = 0; j < 3; j++) {
				local_frame3d(i, j) = local_frame(i, j);
			}
		}
		
		radius_ = Radius;
		
		for (int i = 0; i < n; i++) {
			idx[i] = id[i];
			r_[i] = R[id[i]];
			theta_[i] = Theta[id[i]];
			u1_[i] = U1[id[i]];
			u2_[i] = U2[id[i]];
			mu_[i] = Mu[id[i]];
		}
		// compute euclidean coordinates
		eucl_coord.resize(n, 2);

		for (int i = 0; i < n; i++) {
			eucl_coord(i, 0) = r_[i] * cos(theta_[i]);
			eucl_coord(i, 1) = r_[i] * sin(theta_[i]);
		}

		// construct kd-tree
		if (mat_index != NULL) {
			delete mat_index;
			mat_index = NULL;
		}
		mat_index = new kd_tree_d(eucl_coord, 10); // max leaf = 10
		mat_index->index->buildIndex();

	}
	~Patch() {
		delete mat_index;
	}
	const std::vector<double>& u1() const { return u1_; }
	const std::vector<double>& u2() const { return u2_; }
	const std::vector<double>& r() const { return r_; }
	const std::vector<double>& theta() const { return theta_; }
	const std::vector<unsigned int>& l() const { return idx; }
	const std::vector<double>& mu() const { return mu_; }
	const Eigen::Matrix3d& localFrame3d() const { return local_frame3d; }
	const Eigen::Vector3d& basePoint() const { return bp; }
	double radius() const { return radius_; }
	void getNN(double theta, double r, std::vector<std::size_t>& index, std::vector<double>& dists_sqr) const {
		// do a knn search
		int nb_neighs = index.size();

		std::vector<double> query_pt(2);
		query_pt[0] = r*cos(theta);
		query_pt[1] = r*sin(theta);

		//std::vector<double> out_dists_sqr(nb_neighs);

		nanoflann::KNNResultSet<double> resultSet(nb_neighs);

		resultSet.init(&index[0], &dists_sqr[0]);
		mat_index->index->findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));

		/*std::cout << "knnSearch(nn=" << num_results << "): \n";
		for (size_t i = 0; i<num_results; i++)
			std::cout << "ret_index[" << i << "]=" << ret_indexes[i] << " out_dist_sqr=" << out_dists_sqr[i] << endl;*/
		
	}
private:
	std::vector<double> u1_;
	std::vector<double> u2_;
	std::vector<double> r_;
	std::vector<double> theta_;
	std::vector<double> mu_;
	std::vector<unsigned int> idx; // index
	Eigen::Matrix3d local_frame3d;
	Eigen::Vector3d bp;
	double radius_;
	Eigen::MatrixXd eucl_coord;
	kd_tree_d* mat_index;
};

inline double triangleArea(geodesic::face_pointer f) {
	Eigen::Vector3d u;
	Eigen::Vector3d v;
	geodesic::vertex_pointer a = f->adjacent_vertices()[0];
	geodesic::vertex_pointer b = f->adjacent_vertices()[1];
	geodesic::vertex_pointer c = f->adjacent_vertices()[2];
	u(0) = b->x() - a->x();
	u(1) = b->y() - a->y();
	u(2) = b->z() - a->z();

	v(0) = c->x() - a->x();
	v(1) = c->y() - a->y();
	v(2) = c->z() - a->z();

	return (u.cross(v)).norm() / 2.0;
}
inline double areaAt(geodesic::Vertex& v) {
	double res = 0.0;
	for (auto f : v.adjacent_faces()) {
		res += triangleArea(f);
	}
	return res;
}


//gpc must inherit from localCoordinates

class GPC : public localCoordinates< geodesic::Mesh > {
public:
	typedef nanoflann::KDTreeEigenMatrixAdaptor< Eigen::Matrix<double, -1, -1> > kd_tree_d;
	~GPC() {
		if (mat_index != NULL) {
			delete mat_index;
			mat_index = NULL;
		}
	}
	GPC(geodesic::Mesh& myMesh): localCoordinates<geodesic::Mesh>(myMesh), mesh(myMesh) { //(mesh is redundant)
		mat_index = NULL;
		nv = myMesh.vertices().size();
		nf = myMesh.faces().size();
		mu.resize(nv);
		U.resize(nv);
		theta.resize(nv);
		rho1.resize(nv);
		rho2.resize(nv);
		cos_t.resize(nv);
		sin_t.resize(nv);
		x.resize(nv);
		y.resize(nv);
		eucl_coords.resize(nv, 2);
		bar_coords.resize(nv, 3);
		//coordMat.resize(nv, 2);
		is_candidate.resize(nv);
		is_computed.resize(nv);
		is_computed_tmp.resize(nv);
		mesh_idx.resize(nv);
		patch_idx.resize(nv);
		valid_triangle.resize(nf);
		candidate_position.resize(nv);
		oneRingVertices.resize(nv);
		oneRingEdges.resize(nv);
		oneRingFaces.resize(nv);
		oneRingVinF.resize(nv);
		oneRingIncidentAngles.resize(nv);
		oneRingAngles.resize(nv);
		totalAngleAtVertex.resize(nv);
		Differential.resize(2 * nf, 3);
		Vertices.resize(nv, 3);
		Faces.resize(nf, 3);

		radius = std::numeric_limits<double>::max();
		
		geodesic::vertex_pointer v;
		for (int i = 0; i < nv; i++) {
			v = &myMesh.vertices()[i];
			// compute area at vertex i
			mu[i] = areaAt(*v);
			// extract vertices coordinates
			Vertices(i, 0) = v->x();
			Vertices(i, 1) = v->y();
			Vertices(i, 2) = v->z();
			// compute order of 1 ring neighbourhoods
			set1RingOrder(v, oneRingFaces[i], 
				oneRingEdges[i], oneRingVertices[i], 
				oneRingVinF[i], oneRingIncidentAngles[i],
				oneRingAngles[i], totalAngleAtVertex[i]);
			/*totalAngleAtVertex[i] = 0.0;
			for (unsigned j; j < oneRingIncidentAngles[i].size(); j++) {
				totalAngleAtVertex[i] += oneRingIncidentAngles[i][j];
			}*/
		}
		for (unsigned i = 0; i < nf; i++) {
			geodesic::face_pointer f = &(mesh.faces()[i]);
			for (unsigned j = 0; j < 3; j++) {
				Faces(i, j) = f->adjacent_vertices()[j]->id();
			}		
		}
		computeVerticesNormals();
		//igl::per_vertex_normals(Vertices, Faces, VertexNormals);
	}

	geodesic::Mesh& getMesh()  { return mesh; }

	Eigen::Vector3d getVertexCoord(int i) const {
		return Vertices.row(i).transpose();
	}

	const Eigen::MatrixXd& getVertices() const {
		return Vertices;
	}


	const std::vector<double>& getU1() const {
		return rho1;
	}

	const std::vector<double>& getU2() const {
		return rho2;
	}

	const std::vector<int>& getOneRingVertices(int i) const {
		return oneRingVertices[i];
	}

	virtual double compute(int basepoint_id, double rad, int stop_criterion = 0, double precision = 0.000001) {
		std::clock_t start;
		start = std::clock();
		
		if (mat_index != NULL) {
			delete mat_index;
			mat_index = NULL;
		}

		double newUi = 0.0;
		double newTi = 0.0;
		double newRho1i = 0.0;
		double newRho2i = 0.0;
		int update_face_id = 0;
		geodesic::face_pointer f;
		geodesic::edge_pointer e;

		geodesic::SurfacePoint basepoint(&(mesh.vertices()[basepoint_id]));

		bp_id = basepoint.base_element()->id();

		int a_id;
		int b_id;
		int c_id;
		//std::list< geodesic::vertex_pointer > candidates_tmp;
		Eigen::Vector3d n_bp = VertexNormals.row(bp_id).transpose();
		Eigen::Vector3d normal;


		/*Eigen::Vector2d a;
		Eigen::Vector2d b;
		Eigen::Vector2d c;*/

		bool stop = false;
		

	
		std::fill(U.begin(), U.end(), std::numeric_limits<double>::max());
		std::fill(is_candidate.begin(), is_candidate.end(), false);
		candidates.clear();
		std::fill(is_computed.begin(), is_computed.end(), false);
		std::fill(theta.begin(), theta.end(), 0.0);
		/*std::fill(cos_t.begin(), cos_t.end(), 1.0);
		std::fill(sin_t.begin(), sin_t.end(), 0.0);
		std::fill(x.begin(), x.end(), 1.0);
		std::fill(y.begin(), y.end(), 0.0);
		Differential.setConstant(std::numeric_limits<double>::max());
		std::fill(valid_triangle.begin(), valid_triangle.end(), false);*/
		std::fill(mesh_idx.begin(), mesh_idx.end(), -1);
		std::fill(patch_idx.begin(), patch_idx.end(), -1);

		radius = rad;
		//assert(initializeNeighbourhood(basepoint));
		if (!initializeNeighbourhood(basepoint)) {
			// cout << "initialisation failed" << endl;
			is_boundary = true;
			return rad;
		}
		is_boundary = false;

		is_computed[bp_id] = true;
		is_computed_tmp[bp_id] = true;

		for (int i = 0; i < basepoint.base_element()->adjacent_vertices().size(); i++) {
			geodesic::Vertex*& v = basepoint.base_element()->adjacent_vertices()[i];
			addToCandidtates(v);
			is_computed[v->id()] = true;
			is_computed_tmp[v->id()] = true;

		}
		/*for (unsigned i = 0; i < basepoint.base_element()->adjacent_faces().size(); i++) {
			valid_triangle[basepoint.base_element()->adjacent_faces()[i]->id()] = true;
		}*/




		while (!candidates.empty()) {
			geodesic::Vertex* j = getSmallestCandidate();
			for (auto v : j->adjacent_vertices()) {
				
				update_face_id = computeDistAndAngle(v, newUi, newTi, newRho1i, newRho2i);
		
				//f = v->adjacent_faces()[update_face_id];
				//e = f->opposite_edge(v);
				int v_id = v->id();
				if ((U[v_id] > newUi*(1.0 + precision)) & (update_face_id != -1)) {
					// consider havin temporary variables for more safty and
					// update only when the point is validated
					setDist(v, newUi);
					theta[v_id] = newTi;
					rho1[v_id] = newRho1i;
					rho2[v_id] = newRho2i;
					/*cos_t[v_id] = cos(theta[v_id]);
					sin_t[v_id] = sin(theta[v_id]);
					x[v_id] = U[v_id] * cos_t[v_id];
					y[v_id] = U[v_id] * sin_t[v_id];*/
					if (newUi < radius) {
						// update candidates
						is_computed[v->id()] = true;
						/*switch (stop_criterion) {
						case SELF_COLLISION_RADIAL:
							stop = isCollisionPoint(v);
							break;
						case ORIENTATION_RADIAL:
							normal = VertexNormals.row(v->id()).transpose();
							stop = (normal.dot(n_bp) <= 0.0);
							break;
						default:
							stop = false;
							break;
						}*/

						/*if (stop) {
							is_computed[v->id()] = false;
							radius = std::min(radius, U[j->id()]);
						}*/
						//else {
							addToCandidtates(v);
						//}	
					}
				}
			} // for (auto v : j->adjacent_vertices())
		}

		
		

		// an other possibility is to store computed vertices in increasing order
		// go through this list and check validity of adjacent faces
		// stop as soon as we find an invalid face 
		// cant do it in one pass cuz not all triangles are visited 



		/*for (unsigned i = 0; i < nv; i++) {
			is_computed_tmp[i] = is_computed[i];
		}
		for (unsigned i = 0; i < nv; i++) {
			is_computed_tmp[i] = (is_computed[i] & !isCollisionPoint(&(mesh.vertices()[i])));
		}*/
		/*if ((stop_criterion == ORIENTATION_GLOBAL) || (stop_criterion == ORIENTATION_RADIAL)) {
			for (unsigned i = 0; i < nv; i++) {
				geodesic::vertex_pointer v = &(mesh.vertices()[i]);
				normal = VertexNormals.row(v->id()).transpose();
				is_computed[i] = (is_computed_tmp[i] & (normal.dot(n_bp) > 0.0));
				is_computed_tmp[i] = is_computed[i];
			}
		}
		else {
			for (unsigned i = 0; i < nv; i++) {
				is_computed[i] = is_computed_tmp[i];
			}
		}*/
		
		//extractConnectedComponent();

		

		
		duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
		//return safety_radius;
		/*if ((stop_criterion == ORIENTATION_GLOBAL) || (stop_criterion == SELF_COLLISION_GLOBAL)) {
			radius = rad;
			return rad;
		}
		else {
			for (unsigned i = 0; i < nv; i++) {
				geodesic::vertex_pointer v = &(mesh.vertices()[i]);
				if (!isInteriorPoint(v)) {
					radius = std::min(radius, U[v->id()]); 
				}
			}
		}*/

		
		// now we can compute the euclidean coordinate matrix and its indices
		std::fill(mesh_idx.begin(), mesh_idx.end(), -1);
		nb_valid = 0;
		for (unsigned i = 0; i < nv; i++) {
			is_computed[i] = is_computed[i] & (U[i] <= radius);
			mesh_idx[nb_valid] = i;
			patch_idx[i] = is_computed[i]*nb_valid - (1 - is_computed[i]);

			/*coordMat(nb_valid, 0) = x[i];
			coordMat(nb_valid, 1) = y[i];*/
			nb_valid += is_computed[i];
		}
		eucl_coords.resize(nb_valid, 2);
		for (int i = 0; i < nb_valid; i++) {
			eucl_coords(i, 0) = U[mesh_idx[i]] * cos(theta[mesh_idx[i]]);
			eucl_coords(i, 1) = U[mesh_idx[i]] * sin(theta[mesh_idx[i]]);
		}

		// construct kd-tree

		mat_index = new kd_tree_d(eucl_coords, 10); // max leaf = 10
		mat_index->index->buildIndex();
		

		//coordMat.conservativeResize(nb_valid, 2); // pb here

		return radius;
	}// compute

	

	void getPatchBin(double r, double t, 
		Eigen::Vector3i& contrib_idx, 
		Eigen::Vector3d& bar_coords, 
		Eigen::Vector3d& transport,
		double& t_x) {

		contrib_idx.setConstant(bp_id);
		bar_coords.setZero();
		transport.setZero();

		if (is_boundary) {
			return;
		}
		
		Eigen::Vector2d x;
		x(0) = cos(t)*r;
		x(1) = sin(t)*r;

		Eigen::Vector3i f_sort_idx(0, 1, 2);
		geodesic::face_pointer f = findSurroundingFace(x, bar_coords, f_sort_idx);
		bool bnd = false;
		for (int i = 0; i < 3; i++) {
			contrib_idx(i) = f->adjacent_vertices()[i]->id();
			if (is_computed[contrib_idx(i)]) {
				if (isnan(rho1[contrib_idx(i)])) {
					transport(i) = 0.0;
				}
				else {
					transport(i) = rho1[contrib_idx(i)];
				}
			}
			else {
				//cout << "uuuuuuuuuuu" << endl;
				transport(i) = 0.0;
				bar_coords(i) = 0.0;
				bnd = true;
			}
			
			if (this->isMeshBoundaryPoint(contrib_idx(i))) {
				bar_coords(i) = 0.0;
				transport(i) = 0.0;
				bnd = true;
			}

			if (fabs(transport(i)) >= 2.*M_PI) {
				transport(i) = 0.0;
			}
		}

		if (bnd) {
			int contrib = 0;
			for (int i = 2; i >= 0; i--) {
				if (is_computed[contrib_idx(i)] && !this->isMeshBoundaryPoint(contrib_idx(i))) {
					contrib = contrib_idx(i);
				}
			}
			
			for (int i = 2; i >= 0; i--) {
				if (!(is_computed[contrib_idx(i)] && !this->isMeshBoundaryPoint(contrib_idx(i)))) {
					contrib_idx(i) = contrib;
				}
			}

			if (bar_coords.norm() > 0.01) {
				bar_coords.normalize();
			}

			/*for (int i = 0; i < 3; i++) {
				if (is_computed[contrib_idx(i)]) {
					std::swap(contrib_idx(i), contrib_idx(0));
					std::swap(bar_coords(i), bar_coords(0));
					std::swap(transport(i), transport(0));
					return;
				}
			}*/
		}


		
		// Eigen::Vector3d Rho1;
		Eigen::Vector3d Rho2;
		double t2;
		// transport
		t_x = 0.0;
		// computeDistAndAngleFromInterior(t, bar_coords, f, transport, Rho2, t_x, t2);
		
		reindex_vector(contrib_idx, f_sort_idx);
		reindex_vector(bar_coords, f_sort_idx);
		reindex_vector(transport, f_sort_idx);//*/
	}

	void getPatch(Patch& patch) const {	
		// construct local embedding matrix
		Eigen::Matrix3d local_frame;
		Eigen::Matrix3d local_frame_inv;
		Eigen::Vector3d bp = Vertices.row(bp_id).transpose();
		Eigen::Vector3d x;
		Eigen::Vector3d y;
		Eigen::Vector3d u;
		Eigen::Vector3d v;
		Eigen::Vector3d n;
		x = Vertices.row(bp_id).transpose();
		u = Vertices.row(oneRingVertices[bp_id][0]).transpose() - x;
		u.normalize();
		n = VertexNormals.row(bp_id).transpose();
		n.normalize();
		u -= n.dot(u)*n;
		u.normalize();
		v = n.cross(u);
		local_frame.col(0) = u;
		local_frame.col(1) = v;
		local_frame.col(2) = n;
		local_frame_inv = local_frame.inverse();
		/*for (int i = 0; i < nb_valid; i++) {
			y = Vertices.row(mesh_idx[i]).transpose() - x;
			local_embedd(i, 0) = u.dot(y);
			local_embedd(i, 1) = v.dot(y);
			local_embedd(i, 2) = n.dot(y);
		}*/
		patch.setPatch(nb_valid, mesh_idx, mu, U, theta, rho1, rho2, local_frame_inv, x, radius);
	}

	




	/*void testOrientation(std::vector<double>& test) const {
		test.resize(nv);
		for (unsigned i = 0; i < nv; i++) {
			geodesic::vertex_pointer v = &(mesh.vertices()[i]);
		}
	}*/



	void displayDuration() const {
		cout << "execution took " << duration << endl;
	}

	int nbVertices() const { return nv; }
	int nbFaces() const { return nf; }
	int nbValid() const { return nb_valid; }
	const std::vector<double>& getDist() const { return U; }
	const std::vector<double>& getAngle() const { return theta; }
	 

	const std::vector<bool>& getState() const { return is_computed; }


	//const Eigen::MatrixXd& getCoords() const { return coordMat; }// coordMat.block(0, 0, nb_valid, 2);}
	const std::vector<int>& getValidIdx() const { return mesh_idx; }
	const std::vector<int>& getValidIdxInverse() const { return patch_idx; }


	
	//unsigned int getNbValid() const { return nb_valid; }
	//const std::vector<bool>& isValidTriangle() const { return valid_triangle; }

	bool isMeshBoundaryPoint(int id) const {
		return (mesh.vertices()[id].adjacent_edges().size() > mesh.vertices()[id].adjacent_faces().size());
	}

	bool isMeshBoundaryFace(int id) const {
		int a = mesh.faces()[id].adjacent_vertices()[0]->id();
		int b = mesh.faces()[id].adjacent_vertices()[1]->id();
		int c = mesh.faces()[id].adjacent_vertices()[2]->id();
		return (this->isMeshBoundaryPoint(a) || 
			this->isMeshBoundaryPoint(b) || 
			this->isMeshBoundaryPoint(c));
	}

	const Eigen::MatrixXd& getNormals() const {
		return VertexNormals;
	}

protected:


private:
	kd_tree_d * mat_index;
	Eigen::MatrixXd eucl_coords;
	Eigen::MatrixXd bar_coords;
	Eigen::MatrixXd Vertices; // store the vertices coordinates in matrix format for conveinience
	Eigen::MatrixXi Faces;
	Eigen::MatrixXd VertexNormals;
	geodesic::Mesh& mesh;
	bool is_boundary;
	std::multimap<double, geodesic::Vertex* > candidates;
	std::vector<double> mu;
	//double radius;
	std::vector<double> U;
	std::vector<double> theta;
	std::vector<double> rho1;
	std::vector<double> rho2;
	std::vector<double> cos_t;
	std::vector<double> sin_t;
	std::vector<double> x;
	std::vector<double> y;
	//Eigen::Matrix<double, -1, 2> coordMat;
	std::vector<int> mesh_idx;
	std::vector<int> patch_idx;
	unsigned int nb_valid;

	std::vector< std::multimap<double, geodesic::Vertex* >::iterator > candidate_position;
	std::vector<bool> is_candidate;
	std::vector<bool> is_computed;
	std::vector<bool> is_computed_tmp;
	
	std::vector<bool> valid_triangle;
	std::vector< std::vector< int > > oneRingVertices;
	std::vector< std::vector< int > > oneRingEdges;
	std::vector< std::vector< int > > oneRingFaces;
	std::vector< std::vector<int> > oneRingVinF; // possibly wrong need to check computation
	std::vector< std::vector<double> > oneRingIncidentAngles;
	std::vector< std::vector<double> > oneRingAngles;
	std::vector< double > totalAngleAtVertex;
	Eigen::Matrix<double, -1, 3> Differential;
	//std::vector< double > maxRadiusAtAngle;
	//std::vector< bool > maxRadiusAtAngle;
	double duration;
	//int nv;
	int nf;
	//int bp_id;

	inline bool is_valid_face(geodesic::face_pointer f) const {
		bool res = true;
		for (int i = 0; i < 3; i++) {
			res *= (patch_idx[f->adjacent_vertices()[i]->id()] != -1);
		}
		return res;
	}

	inline geodesic::face_pointer findSurroundingFace(const Eigen::Vector2d& x, Eigen::Vector3d& x_bar, Eigen::Vector3i& f_sort_idx) {
		Eigen::Vector2d a;
		Eigen::Vector2d b;
		Eigen::Vector2d c;
		double na;
		double nb;
		double nc;
		f_sort_idx(0) = 0;
		f_sort_idx(1) = 1;
		f_sort_idx(2) = 2;

		int nb_neighs = std::min((int)(nb_valid), 6);

		if (nb_neighs == 0) {
			//cout << "uu" << endl;
			x_bar.setZero();
			return &mesh.faces()[0];
		}

		
		std::vector<std::size_t> index(nb_neighs);
		std::vector<double> dists_sqr(nb_neighs);
			// do a knn search
		

		std::vector<double> query_pt(2);
		query_pt[0] = x(0);
		query_pt[1] = x(1);

		//std::vector<double> out_dists_sqr(nb_neighs);

		nanoflann::KNNResultSet<double> resultSet(nb_neighs);

		resultSet.init(&index[0], &dists_sqr[0]);
		mat_index->index->findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));//*/

		geodesic::face_pointer f = NULL;

		for (int i = 0; i < nb_neighs; i++) {
			for (int j = 0; j < mesh.vertices()[mesh_idx[index[i]]].adjacent_faces().size(); j++) {
				f = mesh.vertices()[mesh_idx[index[i]]].adjacent_faces()[j];
				if (is_valid_face(f)) {
					a = eucl_coords.row(patch_idx[f->adjacent_vertices()[0]->id()]).transpose();
					b = eucl_coords.row(patch_idx[f->adjacent_vertices()[1]->id()]).transpose();
					c = eucl_coords.row(patch_idx[f->adjacent_vertices()[2]->id()]).transpose();
					na = a.squaredNorm();
					nb = b.squaredNorm();
					nc = c.squaredNorm();
					barycentric_coords_(a, b, c, x, x_bar);
					if ((x_bar(0) >= 0.) && (x_bar(1) >= 0.) && (x_bar(2) >= 0.)) {
						if (nb > nc) {
							std::swap(nb, nc);
							std::swap(f_sort_idx(1), f_sort_idx(2));
						}
						if (na > nc) {
							std::swap(na, nc);
							std::swap(f_sort_idx(0), f_sort_idx(2));
						}
						if (na > nb) {
							std::swap(na, na);
							std::swap(f_sort_idx(0), f_sort_idx(1));
						}
						return f;
					}	
				}
			}
		}

		x_bar.setZero();
		f = mesh.vertices()[mesh_idx[index[0]]].adjacent_faces()[0];
		for (int i = 0; i < 3; i++) {
			if (f->adjacent_vertices()[i]->id() == mesh_idx[index[0]]) {
				x_bar(i) = 1.0;
				return f;
			}
		}
		
	}


	inline bool isInteriorPoint(geodesic::vertex_pointer v) {
		if (v->adjacent_faces().size() != v->adjacent_faces().size()) {
			return false;
		}
		if (!is_computed[v->id()]) {
			return false;
		}
		for (auto n : v->adjacent_vertices()) {
			if (!is_computed[n->id()]) {
				return false;
			}
		}
		return true;
	}

	inline bool orientationCompatibility(
		geodesic::vertex_pointer v1,
		geodesic::vertex_pointer v2) {

		if (this->isMeshBoundaryPoint(v1->id()) || this->isMeshBoundaryPoint(v2->id())) {
			return true;
		}

		// first find the id of v2 in vertices adjacent to v1 (same for v1)
		int v1_id_in_v2 = -1;
		int v2_id_in_v1 = -1;
		//int nb_neigh1 = v1->adjacent_vertices().size();
		//int nb_neigh2 = v2->adjacent_vertices().size();
		int nb_neigh1 = oneRingVertices[v1->id()].size();
		int nb_neigh2 = oneRingVertices[v2->id()].size();

		for (unsigned i = 0; i < nb_neigh1; i++) {
			if (v2->id() == oneRingVertices[v1->id()][i]) {
				v2_id_in_v1 = i;
			}
		}
		for (unsigned i = 0; i < nb_neigh2; i++) {
			if (v1->id() == oneRingVertices[v2->id()][i]) {
				v1_id_in_v2 = i;
			}
		}
		if ((v2_id_in_v1 == -1) || (v1_id_in_v2 == -1)) {
			cout << "not neighbours" << endl;
			return true;
		}
		/*bool t1 = (oneRingVertices[v1->id()][(v2_id_in_v1 + 1) % nb_neigh1] == oneRingVertices[v2->id()][(v1_id_in_v2 + 1) % nb_neigh2]);
		bool t2 = (oneRingVertices[v1->id()][(v2_id_in_v1 + nb_neigh1 - 1) % nb_neigh1] == oneRingVertices[v2->id()][(v1_id_in_v2 + 1) % nb_neigh2]);
		if (!(t1 || t2)) {
			cout << "zzzzzzzz" << endl;
		}*/
		// find relative orientation of tangent planes to v1 and v2
		return (oneRingVertices[v1->id()][(v2_id_in_v1 + 1) % nb_neigh1] != oneRingVertices[v2->id()][(v1_id_in_v2 + 1) % nb_neigh2]);
	}

	inline void reverseOrientation(geodesic::vertex_pointer v) {
		int v_id = v->id();
		/*int deg = v->adjacent_vertices().size();
		if (deg == v->adjacent_faces().size()) {
			return;
		}
		int idx_tmp;
		double val_tmp;
		for (int i = 0; i < deg; i++) {
			std::swap(oneRingFaces[v_id][deg - i], oneRingFaces[v_id][i]);
			std::swap(oneRingIncidentAngles[v_id][deg - i], oneRingIncidentAngles[v_id][i]);
			std::swap(oneRingVinF[v_id][deg - i], oneRingVinF[v_id][i]);
		}


		for (int i = 0; i < deg-1; i++) {
			std::swap(oneRingVertices[v_id][deg - i], oneRingVertices[v_id][i+1]);
			std::swap(oneRingEdges[v_id][deg - i], oneRingEdges[v_id][i+1]);
		}


		oneRingAngles[v_id][0] = 0.0;
		for (int i = 1; i < deg; i++) {
			oneRingAngles[v_id][i] = oneRingAngles[v_id][i-1] + oneRingIncidentAngles[v_id][i-1];
		}*/
		set1RingOrder(v,
			oneRingFaces[v_id],
			oneRingEdges[v_id],
			oneRingVertices[v_id],
			oneRingVinF[v_id],
			oneRingIncidentAngles[v_id],
			oneRingAngles[v_id],
			totalAngleAtVertex[v_id],
			false);
	}
	bool isMeshInteriorPoint_(int id) const {
		if (this->isMeshBoundaryPoint(id)) {
			return false;
		}
		for (int i = 0; i < mesh.vertices()[id].adjacent_vertices().size(); i++) {
			if (this->isMeshBoundaryPoint(mesh.vertices()[id].adjacent_vertices()[i]->id())) {
				return false;
			}
		}
		return true;
	}
	void computeVerticesNormals() {
		VertexNormals.resize(nv, 3);
		geodesic::vertex_pointer v;
		Eigen::Vector3d e1;
		Eigen::Vector3d e2;
		Eigen::Vector3d e3;
		Eigen::Vector3d normal;
		int pt_id;
		int n_id;
		int deg;
		//igl::per_vertex_normals(Vertices, Faces, VertexNormals);
		for (unsigned i = 0; i < nv; i++ ) {
			pt_id = mesh.vertices()[i].id();
			deg = mesh.vertices()[i].adjacent_vertices().size();
			normal.setZero();
			if (deg == mesh.vertices()[i].adjacent_faces().size()) {
				for (unsigned j = 0; j < deg; j++) {
					n_id = oneRingVertices[pt_id][j];
					e1 = (Vertices.row(n_id) - Vertices.row(pt_id)).transpose();
					e1.normalize();
					n_id = oneRingVertices[pt_id][(j + 1) % deg];
					e2 = (Vertices.row(n_id) - Vertices.row(pt_id)).transpose();
					e2.normalize();
					e3 = e1.cross(e2);
					e3.normalize();
					normal += oneRingAngles[pt_id][j] * e3;
				}
				normal.normalize();
			}
			
			//cout << normal.dot(VertexNormals.row(pt_id).transpose()) << endl;
			
			/*if (normal.dot(VertexNormals.row(pt_id).transpose()) < 0.0) {
				VertexNormals.row(pt_id) = -normal.transpose();
			}
			else {
				VertexNormals.row(pt_id) = normal.transpose();
			}*/
			VertexNormals.row(pt_id) = normal.transpose();
		}

		// make normals compatible
		std::multimap<double, geodesic::vertex_pointer> trial;
		for (unsigned i = 0; i < nv; i++) {
			is_computed_tmp[i] = false;
		}

		// init

		// find interior point
		int i_ = 0;
		for (int i = 0; i < nv; i++) {
			if (this->isMeshInteriorPoint_(i)) {
				i_ = i;
				i = nv;
				//exit(666);
			}
		}
		v = &(mesh.vertices()[i_]);



		trial.insert(std::make_pair(0.0, v));
		is_computed_tmp[0] = true;
		double uv = 0.0;
		while (!trial.empty()) {
			v = (trial.begin())->second;
			uv = (trial.begin())->first;
			trial.erase(trial.begin());

			for (auto j : v->adjacent_vertices()) {
				if (!is_computed_tmp[j->id()]) {
					if (!orientationCompatibility(v, j)) {
					//if (VertexNormals.row(v->id()).transpose().dot(VertexNormals.row(j->id()).transpose()) < 0.0) {
						//cout << "ee" << endl;
						//cout << orientationCompatibility(v, j) << endl;
						reverseOrientation(j);
						//cout << orientationCompatibility(v, j) << endl;
						VertexNormals.row(j->id()) = -VertexNormals.row(j->id());

					}
					else {
						//cout << orientationCompatibility(v, j) << endl;
					}

					trial.insert(std::make_pair(uv+1.0, j));
					is_computed_tmp[j->id()] = true;
				}
			}
		}
		for (unsigned i = 0; i < nv; i++) {
			is_computed_tmp[i] = false;
		}
		trial.clear();
	}

	void extractConnectedComponent() {
		std::multimap<double, geodesic::vertex_pointer> trial;
		for (unsigned i = 0; i < nv; i++) {
			is_computed_tmp[i] = is_computed[i];
			is_computed[i] = false;
		}
		// init 
		geodesic::vertex_pointer v = &(mesh.vertices()[bp_id]);
		trial.insert(std::make_pair(U[bp_id], v));
		is_computed[bp_id] = true;
		for (auto j : v->adjacent_vertices()) {
			trial.insert(std::make_pair(U[j->id()], j));
			is_computed[j->id()] = true;
		}
		while (!trial.empty()) {
			v = (trial.begin())->second;
			trial.erase(trial.begin());

			for (auto j : v->adjacent_vertices()) {
				if (is_computed_tmp[j->id()] & (!is_computed[j->id()])){
					trial.insert(std::make_pair(U[j->id()], j));
					is_computed[j->id()] = true;
				}
			}	
		}
		trial.clear();

	}

	inline bool isCollisionPoint(geodesic::vertex_pointer v) {
		if (!isInteriorPoint(v)) {
			return false;
		}
		int deg = v->adjacent_vertices().size();
		int v_id = v->id();
		const std::vector<int>& neigh = oneRingVertices[v_id];
		Eigen::Vector3d ei;
		Eigen::Vector3d ej;
		int j = 0;
		bool pos0;
		double z;
		ei << (x[neigh[0]] - x[v_id]), (y[neigh[0]] - y[v_id]), 0.0;
		ej << (x[neigh[1]] - x[v_id]), (y[neigh[1]] - y[v_id]), 0.0;
		z = (ei.cross(ej))(2);
		if (z == 0.0) {
			//cout << "uu" << endl;
			return true;
		}
		pos0 = (z > 0.0);
		for (int i = 1; i < deg; i++) {
			ei << (x[neigh[i]] - x[v_id]), (y[neigh[i]] - y[v_id]), 0.0;
			j = (i + 1) % deg;
			ej << (x[neigh[j]] - x[v_id]), (y[neigh[j]] - y[v_id]), 0.0;
			z = (ei.cross(ej))(2);
			if (z == 0.0) {
				//cout << "uu" << endl;
				return true;
			}
			if ((z > 0.0) != pos0) {
				return true;
			}
		}
		return false;
	}



	void computeDifferential() {
		Eigen::Matrix3d P;
		Eigen::Matrix2d Q;
		Eigen::Matrix<double, 2, 3> M;
		M << 1.0, 0.0, 0.0,
			0.0, 1.0, 0.0;
		geodesic::face_pointer f;
		int a_id;
		int b_id;
		int c_id;
		for (unsigned i = 0; i < nf; i++) {
			f = &(mesh.faces()[i]);
			a_id = f->adjacent_vertices()[0]->id();
			b_id = f->adjacent_vertices()[1]->id();
			c_id = f->adjacent_vertices()[2]->id();
			if (is_computed[a_id] & is_computed[b_id] & is_computed[c_id]) {
				P.col(0) = (Vertices.row(b_id) - Vertices.row(a_id)).transpose();
				P.col(1) = (Vertices.row(c_id) - Vertices.row(a_id)).transpose();
				P.col(2) = P.col(0).cross(P.col(1));
				P.col(2).normalize();
				
				Q << x[b_id] - x[a_id], x[c_id] - x[a_id],
					y[b_id] - y[a_id], y[c_id] - y[a_id];
				
				Differential.block<2, 3>(2 * i, 0) = Q*M*P.inverse();
				//cout << Differential.block<2, 3>(2 * i, 0) << endl;
			}	
		}
	}


	void initialize1RingNeighbourhood(geodesic::base_pointer basepoint) {
		bp_id = basepoint->id();
		geodesic::face_pointer f;
		geodesic::vertex_pointer bp = &(mesh.vertices()[bp_id]);
		geodesic::vertex_pointer v;
		int v_id = 0;
		Eigen::Vector3d e;

		
		int nb_adj_faces = oneRingFaces[bp_id].size();
		int nb_neigh = oneRingVertices[bp_id].size();

		U[bp_id] = 0.0;
		theta[bp_id] = 0.0;
		rho1[bp_id] = 0.0;
		rho2[bp_id] = PI/2.0;
		cos_t[bp_id] = 1.0;
		sin_t[bp_id] = 0.0;
		x[bp_id] = 0.0;
		y[bp_id] = 0.0;

		double rho1New;
		double rho2New;

		for (unsigned i = 0; i < nb_neigh; i++) {
			//cout << "v_id " << oneRingVertices[bp_id][i] << endl;
			v = &(mesh.vertices()[oneRingVertices[bp_id][i]]);
			v_id = v->id();
			rho1[v_id] = transferAngle(
				rho1[bp_id],
				bp,
				v,
				oneRingAngles[bp_id],
				oneRingAngles[v->id()],
				oneRingVertices[bp_id],
				oneRingVertices[v->id()]);
			rho2[v_id] = transferAngle(
				rho2[bp_id],
				bp,
				v,
				oneRingAngles[bp_id],
				oneRingAngles[v->id()],
				oneRingVertices[bp_id],
				oneRingVertices[v->id()]);

			
			theta[v_id] = oneRingAngles[bp_id][i];
			e = Vertices.row(v_id).transpose() - Vertices.row(bp_id).transpose();
			U[v_id] = e.norm();
			cos_t[v_id] = cos(theta[v_id]);
			sin_t[v_id] = sin(theta[v_id]);
			x[v_id] = U[v_id] * cos_t[v_id];
			y[v_id] = U[v_id] * sin_t[v_id];
		}

		/*for (unsigned i = 0; i < nb_neigh; i++) {
			cout << "theta " << theta[oneRingVertices[bp_id][i]] << endl;
			cout << "u1 " << rho1[oneRingVertices[bp_id][i]] << endl;
			cout << "u2 " << rho2[oneRingVertices[bp_id][i]] << endl;
			cout << endl;
		}*/

	}

	void initializeEdge(geodesic::SurfacePoint& basepoint) {
		geodesic::base_pointer e = basepoint.base_element();
		int edge_id = e->id();
		int v1_id = e->adjacent_vertices()[0]->id();
		int v2_id = e->adjacent_vertices()[1]->id();

		theta[v1_id] = 0.0;
		theta[v2_id] = PI;
		
		Eigen::Vector3d a;
		a << basepoint.x(), basepoint.y(), basepoint.z();
		Eigen::Vector3d b;
		b << mesh.vertices()[v1_id].x(),
			mesh.vertices()[v1_id].y(),
			mesh.vertices()[v1_id].z();

		U[v1_id] = (b-a).norm();

		b << mesh.vertices()[v2_id].x(),
			mesh.vertices()[v2_id].y(),
			mesh.vertices()[v2_id].z();

		U[v2_id] = (b - a).norm();
	}
	
	bool initializeNeighbourhood(geodesic::SurfacePoint& basepoint) {
		
		switch (basepoint.type()) {
		case geodesic::PointType::VERTEX:
			if (isMeshBoundaryPoint(basepoint.base_element()->id())) {
				// cout << "boundary point" << endl;
				return false;
			}
			initialize1RingNeighbourhood(basepoint.base_element());
			return true;
			break;
		case geodesic::PointType::EDGE:
			//initializeEdge(basepoint);
			return false;
			break;
		case geodesic::PointType::FACE:
			return false;
			break;
		default:
			return false;
			break;
		}
		return true;
	}

	void addToCandidtates(geodesic::Vertex* v){
		int i = v->id();
		if (!is_candidate[i]) {
			candidate_position[i] = candidates.insert(std::make_pair(U[i], v));
			is_candidate[i] = true;
		}
	}

	geodesic::Vertex* getSmallestCandidate() {
		std::multimap<double, geodesic::Vertex* >::iterator it = candidates.begin();
		geodesic::Vertex* v = it->second;
		int i = v->id();
		candidates.erase(it);
		is_candidate[i] = false;
		// find the fist valid candidate
		/*while ((!is_candidate[i]) || (U[i] != it->first)) {
			candidates.erase(it);
			it = candidates.begin();
			i = it->second->id();
		}*/
		// find the last valid candidate with this value
		return v;
	}

	void setDist(geodesic::Vertex* v, double u){
		int i = v->id();
		if (is_candidate[i]) {
			candidates.erase(candidate_position[i]);
			candidate_position[i] = candidates.insert(std::make_pair(u, v));
		}
		//else {
			U[i] = u;
		//}
	}

	// compute distance angle to an interior point and transfer to vertices of the face
	// return false if one of the vertices of the face is on the boundary
	void computeDistAndAngleFromInterior(
		double t,
		const Eigen::Vector3d& x_bar, 
		geodesic::face_pointer f, 
		Eigen::Vector3d& Rho1, 
		Eigen::Vector3d& Rho2,
		double& t1,
		double& t2,
		bool out=false) const {

		if (!this->is_valid_face(f)) {
			for (int i = 0; i < 3; i++) {
				Rho1(i) = rho1[f->adjacent_vertices()[i]->id()];
				Rho2(i) = rho2[f->adjacent_vertices()[i]->id()];
			}
			t1 = 0;
			t2 = 0;
			return;
		}


		//double u;
		//double theta;

		double alpha;

		// find angular sector containing the interior point
		std::vector<bool> ang_sec(3);
		for (int l = 0; l < 3; l++) {
			ang_sec[l] = is_in_angular_sector(
				t, theta[f->adjacent_vertices()[(l + 1) % 3]->id()],
				theta[f->adjacent_vertices()[(l + 1) % 3]->id()]);
		}

		double max_ = -std::numeric_limits<double>::max();
		int i_ = -1;
		for (int l = 0; l < 3; l++) {
			if (ang_sec[l] && max_ < U[f->adjacent_vertices()[l]->id()] ) {
				max_ = U[f->adjacent_vertices()[l]->id()];
				i_ = l;
			}
		}

		int i;
		int j;
		int k;

		int j_ = 1;
		int k_ = 2;

		// sort w.r.t U
		if (i_ == -1) {
			i_ = 0;
			i = f->adjacent_vertices()[0]->id();
			j = f->adjacent_vertices()[1]->id();
			k = f->adjacent_vertices()[2]->id();
			if (U[i] < U[j]) {
				std::swap(i, j);
				std::swap(i_, j_);
			}

			if (U[i] < U[k]) {
				std::swap(i, k);
				std::swap(i_, k_);
			}
		}
		else {
			i = f->adjacent_vertices()[i_]->id();
			j = f->adjacent_vertices()[(i_ + 1) % 3]->id();
			k = f->adjacent_vertices()[(i_ + 2) % 3]->id();
		}

		if (U[j] < U[k]) {
			std::swap(j, k);
			std::swap(j_, k_);
		}


		/*if (U[i] < std::numeric_limits<double>::min()) {
		return;
		}*/

		Eigen::Vector3d v_i = Vertices.row(i).transpose();
		Eigen::Vector3d vj = Vertices.row(j).transpose();
		Eigen::Vector3d vk = Vertices.row(k).transpose();

		Eigen::Vector3d vi = x_bar(i_)*v_i + x_bar(j_)*vj + x_bar(k_)*vk;

		Eigen::Vector3d ej = vj - vi;
		Eigen::Vector3d ek = vk - vi;
		Eigen::Vector3d ekj = vk - vj;

		double rhov1 = 0.0;
		double rhov2 = 0.0;
		double uv = 0.0;
		double tv = 0.0;

		double A = (ej.cross(ek)).norm();
		double ekj2 = ekj.squaredNorm();
		double H = sqrt((ekj2 - pow(U[j] - U[k], 2))*(pow(U[j] + U[k], 2) - ekj2));
		double xj = (A*(ekj2 + U[k] * U[k] - U[j] * U[j]) + ek.dot(ekj)*H) / (2.*A*ekj2);
		double xk = (A*(ekj2 - U[k] * U[k] + U[j] * U[j]) - ej.dot(ekj)*H) / (2.*A*ekj2);

		Eigen::Vector3d s_p = vi + xj * ej + xk * ek;
		Eigen::Vector3d a = vj - s_p;
		a.normalize();
		Eigen::Vector3d b = vi - s_p;
		b.normalize();
		double phi_ij = acos(a.dot(b));
		b = vk - s_p;
		b.normalize();
		double phi_kj = acos(a.dot(b));

		

		//orthonormalize(rhoj1, rhoj2);
		//orthonormalize(rhok1, rhok2)
		int intp = 0;
		if ((xj > 0.) && (xk > 0.)) {
			uv = (s_p - vi).norm();
			if (phi_ij > phi_kj) {
				alpha = phi_kj / phi_ij;
				intp = 0;
			}
			else {
				alpha = phi_ij / phi_kj;
				intp = 1;
			}

		}
		else {
			if (U[j] + ej.norm() > U[k] + ek.norm()) {
				intp = 2;
			}
			else {
				intp = 3;
			}
		}

		transportToFace(
			x_bar,
			f,
			Rho1,
			Rho2,
			t1,
			t2,
			alpha,
			intp,
			j_,
			k_,
			rho1,
			rho2,
			oneRingAngles,
			oneRingVertices,
			Vertices);//*/
	}

	// return false if one of the vertices of the face is on the boundary
	bool computeDistAndAngleFromFace(geodesic::Vertex* v, int f_id, double& u, double& t, double& Rho1, double& Rho2) {
		//double u;
		//double theta;
		geodesic::face_pointer f = v->adjacent_faces()[f_id];
		if (this->isMeshBoundaryPoint(v->id()) || this->isMeshBoundaryFace(f->id())) {
			t = 0.0;
			Rho1 = 0.0; 
			Rho2 = 0.0;
			u = std::numeric_limits<double>::max();
			return false;

		}
		double alpha;

		
		int i;
		int j;
		int k;
		for (int l = 0; l < 3; l++) {
			if (f->adjacent_vertices()[l]->id() == v->id()) {
				i = l;
			}
		}
		j = (i + 1) % 3;
		k = (i + 2) % 3;

		geodesic::vertex_pointer pvi = f->adjacent_vertices()[i];
		geodesic::vertex_pointer pvj = f->adjacent_vertices()[j];
		geodesic::vertex_pointer pvk = f->adjacent_vertices()[k];

		i = pvi->id();
		j = pvj->id();
		k = pvk->id();

		
		

		/*if (U[i] < std::numeric_limits<double>::min()) {
			return;
		}*/

		Eigen::Vector3d vi = Vertices.row(i).transpose();
		Eigen::Vector3d vj = Vertices.row(j).transpose();
		Eigen::Vector3d vk = Vertices.row(k).transpose();

		Eigen::Vector3d ej = vj - vi;
		Eigen::Vector3d ek = vk - vi;
		Eigen::Vector3d ekj = vk - vj;


		double A = (ej.cross(ek)).norm();
		double ekj2 = ekj.squaredNorm();
		double H = sqrt( (ekj2 - pow(U[j]-U[k],2))*(pow(U[j] + U[k],2) - ekj2) );
		double xj = (A*(ekj2 + U[k]*U[k] - U[j]*U[j])+ek.dot(ekj)*H) / (2.*A*ekj2);
		double xk = (A*(ekj2 - U[k] * U[k] + U[j] * U[j]) - ej.dot(ekj)*H) / (2.*A*ekj2);

		Eigen::Vector3d s_p = vi + xj*ej + xk*ek;
		Eigen::Vector3d a = vj - s_p;
		a.normalize();
		Eigen::Vector3d b = vi - s_p;
		b.normalize();
		double phi_ij = acos(a.dot(b));
		b = vk - s_p;
		b.normalize();
		double phi_kj = acos(a.dot(b));

		double rhoj1 = transferAngle(rho1[j], pvj, pvi, oneRingAngles[j], oneRingAngles[i], oneRingVertices[j], oneRingVertices[i]);
		double rhok1 = transferAngle(rho1[k], pvk, pvi, oneRingAngles[k], oneRingAngles[i], oneRingVertices[k], oneRingVertices[i]);
		double rhoj2 = transferAngle(rho2[j], pvj, pvi, oneRingAngles[j], oneRingAngles[i], oneRingVertices[j], oneRingVertices[i]);
		double rhok2 = transferAngle(rho2[k], pvk, pvi, oneRingAngles[k], oneRingAngles[i], oneRingVertices[k], oneRingVertices[i]);
		//orthonormalize(rhoj1, rhoj2);
		//orthonormalize(rhok1, rhok2)

		if ((xj > 0.) && (xk > 0.)) {
			u = (s_p - vi).norm();
			if (phi_ij > phi_kj) {
				alpha = phi_kj / phi_ij;
				//t = principalAngle((1 - alpha)*theta[k] + alpha*theta[j]);//angleMix(1.-alpha, theta[j], theta[k]);
				t = angleMix(alpha, theta[k], theta[j]);
				Rho1 = angleMix(alpha, rhok1, rhoj1);
				Rho2 = angleMix(alpha, rhok2, rhoj2);
			}
			else {
				alpha = phi_ij / phi_kj;
				//t = principalAngle((1 - alpha)*theta[j] + alpha*theta[k]);
				t = angleMix(alpha, theta[j], theta[k]);
				Rho1 = angleMix(alpha, rhoj1, rhok1);
				Rho2 = angleMix(alpha, rhoj2, rhok2);
			}
						
		}
		else {
			if (U[j] + ej.norm() > U[k] + ek.norm()) {
				u = U[k] + ek.norm();
				t = theta[k];
				Rho1 = rhok1;
				Rho2 = rhok2;

			}
			else {
				u = U[j] + ej.norm();
				t = theta[j];
				Rho1 = rhoj1;
				Rho2 = rhoj2;
			}
		}
		//orthonormalize(Rho1, Rho2);
		/*if ((U[j] < std::numeric_limits<double>::min()) || (U[k] < std::numeric_limits<double>::min())) {
			t = theta[i];
		}*/
		return !(this->isMeshBoundaryPoint(i) || this->isMeshBoundaryPoint(j) || this->isMeshBoundaryPoint(k));
	}

	int computeDistAndAngle(geodesic::Vertex* v, double& u, double& t, double& Rho1, double& Rho2) {
		u = std::numeric_limits<double>::max();
		double u_tmp = 0.0;
		double t_tmp = 0.0;
		double r1_tmp = 0.0;
		double r2_tmp = 0.0;
		bool is_not_bd = false;
		int face_id = -1;
		int v_in_f_id;
		geodesic::face_pointer f;
		for (int i = 0; i < v->adjacent_faces().size(); i++) {
			
			f = v->adjacent_faces()[i];
			v_in_f_id = oneRingVinF[v->id()][i];
			geodesic::edge_pointer e = f->opposite_edge(v);

			if(is_computed[e->adjacent_vertices()[0]->id()] & is_computed[e->adjacent_vertices()[1]->id()]){
			//if (is_computed[ f->adjacent_vertices()[(v_in_f_id + 1) % 3]->id() ] &
			//	is_computed[ f->adjacent_vertices()[(v_in_f_id + 2) % 3]->id()] ) {
				
				is_not_bd = computeDistAndAngleFromFace(v, i, u_tmp, t_tmp, r1_tmp, r2_tmp);
				u_tmp = fabs(u_tmp);
				
				if (u > u_tmp) {
					u = u_tmp;
					t = t_tmp;
					if (is_not_bd) {
						Rho1 = r1_tmp;
						Rho2 = r2_tmp;
					}
					face_id = i;
				}
			}
		}
		return face_id;
	}
};
	
#endif