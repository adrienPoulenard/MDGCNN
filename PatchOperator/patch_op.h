#pragma once
#ifndef PATCH_OP_H
#define PATCH_OP_H

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


using namespace std;

inline double angularDist(double t1, double t2) {
	if (t1 > t2) {
		std::swap(t1, t2);
	}
	if (t2 - t1 > M_PI) {
		t1 += 2 * M_PI;
	}
	return fabs(t2 - t1);
}

bool saveTriplets(const std::string& path, const std::map<std::pair<int, int >, float >& T) {
	ofstream myfile(path.c_str());
	if (myfile.is_open())
	{
		for (auto t : T) {
			myfile << t.first.first << " ";
			myfile << t.first.second << " ";
			myfile << t.second;
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

bool saveTriplets(const std::string& path, const std::vector< Eigen::Triplet<float> >& T) {
	ofstream myfile(path.c_str());
	if (myfile.is_open())
	{
		for (auto t : T) {
			myfile << t.row() << " ";
			myfile << t.col() << " ";
			myfile << t.value();
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

template <typename T>
bool saveVector(const std::string& path, const std::vector<T>& v) {
	ofstream myfile(path.c_str());
	
	if (myfile.is_open())
	{
		for (int i = 0; i < v.size(); i++) {
			myfile << v[i];
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















inline bool direct(const Patch& P, int n) {
	return ((cos(P.u1()[n])*sin(P.u2()[n]) - sin(P.u1()[n]) * cos(P.u2()[n])) > 0.0);
}
inline void angular_bin(double& a, int& p, int& q, const Patch& P, int j, int n, int ndir) {

	bool reverse = (direct(P, n) != (j < ndir));
	int shift = 0;
	double theta = P.u1()[n];
	if (theta >= 2.*M_PI) {
		theta -= 2.*M_PI;
	}
	if (reverse) {
		theta = 2.*M_PI - P.u1()[n];
		shift = ndir;
	}
	p = int((theta * ndir) / (2.0*M_PI) + j) % ndir + shift;
	q = (p + 1) % ndir + shift;
	double t1 = theta + (2.*(j%ndir)*M_PI) / ndir;
	if (t1 >= 2.*M_PI) {
		t1 -= 2.*M_PI;
	}
	a = (ndir*angularDist(t1, (2.*(p % ndir)*M_PI) / ndir)) / (2.*M_PI);
	
}

inline int angular_bin(double theta, int ndir) {
	if (theta < 0.0) {
		theta += 2.*M_PI;
	}
	if (theta >= 2.*M_PI) {
		theta -= 2.*M_PI;
	}
	double a = (ndir*theta) / (2.*M_PI);
	int bin = int(a);
	if (a - bin > 0.5) {
		return bin + 1;
	}
	return bin;
}



void computeSinglePatchOperator(int i, int nv, int nb_dir, int nb_rings, const Patch& P, 
								std::list< Eigen::Triplet<float> > & patch_op,
								std::vector< int >& connectivity,
								std::vector< int >& transport,
								std::vector< float >& local_frame3d,
								std::vector< float >& base_point,
								const bool oriented = true) {
	//int np = P.l().size(); // number of vertices in the patch
	
	int nb_frames = nb_dir;
	if (!oriented) {
		nb_frames = 2 * nb_dir;
	}
	
	Eigen::Triplet<float> triplet;
	double M_ijklm = 0.0;
	double K_ijkl = 0.0;
	int p_ijl = 0;
	int q_ijl = 0;
	double a_ijl = 0.0;
	double radius = P.radius();

	double theta;
	double r;
	int shift = 0;

	// loop over patch op indices
	// i: base point index (target)
	// j: frame index (target)
	// k: ring (radius) index (target)
	// l: base point index (source)
	// m: frame index (source)

	//int m = 0;

	//wrong indexig of the matrix M
	//not i*j*k ... but i*nb_dir*nb_rings + j*nb_rings + k

	int I = 0;
	int J = 0;
	int l = 0;

	//std::pair< int, int > IJ;
	//std::pair< std::map<std::pair< int, int >, float >::iterator, bool > insert;
	Eigen::MatrixXf K_mu(nb_dir, nb_rings);
	K_mu.setZero();
	radius += std::numeric_limits<float>::min();
	// bins area
	/*
	for (int n = 0; n < P.l().size(); n++) {
		int k = std::min(int((P.r()[n] * nb_rings) / radius),nb_rings);
		int j = int((P.theta()[n] * nb_dir) / (2.*M_PI)) % nb_dir;
		//K_mu(j, k) += P.mu()[n];
		//K_mu(j, k) += 1.0;
		K_mu(j, k) = 1.0; // normalization in the network def
	}
	// normalization
	
	for (int k = 0; k < nb_rings; k++) {
		for (int j = 0; j < nb_dir; j++) {
			if (K_mu(j, k) > 0.0) {
				//K_mu(j, k) = (2.*k + 1.) / ((nb_dir*nb_rings)*K_mu(j, k));
				//K_mu(j, k) = 1.0; // normalization in the network def
			}	
		}	
	}
	
	for (int n = 0; n < P.l().size(); n++) {
		int k = std::min(int((P.r()[n] * nb_rings) / radius), nb_rings);
		int j = int((P.theta()[n] * nb_dir) / (2.*M_PI)) % nb_dir;
		
		//int l = P.l().size();

		// direct
		//angular_bin(a_ijl, p_ijl, q_ijl, P, j, n, nb_dir);
		//K_ijkl = P.mu()[n] * K_mu(j, k);
		K_ijkl =  K_mu(j, k);

		
		
		if (a_ijl < 0.5) {
			shift = p_ijl;
		}
		else {
			shift = q_ijl;
		}
		
		
		if (oriented) {
			shift = angular_bin(P.u1()[n], nb_dir) % nb_dir;
		}

		//I = nv*(2 * nb_dir*k + j) + i;
		//J = nv*p_ijl + P.l()[n];

		//I = nb_rings*(nb_frames*i + j) + k;
		I = nb_frames*(nb_rings*i + k) + j;
		J = nb_frames*P.l()[n] + shift;
		
		

		//I = nv*(nb_frames*k + j) + i;
		//J = nv*shift + P.l()[n];


		//patch_op.push_back(Eigen::Triplet<float>(I, J, K_ijkl));

		if (!oriented) {
			// add contributors to the reversed frame convolution
		}
		/*J = p_ijl + 2*nb_dir*P.l()[n];
		//cout << K_ijkl << endl;
		patch_op.push_back(Eigen::Triplet<float>(I, J, (1.0 - a_ijl)*K_ijkl));
		
		//J = P.l()[n] + nv*q_ijl;
		J = 2 * nb_dir*P.l()[n] + q_ijl;
		patch_op.push_back(Eigen::Triplet<float>(I, J, a_ijl*K_ijkl));*/
		// reversed
		/*angular_bin(a_ijl, p_ijl, q_ijl, P, j+nb_dir, n, nb_dir);
		K_ijkl = P.mu()[n] * K_mu(j, k);
		I = nv*(2 * nb_dir*k + j) + i;
		J = nv*p_ijl + P.l()[n];
		patch_op.push_back(Eigen::Triplet<float>(I, J, (1.0 - a_ijl)*K_ijkl));
		J = P.l()[n] + nv*q_ijl;
		patch_op.push_back(Eigen::Triplet<float>(I, J, a_ijl*K_ijkl));// 


		//K_mu(j, k);
	} */


	// set local frame and base point
	for (int j = 0; j < 3; j++) {
		base_point[3 * i + j] = P.basePoint()(j);
		for (int k = 0; k < 3; k++) {
			local_frame3d[3 * (3 * i + j) + k] = P.localFrame3d()(j, k);
		}
	}
	

	// make sure each bin has a least one contributor
	std::vector< std::size_t > nn_idx(1);
	std::vector< double > nn_sqr(1);
	for (int k = 0; k < nb_rings; k++) {
		for (int j = 0; j < nb_dir; j++) {
			//if (K_mu(j, k) < 1.0) {
				// add nearest neighbour
				P.getNN((2.*j*M_PI) / nb_dir, (radius*(k + 1.0)) / nb_rings, nn_idx, nn_sqr);
				int n = (int)(nn_idx[0]);
				/*angular_bin(a_ijl, p_ijl, q_ijl, P, j, n, nb_dir);
				if (a_ijl < 0.5) {
					shift = p_ijl;
				}
				else {
					shift = q_ijl;
				}
				if (oriented) {
					shift = shift % nb_dir;
				}*/

				//if (oriented) {
					//shift = angular_bin(P.u1()[n], nb_dir) % nb_dir;
				//}
				shift = (int(round((nb_dir*P.u1()[n]) / (2.*M_PI))) + nb_dir) % nb_dir;
				I = nb_frames*(nb_rings*i + k) + j;
				J = nb_frames*P.l()[n] + shift;

				patch_op.push_back(Eigen::Triplet<float>(I, J, 1.0));

				connectivity[I] = P.l()[n];
				transport[I] = shift;


				/*if (3 * I >= frame3d.size()) {
					cout << "err" << endl;
					cout << frame3d.size() << endl;
					cout << 3 * I << endl;
				}
				frame3d[3 * I + 0] = 0.0;
				frame3d[3 * I + 1] = 0.0;
				frame3d[3 * I + 2] = 0.0;*/

				if (!oriented) {
					// add contributors to the reversed frame convolution
				}
			//}
		}
	}
	// */
}

class PatchOperator {
public:
	PatchOperator(const std::string& path) {

		if (loadMesh(path, mesh)) {
			gpc = new GPC(mesh);
		} else {
			exit(666);
			// throw error
		}
	}
	PatchOperator(geodesic::Mesh& myMesh): mesh(myMesh) {
		gpc = new GPC(myMesh);
	}
	PatchOperator(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F) {
		std::vector<int> idx;
		std::vector<double> points;
		std::vector<unsigned> faces;
		convert(V, F, points, faces);
		mesh.initialize_mesh_data(points, faces);
		gpc = new GPC(mesh);
	}
	~PatchOperator() {
		delete gpc;
	}


	void compute(int ndir, int nrings, double radius, const bool oriented = true, double precision = 0.000001) {
		if (gpc == NULL) {
			exit(666);
		}
		int nv = gpc->getNbVertices();
		connectivity.resize(nv*nrings*ndir);
		transport_bin.resize(nv*nrings*ndir);
		local_frame3d.resize(nv * 3 * 3);
		base_point.resize(nv * 3);
		Patch patch;
		int percent = 0;
		for (int i = 0; i < nv; i++) {
			gpc->compute(i, radius, 0, precision);
			gpc->getPatch(patch);
			
			computeSinglePatchOperator(i, nv, ndir, nrings, patch, 
									   triplets_list,
									   connectivity,
									   transport_bin,
									   local_frame3d,
									   base_point,
									   oriented);
			
			if (nv*percent / 100. < i) {
				percent++;
				cout << percent << " percent of patch operator computed" << endl;
			}
		}
		// copy triplets into a vector
		int i = 0;
		triplets.resize(triplets_list.size());
		while (!triplets_list.empty()) {
			triplets[i] = triplets_list.front();
			triplets_list.pop_front();
			i++;
		}
		// fill the patch operator matrix:
		int nframes = ndir;
		if (!oriented) {
			nframes = 2 * ndir;
		}
		patch_op.resize(nv * nframes*nrings, nv * 2 * ndir);
		patch_op.setFromTriplets(triplets.begin(), triplets.end());

	}


	void compute_(int ndir, int nrings, double radius, const bool oriented = true, double precision = 0.000001) {
	}

	

	/*const std::list< Eigen::Triplet<float> > getTriplets() const {
		return triplets_list;
	}*/

	const Eigen::SparseMatrix<float>& getPatchOpMat() const {
		return patch_op;
	}

	bool savePatchOp(const std::string& path) const {
		return saveTriplets(path, triplets);
	}

	bool saveConnectivityTransport(const std::string& path_c, const std::string& path_t) const {
		return (saveVector(path_c, connectivity) && saveVector(path_t, transport_bin));
	}

	bool saveConnectivity(const std::string& path) const {
		return saveVector(path, connectivity);
	}

	bool saveTransport(const std::string& path) const {
		return saveVector(path, transport_bin);
	}

	bool saveLocalFrame3d(const std::string& path) const {
		return saveVector(path, local_frame3d);
	}

	bool saveBasePoint(const std::string& path) const {
		return saveVector(path, base_point);
	}

	GPC& Gpc() {
		return *gpc;
	}

protected:
	GPC* gpc;
	std::list< Eigen::Triplet<float> > triplets_list;
	std::vector< Eigen::Triplet<float> > triplets;
	Eigen::SparseMatrix<float> patch_op;
	//std::map< std::pair<int, int>, float > patch_op;
	geodesic::Mesh mesh;
	Eigen::MatrixXi bin_count;
	std::vector<int> connectivity;
	std::vector<int> transport_bin;
	//std::vector<float> transport;
	std::vector<float> bar_coords;
	std::vector<float> local_frame3d;
	std::vector<float> base_point;
};



// compute patch operators for a shape collection
// we can split the loop over mesh across different cores
/*
bool patchOpShapeCollection(const std::string& source_path, const std::string& dest_path,
							std::vector< const PatchModel* > models, std::vector< bool > composite, 
							double rad, double precision = 0.000001) {
	std::vector<std::string> meshFiles;
	PatchOperator* Pop = NULL;
	if (!getFilesList(source_path + "/", "*.off*", meshFiles)) {
		cout << "failed listing meshes" << endl;
	}
	std::string out_ext = ".patchOp";
	std::string dest;
	int progression = 0;
	int nb_models = std::min(models.size(), composite.size());
	for (int i = 0; i < meshFiles.size(); i++) {
		std::string name;
		progression = int(100.0*i / meshFiles.size());
		cout << "patch operator progession: " << progression << endl;
		Pop = new PatchOperator(source_path + "/" + meshFiles[i]);
		for (int m_id = 0; m_id < nb_models; m_id++) {
			Pop->compute(*models[m_id], composite[m_id], rad, precision);
			name = meshFiles[i].substr(0, meshFiles[i].size() - 4);
			cout << "name " << name << endl;
			if (composite[m_id]) {
				dest = dest_path + "/" + "composite/" + models[m_id]->getName();
			}
			else {
				dest = dest_path + "/" + "simple/" + models[m_id]->getName();
			}
			if (!dirExists(dest)) {
				std::string path = dest_path + "/" + "simple";
				CreateDirectory(path.c_str(), NULL);
				path = dest_path + "/" + "composite";
				CreateDirectory(path.c_str(), NULL);
				if (!CreateDirectory(dest.c_str(), NULL)) {
					cout << "unable to create directory: " << dest << endl;
				}
			}

			if (!Pop->savePatchOp(dest + "/" + name + ".spmat")) {
				cout << "failed computing patch operators !" << endl;
				return false;
			}
		}
		delete Pop;
		Pop = NULL;
	}
	cout << "patch operators computed !" << endl;
	return true;
}*/

#endif