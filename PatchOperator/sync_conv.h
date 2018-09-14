#pragma once
#ifndef SYNC_CONV_H
#define SYNC_CONV_H

#include "patch_op.h"
#include <Eigen/Sparse>
#include <igl/viewer/Viewer.h>
#include <igl/parula.h>

#include <igl/readOFF.h>

using namespace std;

// Synchronize F along patch of center pt in the frame-th frame
template< typename T>
void Synchronize(int pt, int frame, GPC& gpc, const Eigen::Matrix<T,-1,-1>& F, Eigen::Matrix<T, -1, -1>& f,
	double radius = std::numeric_limits<double>::max()) {
	int ndir = F.cols()/2;
	int nv = F.rows();
	if (f.rows() != nv || f.cols() != 1) {
		f.resize(nv, 1);
	}
	f.setZero();
	gpc.compute(pt, radius);
	Patch patch;
	gpc.getPatch(patch);
	int nv_patch = patch.l().size();
	double a = 0.;
	int p = 0;
	int q = 0;
	int j = 0;
	for (int i = 0; i < nv_patch; i++) {
		angular_bin(a, p, q, patch, frame, i, ndir);
		j = patch.l()[i];
		f(j, 0) = a*F(j,q) + (1. - a)*F(j,p);
	}
}

// utils 
void rotMat(double theta, Eigen::Matrix2d& rot) {
	double c = cos(theta);
	double s = sin(theta);
	rot(0, 0) = c;
	rot(1, 1) = c;
	rot(0, 1) = -s;
	rot(1, 0) = s;
}


// synchronized convolution 

template< typename T>
void syncConv(GPC& gpc, const Eigen::Matrix<T, -1, -1>& F, double (*g)(double, double), Eigen::Matrix<T, -1, -1>& C, const std::vector<unsigned int>& compute_at,
	double radius = std::numeric_limits<T>::max()) {
	int ndir = F.cols() / 2;
	int nv = F.rows();
	Patch P;
	//Eigen::Matrix2d tau;
	double a = 0.;
	int p = 0;
	int q = 0;
	double theta;
	double r;

	//if (C.rows() != nv || C.cols() != F.cols()) {
		C.resize(nv, F.cols());
	//}
	C.setZero();


	int percent = 0;
	int nv_to_compute = compute_at.size();

	for (int i = 0; i < nv; i++) {
	//for (int m = 0; m < nv_to_compute; m++) {
		/*if ((percent < int(100 * m / nv_to_compute))) {
			percent++;
			cout << "sync conv progress: " << percent << " percent" << endl;
		}*/
		if ((percent < int(100 * i / nv))) {
			percent++;
			cout << "sync conv progress: " << percent << " percent" << endl;
		}
		// compute patch
		//int i = compute_at[m];
		gpc.compute(i, radius);
		gpc.getPatch(P);
		int j = 0;
		int nv_patch = P.l().size();
		for (int k = 0; k < ndir; k++) {
			for (int n = 0; n < nv_patch; n++) {
				j = P.l()[n];
				angular_bin(a, p, q, P, k, n, ndir);
				theta = P.theta()[n] - (2.*k*M_PI) / ndir;
				r = P.r()[n];
				C(i, k) += (a*F(j, q) + (1. - a)*F(j, p))*g(theta, r)*P.mu()[n];
			}
		}
		for (int k = 0; k < ndir; k++) {
			for (int n = 0; n < nv_patch; n++) {
				j = P.l()[n];
				angular_bin(a, p, q, P, k + ndir, n, ndir);
				theta = -P.theta()[n] - (2.*k*M_PI) / ndir;
				r = P.r()[n];
				C(i, k+ndir) += (a*F(j, q) + (1. - a)*F(j, p))*g(theta, r)*P.mu()[n];
			}
		}
	}
}

template< typename T >
void frameBundlePullBack(const Eigen::Matrix<T, -1, -1>& f, Eigen::Matrix<T, -1, -1>& F, int nframes = -1) {
	int nv = f.rows();
	if (nframes < 1) {
		nframes = F.cols();
	}
	if (F.rows() != nv || F.cols() != nframes) {
		F.resize(nv, nframes);
	}
	for (int i = 0; i < nv; i++) {
		F.row(i).setConstant(f(i, 0));
	}
}


double flatGaussian(double theta, double r) {
	double x = r*cos(theta);
	double y = r*sin(theta);
	//double sigma = M_PI/10.;
	double sigma = M_PI / 5.;
	double ratio = 5.;
	return exp(-(x*x + ratio*ratio*y*y) / sigma);
}

double gaussianFrame(double theta, double r) {
	double x = r*cos(theta);
	double y = r*sin(theta);
	//double sigma = M_PI/10.;
	double sigma = M_PI / 5.;
	double ratio = 5.;
	return x*exp(-(x*x + ratio*ratio*y*y) / sigma) + y*exp(-(ratio*ratio*x*x + y*y) / sigma);
}

template< typename T>
void testConvolution(int bp, int ndir, double radius, GPC& gpc,  Eigen::Matrix<T, -1, -1>& C) {
	int nv = gpc.getNbVertices();
	int nframes = ndir * 2;
	Patch patch;
	gpc.compute(bp, M_PI / 2.0);
	gpc.getPatch(patch);
	double val = 1000.0;
	Eigen::MatrixXd dirac(nv, 1);
	dirac.setZero();
	dirac(bp, 0) = val;
	Eigen::MatrixXd D;
	frameBundlePullBack(dirac, D, nframes);
	
	syncConv(gpc, D, &gaussianFrame, C, patch.l(), radius);
}

bool unitTest(int bp, int ndir, int frame, double radius) {

	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	if (!igl::readOFF("C:/Users/Adrien/Documents/shapes/sphere/sphere_1002.off", V, F))
	{
		cout << "failed to load mesh" << endl;
		return false;
	}


	int nv = V.rows();


	Eigen::MatrixXd C;
	std::vector<double> points;
	std::vector<unsigned> facets;
	convert(V, F, points, facets);
	geodesic::Mesh mesh;
	mesh.initialize_mesh_data(points, facets);
	GPC gpc(mesh);
	Eigen::MatrixXd f(nv, 1);
	f.setZero();
	testConvolution(bp, ndir, radius, gpc, C);
	
	Synchronize(bp, frame, gpc, C, f);

	gpc.compute(bp, 100.0);
	/*
	for (int i = 0; i < nv; i++) {
		f(i, 0) = gpc.getAngle()[i];
	}*/


	
	int k = 0;
	Eigen::MatrixXd angle(gpc.getDist().size(), 1);
	Eigen::MatrixXd U1(gpc.getDist().size(), 3);
	U1.setZero();
	Eigen::MatrixXd U2(gpc.getDist().size(), 3);
	U2.setZero();
	Eigen::MatrixXd BaseDir(gpc.getDist().size(), 3);
	BaseDir.setZero();

	for (int i = 0; i < gpc.nbValid(); i++) {
		Eigen::Vector3d u;
		Eigen::Vector3d v;
		//Eigen::Vector3d w;
		int l = gpc.getValidIdx()[i];

		int u_id = gpc.getOneRingVertices(l)[0];
		int v_id = gpc.getOneRingVertices(l)[1];


		u = gpc.getVertexCoord(u_id);
		u -= gpc.getVertexCoord(l);
		u.normalize();

		v = gpc.getVertexCoord(v_id);
		v -= gpc.getVertexCoord(l);
		v.normalize();
		v -= v.dot(u)*u;
		v.normalize();
		double sz = 0.07;
		U1.row(l) = sz*(cos(gpc.getU1()[l])*u + sin(gpc.getU1()[l])*v);
		U2.row(l) = sz*(cos(gpc.getU2()[l])*u + sin(gpc.getU2()[l])*v);
		BaseDir.row(l) = sz*u;

		//U1.row(l) = gecAlgorithm.pTransport1(i);
	}
	for (int i = 0; i < gpc.getDist().size(); i++) {
		if (gpc.getState()[i]) {

			angle(i, 0) = gpc.getAngle()[i];

			//C(i, 0) = 1.0;
		}
		else {
			angle(i, 0) = 0.0;
		}
	}
	
	igl::viewer::Viewer viewer;
	viewer.callback_key_down = [&](igl::viewer::Viewer & viewer, unsigned char key, int)->bool
	{
		switch (key)
		{
		default:
			return false;
		case ' ':
		{
			viewer.data.set_mesh(V, F);
			viewer.data.compute_normals();
			viewer.data.set_colors(f);
			//viewer.data.set_colors(angle);
			viewer.data.add_edges(V, V + U1, Eigen::RowVector3d(0, 0, 0));
			//viewer.data.add_edges(V, V + BaseDir, Eigen::RowVector3d(255, 255, 255));*/
			return true;
		}
		}
	};
	viewer.callback_key_down(viewer, ' ', 0);
	viewer.core.show_lines = false;
	viewer.launch(); 
	return true;
}



#endif // !VISUALIZE_H
