#pragma once
#ifndef LOCAL_FRAMES_H
#define LOCAL_FRAMES_H

#include <nanoflann.hpp>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <algorithm>    // std::sort
#include "visualize.h"
#include "utils.h"
#include "kdtree3d.h"


void gradient_grid(Eigen::Matrix<double, 6, 3>& M, const Eigen::Vector3d& v, double delta) {
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			M(2 * i, j) = v(j) - delta;
			M(2 * i + 1, j) = v(j) + delta;
		}
	}
}


void hessian_grid(Eigen::Matrix<double, 27, 3>& M, const Eigen::Vector3d& v, double delta) {
	int i;
	int j;
	int k;

	for (int I = 0; I < 27; I++) {
		k = I % 3;
		j = ((I - k)/3) % 3;
		i = (I - 3 * j - k) / 9;
		M(I, 0) = v(0) + (i - 1)*delta;
		M(I, 1) = v(1) + (j - 1)*delta;
		M(I, 2) = v(2) + (k - 1)*delta;	
	}
}


void compute_grid_hessian(Eigen::Matrix3d& H, const std::vector<double>& f, double delta, bool normalize) {
	double hxx = 0.;
	double hyy = 0.;
	double hzz = 0.;
	double hxy = 0.;
	double hxz = 0.;
	double hyz = 0.;

	int i;
	int j;
	int k;

	/*std::vector<double> k2(3);
	double t = 0.5;
	k2[0] = t/2.;
	k2[1] = 1. - t;
	k2[2] = t/2.;*/

	std::vector<double> k00(3);
	k00[0] = 1.;
	k00[1] = -2.;
	k00[2] = 1.;

	bool x1 = true;
	bool y1 = true;
	bool z1 = true;

	for (int I = 0; I < 27; I++) {
		k = I % 3;
		j = ((I - k) / 3) % 3;
		i = (I - 3 * j - k) / 9;

		x1 = (i == 1);
		y1 = (j == 1);
		z1 = (k == 1);

		hxx += y1*z1*f[I] * k00[i];
		hyy += x1*z1*f[I] * k00[j];
		hzz += x1*y1*f[I] * k00[k];

		hxy += z1*(i - 1)*(j - 1)*f[I];
		hxz += y1*(i - 1)*(k - 1)*f[I];
		hyz += x1*(j - 1)*(k - 1)*f[I];
	}

	H << hxx, hxy, hxz,
		 hxy, hyy, hyz,
		 hxz, hyz, hzz;

	H /= delta;

	if (normalize) {
		Eigen::Matrix3d H2 = H.transpose()*H;
		double norm = sqrt(H2.trace());
		if (norm > +std::numeric_limits<double>::min()) {
			H /= norm;
			//cout << H << endl;
		}
	}
}

void compute_grid_gradient(Eigen::Vector3d& G, const std::vector<double>& f, double delta) {
	G(0) = f[1] - f[0];
	G(1) = f[3] - f[2];
	G(2) = f[5] - f[4];
	G /= delta;
}

void test_grid_hessian(const std::string& shape, double delta) {
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;

	if (!igl::readOFF(shape, V, F))
	{
		cout << "failed to load mesh" << endl;
		exit(666);
	}

	kdtree3d<double> kd(V);
	Eigen::Matrix<double, 27, 3> h;
	Eigen::Matrix3d H;
	Eigen::Vector3d eigs;
	Eigen::Matrix3d eigvec;
	Eigen::MatrixXd C(V.rows(), 3);
	Eigen::MatrixXd X(V.rows(), 3);
	std::vector<double> f(27);


	for (int i = 0; i < V.rows(); i++) {
		hessian_grid(h, V.row(i).transpose(), delta);
		for (int j = 0; j < 27; j++) {
			f[j] = kd.distToNN(h.row(j).transpose());
		}
		compute_grid_hessian(H, f, delta, true);
		Eigs3D(H, eigs, eigvec);
		C.row(i) = eigs.transpose();

		X.row(i) = eigvec.col(2).transpose();
		cout << eigs.transpose() << endl;
	}
	X *= 0.07;
	visualize(V, F, C, X);
}




#endif // !LOCAL_FRAMES_H
