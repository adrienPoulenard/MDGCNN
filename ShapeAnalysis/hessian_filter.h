#pragma once
#ifndef HESSIAN_FILTER_H
#define HESSIAN_FILTER_H

#include <nanoflann.hpp>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <algorithm>    // std::sort
#include "visualize.h"
#include "kdtree3d.h"

#define LINE 0
#define PLATE 1
#define SPHERE 2
#define STRIP 3

void hessian3d(const Eigen::MatrixXd& V, const std::vector<int>& idx, const std::vector<double>& mu, const Eigen::Vector3d& p, double sigma, Eigen::Matrix3d& H) {

	double Mu = 0.0;
	double hxx = 0.0;
	double hyy = 0.0;
	double hzz = 0.0;

	double hxy = 0.0;
	double hxz = 0.0;
	double hyz = 0.0;

	double x;
	double y;
	double z;

	double s2 = 1./(sigma*sigma);
	double s4 = 1./(s2*s2);

	double g;
	double N = pow(1. / (sqrt(2.*M_PI*sigma*sigma)), 3.0);

	for (int i = 0; i < idx.size(); i++) {
		Mu += mu[idx[i]];
		x = V(idx[i], 0) - p(0);
		y = V(idx[i], 1) - p(1);
		z = V(idx[i], 2) - p(2);
		g = exp(-s2*(x*x+y*y+z*z)/2.);
		hxx += (s4*x*x - s2)*g;
		hyy += (s4*y*y - s2)*g;
		hyy += (s4*z*z - s2)*g;
		hxy += s4*x*y*g;
		hxz += s4*x*z*g;
		hyz += s4*y*z*g;
	}

	H(0, 0) = hxx;
	H(1, 1) = hyy;
	H(2, 2) = hzz;

	H(0, 1) = hxy;
	H(0, 2) = hxz;
	H(1, 2) = hyz;

	H(1, 0) = hxy;
	H(2, 0) = hxz;
	H(2, 1) = hyz;

	H /= Mu;
	//H *= N;
}

void hessian3d(const Eigen::MatrixXd& V, const std::vector<int>& idx, const Eigen::Vector3d& p, double sigma, Eigen::Matrix3d& H) {

	double Mu = 0.0;
	double hxx = 0.0;
	double hyy = 0.0;
	double hzz = 0.0;

	double hxy = 0.0;
	double hxz = 0.0;
	double hyz = 0.0;

	double x;
	double y;
	double z;

	double s2 = 1. / (sigma*sigma);
	double s4 = 1. / (s2*s2);

	double g;
	double N = pow(1. / (sqrt(2.*M_PI*sigma*sigma)), 3.0);

	for (int i = 0; i < idx.size(); i++) {
		Mu += 1.;
		x = V(idx[i], 0) - p(0);
		y = V(idx[i], 1) - p(1);
		z = V(idx[i], 2) - p(2);
		/*x = 0.0;
		y = 0.0;
		z = 0.0;*/
		g = exp(-s2*(x*x + y*y + z*z) / 2.);
		hxx += (s2*x*x - 1.)*g;
		hyy += (s2*y*y - 1.)*g;
		hyy += (s2*z*z - 1.)*g;
		hxy += s2*x*y*g;
		hxz += s2*x*z*g;
		hyz += s2*y*z*g;
	}

	H(0, 0) = hxx;
	H(1, 1) = hyy;
	H(2, 2) = hzz;

	H(0, 1) = hxy;
	H(0, 2) = hxz;
	H(1, 2) = hyz;

	H(1, 0) = hxy;
	H(2, 0) = hxz;
	H(2, 1) = hyz;

	
	H /= idx.size();
	//H *= N;
}


// compute a "probability" for each local pattern
inline void HessianPattern(const Eigen::Vector3d& eigs, Eigen::Vector4d& p) {

	std::vector<double> l(3);

	/*l[0] = 1.0 / (fabs(eigs(0)) + std::numeric_limits<double>::max());
	l[1] = 1.0 / (fabs(eigs(1)) + std::numeric_limits<double>::max());
	l[2] = 1.0 / (fabs(eigs(2)) + std::numeric_limits<double>::max());*/

	l[0] = fabs(eigs(0));
	l[1] = fabs(eigs(1));
	l[2] = fabs(eigs(2));

	// sort in ascending order
	std::sort(l.begin(), l.end());
	//std::swap(l[0], l[2]);
	double Ra = l[1] / l[2];
	double Rb = l[0] / sqrt(l[1] * l[2]);
	
	double mu_l = (l[1] - l[0]) / (l[0] + l[1] + l[2]);
	//double mu_l = (1.-exp(-2.*Ra*Ra))*exp(-2.*Rb*Rb);
	double mu_p = 2 * (l[1] - l[2]) / (l[0] + l[1] + l[2]);
	double mu_s = 3 * l[2] / (l[0] + l[1] + l[2]);

	//mu_l = std::min(1.0, mu_l);
	mu_p = std::min(1.0, mu_p);
	mu_s = std::min(1.0, mu_s);
	double mu_st = (1. - mu_l)*(1. - mu_p)*(1. - mu_s);

	p(LINE) = mu_l;
	p(PLATE) = mu_p;
	p(SPHERE) = mu_s;
	p(STRIP) = mu_st;
}



void computeLocalHessianPaterns(const Eigen::MatrixXd& V, double radius, Eigen::MatrixXd& p) {
	p.resize(V.rows(), 4);
	kdtree3d<double> kd(V);
	int nb_nn = 0;
	std::vector<int> idx;
	Eigen::Vector3d eigs;
	Eigen::Matrix3d H;
	Eigen::Matrix3d eigvec;
	Eigen::Vector4d p_;
	for (int i = 0; i < V.rows(); i++) {
		nb_nn = kd.radiusSearch(V.row(i).transpose(), radius, idx);
		
		hessian3d(V, idx, V.row(i).transpose(), radius/3.0, H);
		
		Eigs3D(H, eigs, eigvec);
		//cout << eigs.transpose() << endl;
		HessianPattern(eigs, p_);
		p.row(i) = p_.transpose();
	}
}


void gaussian_test(const std::string& path, int v_id, double radius) {
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	if (!igl::readOFF(path, V, F))
	{
		cout << "failed to load mesh" << endl;
		exit(666);
	}
	std::vector<double> f(V.rows());
	double x;
	double y;
	double z;
	double R2 = radius*radius;
	double r2;
	double sigma = radius / 3.0;
	double s2 = 1. / (sigma*sigma);
	for (int i = 0; i < V.rows(); i++) {
		x = V(i, 0) - V(v_id, 0);
		y = V(i, 1) - V(v_id, 1);
		z = V(i, 2) - V(v_id, 2);
		r2 = x*x + y*y + z*z;
		if (r2 < R2) {
			f[i] = exp(-s2*(r2) / 2.);
			//f[i] = (s2*x*z)*exp(-s2*(r2) / 2.);
		}
		else {
			f[i] = 0.0;
		}
	}
	visualize(V, F, f);
}


void test_local_hessian_paterns(const std::string& path, int patern, double radius) {
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	if (!igl::readOFF(path, V, F))
	{
		cout << "failed to load mesh" << endl;
		exit(666);
	}
	std::vector<double> f(V.rows());
	Eigen::MatrixXd p;
	computeLocalHessianPaterns(V, radius/3.0, p);
	for (int i = 0; i < V.rows(); i++) {
		f[i] = p(i, patern);
		cout << f[i] << endl;
	}
	visualize(V, F, f);
}

#endif // !HESSIAN_FILTER_H

