#pragma once
#ifndef IMG_2_CYLINDER_H
#define IMG_2_CYLINDER_H

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <igl/decimate.h>
#include "visualize.h"
#include "utils.h"
#include<igl/writeOFF.h>

using namespace std;

void sphere2cylinder(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
	Eigen::MatrixXd& U) {
	if ((U.rows() != V.rows()) || (U.cols() != V.cols())) {
		U.resize(V.rows(), V.cols());
	}
	Eigen::Vector3d c;
	c.setZero();
	Eigen::Vector3d v;
	Eigen::Vector3d z;
	z(0) = 0.;
	z(1) = 0.;
	z(2) = 1.;
	double b = 1.0;
	double s = 0.;
	double a = (2.*M_PI) / (2.*M_PI + 1);
	for (int i = 0; i < V.rows(); i++) {
		c += V.row(i).transpose();
	}
	c /= V.rows();
	for (int i = 0; i < V.rows(); i++) {
		v = V.row(i).transpose() - c;
		v.normalize();
		s = sqrt(1. - a*a);
		if (fabs(v(2)) > a) {
			// spherical cap
			v(0) /= 2.*s*M_PI;
			v(1) /= 2.*s*M_PI;
			if (v(2) > 0.) {
				v(2) = 0.5;//sqrt(1. - a*a);
			}
			else {
				v(2) = -0.5;//-sqrt(1. - a*a);
			}
			U.row(i) = v.transpose();
		}
		else {
			// cylinder
			b = sqrt(v(0)*v(0) + v(1)*v(1));
			v(0) /= b;
			v(1) /= b;
			v(0) /= 2.*M_PI;
			v(1) /= 2.*M_PI;
			v(2) *= 0.5 / a;
			U.row(i) = v.transpose();
		}
	}
}


void sphere2cylinder(const std::string& path, Eigen::MatrixXd& U) {
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	if (!igl::readOFF(path, V, F))
	{
		//cout << "failed to load mesh" << endl;
		exit(666);
	}
	sphere2cylinder(V, F, U);
}

void sphere2cylinder(const std::string& path) {
	Eigen::MatrixXd U;
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	if (!igl::readOFF(path, V, F))
	{
		//cout << "failed to load mesh" << endl;
		exit(666);
	}
	sphere2cylinder(V, F, U);
	visualize(U, F);
}

void reducedCylinders(const std::string& sphere, const std::string& dst, const std::vector<double>& ratios) {
	Eigen::MatrixXd Vc;
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	if (!igl::readOFF(sphere, V, F))
	{
		//cout << "failed to load mesh" << endl;
		exit(666);
	}

	

	Eigen::MatrixXd U;
	Eigen::MatrixXi G;
	Eigen::VectorXi J;
	Eigen::VectorXi I;

	// sphere2cylinder(V, F, Vc);


	for (int i = 0; i < ratios.size(); i++) {
		igl::decimate(V, F, (size_t)(F.rows()*ratios[i]), U, G, J, I);
		sphere2cylinder(U, G, Vc);
		if (!igl::writeOFF(dst + "/" + "cylinder_" + std::to_string(Vc.rows()) + ".off", Vc, G))
		{
			//cout << "failed to save mesh" << endl;
			exit(666);
		}
	}
}

void reducedCylinders(const std::string& sphere, const std::string& dst, int n) {
	std::vector<double> ratios(n);
	ratios[0] = 1.0;
	for (int i = 0; i < n-1; i++) {
		ratios[i + 1] = ratios[i] / 2.;
	}
	reducedCylinders(sphere, dst, ratios);
}



void mapImage2cylinder(int w, int h, 
	const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
	Eigen::MatrixXd& UV,
	double margin = 0.025) {
	int nv = V.rows();
	double x;
	double y;
	double z;
	double r;
	int m;
	int n;
	UV.resize(nv, 2);
	
	Eigen::Vector3d v(0.0, 0.0, 0.0);
	for (int i = 0; i < nv; i++) {
		v = V.row(i).transpose();
		x = (w*angle_(v(0), v(1)) / (2.*M_PI))/(1. - 2 * margin);
		y = h*(v(2) + 0.5)/ (1. - 2 * margin);
		UV(i, 0) = x;
		UV(i, 1) = y;
	}
}




void rgbImg2cylinder(std::vector<uint8_t>& f, const std::vector<uint8_t>& img, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, double margin = 0.025, bool flip = false, int w = -1, int h = -1) {
	int nv = V.rows();
	double x;
	double y;
	double z;
	double r;
	int m;
	int n;

	if (w < 0 || h < 0) {
		w = int(sqrt(img.size() / 3));
		h = w;
		if (w*h < img.size() / 3) {
			cout << "wrong size " << endl;
			w++;
			h++;
		}
	}
	int occupancy = 0;
	std::fill(f.begin(), f.end(), 0);
	Eigen::Vector3d v(0.0, 0.0, 0.0);
	for (int i = 0; i < nv; i++) {
		v = V.row(i).transpose();
		if (fabs(v(2)) < 0.5 - margin) {
			
			x = angle_(v(0), v(1)) / (2.*M_PI);
			y = v(2) + 0.5;
			occupancy++;
			if ((x < 1.0 - 2.*margin) && (y < 1.0 - 2.*margin)) {
				m = int(std::round((1.01 * w * x / (1. - 2 * margin))));
				n = int(std::round((1.01 * h * y / (1. - 2 * margin))));
				if (((n >= 0) & (n < h)) & ((m >= 0) & (m < w))) {
					//f(i, j) = double(img[3*(m*h + n) + j]) / 255.0; //(255.0- double(img[n*w + m]))/255.0;
					//f[3*i+j] = img[3 * (m*h + n) + j];
					for (int j = 0; j < 3; j++) {
						f[3 * i + j] = img[w*h*j + (m*h + n)];
					}
					
				}
			}
		}
	}
	//cout << "occupancy: " << occupancy << endl;
}






#endif