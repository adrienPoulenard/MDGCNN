#pragma once
#ifndef POLYGONS_ON_SPHERES_H
#define POLYGONS_ON_SPHERES_H

#include <iostream>
#include <fstream>

#include "permute_dataset.h"

#include <Eigen/Dense>
#include "create_dataset.h"
#include "utils.h"
#include <stdlib.h>     /* srand, rand */
#include <math.h>
#include "visualize.h"

#define MAX_BRANCHES 10
#define MIN_BRANCHES 3
#define N_REP 2



int get_south_pole(const Eigen::Vector3d& v, const Eigen::MatrixXd& V) {
	int argmin = 0;
	double min_ = std::numeric_limits<double>::max();
	double x = 0.0;
	for (int i = 0; i < V.rows(); i++) {
		x = v.dot(V.row(i).transpose());
		if (min_ > x) {
			min_ = x;
			argmin = i;
		}
	}
	return argmin;
}

int get_north_pole(const Eigen::Vector3d& v, const Eigen::MatrixXd& V) {
	Eigen::Vector3d w = -v;
	return get_south_pole(w, V);
}

double angle(double x, double y) {
	if (y != 0. || x != 0.) {
		M_PI + atan2(y, x);
	}
	else {
		return 0;
	}
}


void barycentric_coords(
	const Eigen::Vector2d& a,
	const Eigen::Vector2d& b,
	const Eigen::Vector2d& c,
	const Eigen::Vector2d& x,
	Eigen::Vector3d& l) {
	Eigen::Matrix2d T;
	//const Eigen::Vector2d& a,
	//const Eigen::Vector2d& b,
	T.col(0) = a - c;
	T.col(1) = b - c;
	Eigen::Vector2d l_;
	l_ = T.inverse()*(x - c);
	l(0) = l_(0);
	l(1) = l_(1);
	l(2) = 1. - l_(0) - l_(1);
}

bool is_in_triangle(const Eigen::Vector2d& x,
	const Eigen::Vector2d& a,
	const Eigen::Vector2d& b,
	const Eigen::Vector2d& c) {

	Eigen::Vector3d l;
	barycentric_coords(a, b, c, x, l);
	return ((l(0) >= 0.) && (l(1) >= 0.) && (l(2) >= 0.));
}



void polygon_to_sphere_(bool north,
	const std::vector<double>& r,
	const Eigen::MatrixXd& V,
	std::vector<int>& labels,
	double offset = 0.) {


	Eigen::Vector3d X;
	Eigen::Vector3d Y;
	Eigen::Vector3d Z;

	X << cos(offset), sin(offset), 0.;
	Y << cos(offset + M_PI/2.), sin(offset + M_PI / 2.), 0.;
	Z << 0., 0., 1.;
	if (north) {
		Z << 0., 0., -1.;
	}
	int bp = 0;
	

	bp = get_south_pole(Z, V);


	Eigen::Vector3d pt3d;
	Eigen::Vector2d pt2d;

	//X = cos(offset)*u + sin(offset)*v;
	//Y = cos(offset+M_PI/2.)*u + sin(offset+M_PI/2.)*v;

	double x = 0.;
	double y = 0.;
	double z = 0.;
	double t = 0.;

	Eigen::Vector2d a;
	Eigen::Vector2d b;
	Eigen::Vector2d c;
	c.setZero();
	int nv = V.rows();
	int npv = r.size();
	if (labels.size() != nv) {
		labels.resize(nv);
	}
	int k = 0;
	double dt = 2.*M_PI / npv;
	
	for (int i = 0; i < nv; i++) {
		//labels[i] = 0;
		pt3d = V.row(i).transpose();
		pt2d(0) = X.dot(pt3d);
		pt2d(1) = Y.dot(pt3d);

		if (pt2d.norm() > 0.0) {
			double n_ = pt2d.norm();
			pt2d.normalize();
			pt2d *= asin(n_);

		}


		for (int j = 0; j < npv; j++) {
			a(0) = cos(2.*j*M_PI / npv);
			a(1) = sin(2.*j*M_PI / npv);
			a *= r[j];
			b(0) = cos(2.*(j + 1)*M_PI / npv);
			b(1) = sin(2.*(j + 1)*M_PI / npv);
			b *= r[(j + 1) % npv];
			if (is_in_triangle(pt2d, a, b, c) && (Z.dot(V.row(i).transpose()) >=0)) {
				labels[i] = npv/2 - MIN_BRANCHES + 1;
			}
		}
	}

}

void generate_random_polygon(int nv, double radius, std::vector<double>& r) {
	r.resize(nv);
	//cout << "n_branches " << nv << endl;
	double t = 0.7;//0.55
	double t2 = 0.7;
	for (int i = 0; i < nv; i++) {
		r[i] = radius*((random_real()*t2 + (1.-t2))*(i % 2)*t + (1.-t));
		//r[i] = radius*((i%2)*t + (1.-t));
	}
	// normalize
	double max_ = -std::numeric_limits<double>::max();
	for (int i = 0; i < nv; i++) {
		max_ = std::max(max_, r[i]);
	}
	
	for (int i = 0; i < nv; i++) {
		r[i] = radius*r[i]/max_;
	}
}

void rand_frame(Eigen::Vector3d& u, Eigen::Vector3d& v, Eigen::Vector3d& w) {
	w(0) = random_real();
	w(1) = random_real();
	w(2) = random_real() + std::numeric_limits<double>::min();

	w.normalize();

	u(0) = random_real() + std::numeric_limits<double>::min();
	u(1) = random_real();
	u(2) = random_real();
	u.normalize();

	while (w.cross(u).norm() < 0.01) {
		u(0) = random_real() + std::numeric_limits<double>::min();
		u(1) = random_real();
		u(2) = random_real();
		u.normalize();
	}

	u - u.dot(w)*w;
	u.normalize();
	v = w.cross(u);

}

// star
void rand_polygon_to_sphere(bool north, const Eigen::MatrixXd& V,
	std::vector<int>& labels) {

	// generate random polygon with npv vertices
	int nv = V.rows();
	int npv = 2*std::max((std::rand() % MAX_BRANCHES), MIN_BRANCHES);
	std::vector<double> r;
	generate_random_polygon(npv, 1.3*M_PI/2.0, r);
	double offset = 2.*M_PI*random_real();
	std::vector<bool> f(nv);
	polygon_to_sphere_(north,
		r,
		V,
		labels,
		offset);
}



void generate_permuted_spheres(int n, const std::string& sphere, 
	const std::string& sphere_directory, const std::string& permuted_spheres) {
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	Eigen::MatrixXd V_;
	Eigen::MatrixXi F_;

	if (!igl::readOFF(sphere_directory + "/" + sphere + ".off", V, F))
	{
		cout << "failed to load mesh" << endl;
		exit(666);
	}
	
	int nv = V.rows();
	std::vector<int> permutation(nv);
	for (int i = 0; i < nv; i++) {
		permutation[i] = i;
	}

	std::string perm_sphere;
	for (int i = 0; i < n; i++) {
		std::random_shuffle(permutation.begin(), permutation.end());
		permuteMesh(permutation,
			V, F,
			V_, F_);
		perm_sphere = permuted_spheres + "/" + sphere + "_" + std::to_string(i) + ".off";
		igl::writeOFF(perm_sphere, V_, F_);
	}
}


void rand_2_polygons_to_sphere(const Eigen::MatrixXd& V, std::vector<int>& labels) {
	if (labels.size() != V.rows()) {
		labels.resize(V.rows());
	}
	std::fill(labels.begin(), labels.end(), 0);

	rand_polygon_to_sphere(false, V, labels);
	
	rand_polygon_to_sphere(true, V, labels);
}


void sphere_polygons_labels(const std::string& spheres, const std::string& dataset_path) {
	// list spheres directory
	std::vector< std::string > names;
	getFilesList_(spheres, ".off", names);
	
	// 

	std::vector<int> labels;
	std::vector<float> f;
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	for (int i = 0; i < names.size(); i++) {
		
		if (!igl::readOFF(spheres + "/" + names[i] + ".off", V, F))
		{
			cout << "failed to load mesh" << endl;
			exit(666);
		}
		if (f.size() != V.rows()) {
			f.resize(V.rows());
		}
		rand_2_polygons_to_sphere(V, labels);
		for (int j = 0; j < V.rows(); j++) {
			if (labels[j] > 0) {
				f[j] = 1.0;
			}
			else {
				f[j] = 0.0;
			}
		}
		save_vector_<float>(dataset_path + "/" + "signals" + "/" + names[i] + ".txt", f);
		save_vector_<int>(dataset_path + "/" + "labels" + "/" + names[i] + ".txt", labels);
	}
}

void rand_polygon_test(const std::string& sphere) {
	std::vector<int> labels;
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	if (!igl::readOFF(sphere, V, F))
	{
		cout << "failed to load mesh" << endl;
		exit(666);
	}
	int n = N_REP;
	for (int i = 0; i < n; i++) {
		rand_2_polygons_to_sphere(V, labels);
	}
	
	visualize(V, F, labels);
}

void train_test_txt(const std::string& dataset_path, int ntrain, int ntest) {
	// list label files
	std::vector< std::string > names;
	getFilesList_(dataset_path + "/" + "labels", ".txt", names);
	ofstream myfile;
	myfile.open(dataset_path + "/train.txt");
	for (int i = 0; i < ntrain; i++) {
		myfile << names[i] << "\n";
	}
	myfile.close();

	myfile.open(dataset_path + "/test.txt");
	for (int i = 0; i < ntest; i++) {
		myfile << names[i+ntrain] << "\n";
	}
	myfile.close();
}
#endif // !POLYGONS_ON_SPHERES_H
