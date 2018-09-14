#pragma once
#ifndef RELATIVE_ORIENTATION_H
#define RELATIVE_ORIENTATION_H

#include <iostream>
#include <fstream>

#include "permute_dataset.h"

#include <Eigen/Dense>
#include "create_dataset.h"
#include "utils.h"
#include <stdlib.h>     /* srand, rand */
#include <math.h>
#include "visualize.h"
#include "noise.h"

void draw_orthogonal(const Eigen::MatrixXd& V, std::vector<double>& f) {
	double w = M_PI / 16.;
	double h = M_PI / 5.;
	for (int i = 0; i < V.rows(); i++) {
		f[i] = 0.;
		if ((fabs(V(i, 0)) < sin(w)) && (fabs(V(i, 1)) < sin(h)) && (V(i,2) < 0.0)) {
			f[i] = 1.;
		}
		if ((fabs(V(i, 1)) < sin(w)) && (fabs(V(i, 0)) < sin(h)) && (V(i, 2) > 0.0)) {
			f[i] = 1.;
		}
	}
}

void draw_aligned(const Eigen::MatrixXd& V, std::vector<double>& f) {
	double w = M_PI / 16;
	double h = M_PI / 5.;
	for (int i = 0; i < V.rows(); i++) {
		f[i] = 0.;
		if ((fabs(V(i, 1)) < sin(w)) && (fabs(V(i, 0)) < sin(h))) {
			f[i] = 1.;
		}
	}
}



void draw_lines(bool north_pole, double lateral_angle, double angle, 
	const std::vector<double>& h, 
	const std::vector<double>& w, 
	const std::vector<double>& space,
	const Eigen::MatrixXd& V, std::vector<double>& f) {
	int nv = V.rows();
	if (f.size() != nv) {
		f.resize(nv);
	}
	
	double sgn = 1.0;
	if (north_pole) {
		sgn = -1.0;
	}
	int nb_lines = h.size();
	double c = cos(angle);
	double s = sin(angle);
	double c_l = cos(lateral_angle);
	double s_l = sin(lateral_angle);
	double X;
	double Y;
	double Z;
	double x;
	double y;
	double x_min;
	double y_min;
	bool cond_x;
	bool cond_y;
	//double H = 0.0;
	double W = 0.0;

	for (int j = 0; j < nb_lines; j++) {
		W += w[j] + space[j];
	}

	for (int i = 0; i < nv; i++) {
		//f[i] = 0.0;
		X = V(i, 0);
		Y = c_l*V(i, 1) - s_l*V(i, 2);
		Z = s_l*V(i, 1) + c_l*V(i, 2);

		x = c*X -s*Y;
		y = s*X + c*Y;
		y_min = -W / 2.;
		for (int j = 0; j < nb_lines; j++) {
			x_min = -h[j] / 2.0;
			cond_x = (x < x_min + h[j]) && (x > x_min);
			cond_y = (y < y_min + w[j]) && (y > y_min);
			if (cond_x&&cond_y&&(sgn*V(i, 2) < 0.0)) {
				f[i] = 1.0;
			}
			y_min += w[j] + space[j];
		}
	}
}

int generate_random_patern(bool north_pole, double lateral_angle, double angle, const Eigen::MatrixXd& V, std::vector<double>& f) {
	int nb_lines = 1;//rand() % 2 + 1;
	std::vector<double> w(nb_lines);
	std::vector<double> h(nb_lines);
	std::vector<double> d(nb_lines);
	double t = 0.4;
	double W = 1.2*(1+t)*1*M_PI / 16;
	double H = 1.2*(1+t)*3.*M_PI / 16;
	for (int i = 0; i < nb_lines; i++) {
		w[i] = ((1. - t) + t*random_real())*(W);
		h[i] = ((1. - t) + t*random_real())*(H);
		d[i] = ((1. - t) + t*random_real())*(W);
	}
	d[nb_lines - 1] = 0.0;

	draw_lines(north_pole, lateral_angle, angle, h, w, d, V, f);
	return nb_lines;
}

int draw_random_lines(const Eigen::MatrixXd& V, std::vector<double>& f, double lateral_angle) {
	int a = rand() % 3;
	double angle = a*M_PI / 4.;
	if (f.size() != V.rows()) {
		f.resize(V.rows());
	}
	for (int k = 0; k < V.rows(); k++) {
		f[k] = 0.0;
	}
	int i = generate_random_patern(true, lateral_angle, 0.0, V, f);
	int j = generate_random_patern(false, 0.0, angle, V, f);
	//return 3*(2*i + j) + a;
	return a;
}

void local_patterns_(const std::string& spheres_path,
	const std::string& signals_path,
	const std::string& labels_path,
	double lateral_angle) {
	std::vector< std::string > names;
	getFilesList_(spheres_path, ".off", names);

	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	if (!igl::readOFF(spheres_path + "/" + names[0] + ".off", V, F))
	{
		cout << "failed to load mesh" << endl;
		exit(666);
	}

	std::vector<double> f(V.rows());
	std::vector<int> l(1);


	for (int i = 0; i < names.size(); i++) {
		if (!igl::readOFF(spheres_path + "/" + names[i] + ".off", V, F))
		{
			cout << "failed to load mesh" << endl;
			exit(666);
		}

		l[0] = draw_random_lines(V, f, lateral_angle);

		//local_average(F, f, 3);
		//salt_noise(1./20, f);
		perturb_bnd(1. / 10, F, f);
		//local_average(F, f, 1);
		//snow_noise(0.1, f);

		save_vector_(signals_path + "/" + names[i] + ".txt", f);
		save_vector_(labels_path + "/" + names[i] + ".txt", l);
	}
}


void local_patterns(const std::string& spheres_path, 
	const std::string& signals_path,
	const std::string& labels_path) {
	std::vector< std::string > names;
	getFilesList_(spheres_path, ".off", names);

	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	if (!igl::readOFF(spheres_path + "/" + names[0] + ".off", V, F))
	{
		cout << "failed to load mesh" << endl;
		exit(666);
	}
	
	std::vector<double> f(V.rows());
	std::vector<int> l(1);


	for (int i = 0; i < names.size(); i++) {
		if (!igl::readOFF(spheres_path + "/" + names[i] + ".off", V, F))
		{
			cout << "failed to load mesh" << endl;
			exit(666);
		}
		if (i % 2 == 0) {
			l[0] = 0;
			draw_aligned(V, f);
		}
		else {
			l[0] = 1;
			draw_orthogonal(V, f);
		}
		save_vector_(signals_path + "/" + names[i] + ".txt", f);
		save_vector_(labels_path, f);
	}
}

void visualize_orth(const std::string& sphere, bool orth) {
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	if (!igl::readOFF(sphere, V, F))
	{
		cout << "failed to load mesh" << endl;
		exit(666);
	}

	std::vector<double> f(V.rows());

	if (orth) {
		draw_orthogonal(V, f);
	}
	else {
		draw_aligned(V, f);
	}

	visualize(V, F, f);
}

int visualize_lines(const std::string& sphere, double lateral_angle) {
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	if (!igl::readOFF(sphere, V, F))
	{
		cout << "failed to load mesh" << endl;
		exit(666);
	}

	std::vector<double> f(V.rows());

	int res = draw_random_lines(V, f, lateral_angle);
	//local_average(F, f, 3);
	//salt_noise(1./20., f);
	perturb_bnd(1. / 10, F, f);
	local_average(F, f, 1);
	snow_noise(0.1, f);
	cout << res << endl;
	visualize(V, F, f);

	return res;
}


#endif // !RELATIVE_ORIENTATION_H
