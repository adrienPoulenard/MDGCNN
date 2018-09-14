#pragma once
#ifndef IMG2SPHERE_H
#define IMG2SPHERE_H

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include<math.h>

using namespace std;



double centerObject_(Eigen::MatrixXd& V) {
	Eigen::Vector3d v(0.0, 0.0, 0.0);
	int nv = V.rows();
	for (int i = 0; i < nv; i++) {
		v += V.row(i);
	}
	v /= nv;
	double mean_rad = 0.0;
	for (int i = 0; i < nv; i++) {
		V.row(i) -= v;
		mean_rad += V.row(i).norm();
	}
	mean_rad /= nv;
	return mean_rad;
}

/*void normalizeObject(Eigen::MatrixXd& V) {
Eigen::Vector3d v(0.0, 0.0, 0.0);
int nv = V.rows();
double r = 0.0;
for (int i = 0; i < nv; i++) {
v = V.row(i);
r += v.norm();
}
v /= nv;
for (int i = 0; i < nv; i++) {
V.row(i) /= r;
}
}*/

// we assume that the sphere is centered at the origin 
void img2sphere(Eigen::MatrixXd& f, const std::vector<uint8_t>& img, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, double ratio = 1.0, bool flip = false, int w = -1, int h = -1) {
	int nv = V.rows();
	double x;
	double y;
	double z;
	double r;
	int m;
	int n;
	if (w < 0 || h < 0) {
		w = int(sqrt(img.size()));
		h = w;
		if (w*h < img.size()) {
			cout << "wrong size " << endl;
			w++;
			h++;
		}
	}
	double R = sqrt(w*w + h*h) / (2.0*ratio);
	Eigen::Vector3d v(0.0, 0.0, 0.0);
	for (int i = 0; i < nv; i++) {
		f(i, 0) = 0.0;
		v = V.row(i).transpose();
		v.normalize();
		x = v(0);
		r = sqrt(v(1)*v(1) + v(2)*v(2));
		if (r > 0.001) {
			y = v(1) / r;
			z = v(2) / r;
		}
		else {
			y = 0.0;
			z = 0.0;
		}
		r = R*acos(x);
		y *= r;
		if (flip) {
			y = -y;
		}
		z *= r;
		// pull back value
		m = int(y + w / 2);
		n = int(z + h / 2);
		if (((n >= 0) & (n < h)) & ((m >= 0) & (m < w))) {
			f(i, 0) = double(img[m*h + n]) / 255.0; //(255.0- double(img[n*w + m]))/255.0;
		}
	}
}

void RdbToGray(const std::vector<uint8_t>& img_in, std::vector<uint8_t>& img_out) {
	for (int j = 0; j < 3; j++) {
		for (int i = 0; i < img_in.size() / 3; i++) {
			//img_out[]
		}
	}
}

void rgbImg2sphere(std::vector<uint8_t>& f, Eigen::MatrixXd& UV, const std::vector<uint8_t>& img, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, double ratio = 1.0, bool flip = false, int w = -1, int h = -1) {
	int nv = V.rows();
	double x;
	double y;
	double z;
	double r;
	int m;
	int n;
	UV.resize(nv, 2);
	if (w < 0 || h < 0) {
		w = int(sqrt(img.size() / 3));
		h = w;
		if (w*h < img.size() / 3) {
			cout << "wrong size " << endl;
			w++;
			h++;
		}
	}
	double R = sqrt(w*w + h*h) / (2.0*ratio);
	Eigen::Vector3d v(0.0, 0.0, 0.0);
	for (int j = 0; j < 3; j++) {
		for (int i = 0; i < nv; i++) {
			f[3 * i + j] = 0.0;
			v = V.row(i).transpose();
			v.normalize();
			x = v(0);
			r = sqrt(v(1)*v(1) + v(2)*v(2));
			if (r > 0.001) {
				y = v(1) / r;
				z = v(2) / r;
			}
			else {
				y = 0.0;
				z = 0.0;
			}
			r = R*acos(x);
			y *= r;
			if (flip) {
				y = -y;
			}
			z *= r;
			// pull back value
			UV(i, 0) = y + w / 2.0;
			UV(i, 1) = z + h / 2.0;
			m = int(y + w / 2);
			n = int(z + h / 2);

			if (((n >= 0) & (n < h)) & ((m >= 0) & (m < w))) {
				//f(i, j) = double(img[3*(m*h + n) + j]) / 255.0; //(255.0- double(img[n*w + m]))/255.0;
				//f[3*i+j] = img[3 * (m*h + n) + j];
				f[3 * i + j] = img[w*h*j + (m*h + n)];
			}
		}
	}
}





void rgbImg2sphere(std::vector<uint8_t>& f, const std::vector<uint8_t>& img, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, double ratio = 1.0, bool flip = false, int w = -1, int h = -1) {
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
	double R = sqrt(w*w + h*h) / (2.0*ratio);
	Eigen::Vector3d v(0.0, 0.0, 0.0);
	// int occupancy = 0;
	for (int j = 0; j < 3; j++) {
		for (int i = 0; i < nv; i++) {
			f[3 * i + j] = 0.0;
			v = V.row(i).transpose();
			v.normalize();
			x = v(0);
			r = sqrt(v(1)*v(1) + v(2)*v(2));
			if (r > 0.001) {
				y = v(1) / r;
				z = v(2) / r;
			}
			else {
				y = 0.0;
				z = 0.0;
			}
			r = R*acos(x);
			y *= r;
			if (flip) {
				y = -y;
			}
			z *= r;
			// pull back value
			m = int(y + w / 2);
			n = int(z + h / 2);

			if (((n >= 0) & (n < h)) & ((m >= 0) & (m < w))) {
				//f(i, j) = double(img[3*(m*h + n) + j]) / 255.0; //(255.0- double(img[n*w + m]))/255.0;
				//f[3*i+j] = img[3 * (m*h + n) + j];
				f[3 * i + j] = img[w*h*j + (m*h + n)];
				// occupancy++;
			}
		}
	}
	// occupancy /= 3;
	// cout << "occupancy: " << occupancy << endl;
}


#define epsilon 0.0000001

inline float sgn(float input)
{
	float output = 1.0f;
	if (input < 0.0) {
		output = -1.0f;
	}
	return output;
}

// Simple Stretching map
// mapping a circular disc to a square region
// input: (u,v) coordinates in the circle
// output: (x,y) coordinates in the square
void stretchDiscToSquare(float u, float v, float& x, float& y)
{
	if ((fabs(u) < epsilon) || (fabs(v) < epsilon)) {
		x = u;
		y = v;
		return;
	}

	float u2 = u * u;
	float v2 = v * v;
	float r = sqrt(u2 + v2);

	// a trick based on Dave Cline's idea
	// link Peter Shirley's blog
	if (u2 >= v2) {
		float sgnu = sgn(u);
		x = sgnu * r;
		y = sgnu * r * v / u;
	}
	else {
		float sgnv = sgn(v);
		x = sgnv * r * u / v;
		y = sgnv * r;
	}

}


// Simple Stretching map
// mapping a square region to a circular disc
// input: (x,y) coordinates in the square
// output: (u,v) coordinates in the circle
void stretchSquareToDisc(float x, float y, float& u, float& v)
{
	if ((fabs(x) < epsilon) || (fabs(y) < epsilon)) {
		u = x;
		v = y;
		return;
	}

	float x2 = x * x;
	float y2 = y * y;
	float hypothenusSquared = x2 + y2;

	// code can use fast reciprocal sqrt floating point trick
	// https://en.wikipedia.org/wiki/Fast_inverse_square_root
	float reciprocalHypothenus = 1.0f / sqrt(hypothenusSquared);

	float multiplier = 1.0f;
	// a trick based on Dave Cline's idea
	if (x2 > y2) {
		multiplier = sgn(x) * x * reciprocalHypothenus;
	}
	else {
		multiplier = sgn(y) * y * reciprocalHypothenus;
	}

	u = x * multiplier;
	v = y * multiplier;
}

// Elliptical Grid mapping
// mapping a circular disc to a square region
// input: (u,v) coordinates in the circle
// output: (x,y) coordinates in the square
void ellipticalDiscToSquare(double u, double v, double& x, double& y)
{
	double u2 = u * u;
	double v2 = v * v;
	double twosqrt2 = 2.0 * sqrt(2.0);
	double subtermx = 2.0 + u2 - v2;
	double subtermy = 2.0 - u2 + v2;
	double termx1 = subtermx + u * twosqrt2;
	double termx2 = subtermx - u * twosqrt2;
	double termy1 = subtermy + v * twosqrt2;
	double termy2 = subtermy - v * twosqrt2;
	x = 0.5 * sqrt(termx1) - 0.5 * sqrt(termx2);
	y = 0.5 * sqrt(termy1) - 0.5 * sqrt(termy2);

}


// Elliptical Grid mapping
// mapping a square region to a circular disc
// input: (x,y) coordinates in the square
// output: (u,v) coordinates in the circle
void ellipticalSquareToDisc(double x, double y, double& u, double& v)
{
	u = x * sqrt(1.0 - y * y / 2.0);
	v = y * sqrt(1.0 - x * x / 2.0);
}


// square images only
void shphere_UV_coordinates(const Eigen::MatrixXd& V, 
	const Eigen::MatrixXi& F, 
	Eigen::MatrixXd& UV, int w) {
	int nv = V.rows();
	UV.resize(nv, 2);
	double r;
	double sphere_r;
	Eigen::Vector3d c;
	Eigen::Vector3d z;
	Eigen::Vector3d X(1, 0, 0);
	Eigen::Vector3d Y(0, 1, 0);
	Eigen::Vector3d Z(0, 0, 1);
	c.setZero();

	double u;
	double v;

	double x;
	double y;

	for (int i = 0; i < nv; i++) {
		c += V.row(i).transpose();
	}
	c /= nv;
	for (int i = 0; i < nv; i++) {
		z = V.row(i).transpose() - c;
		z.normalize();
		sphere_r = (2.*acos(fabs(z.dot(Z)))) / M_PI;
		r = sqrt(z(0)*z(0) + z(1)*z(1));
		u = z(0);
		v = z(1);
		u *= (sphere_r / (r + epsilon));
		v *= (sphere_r / (r + epsilon));
		ellipticalDiscToSquare(u, v, x, y);
		//cout << x << " " << y << endl;
		double a = 1.0;
		x *= a;
		y *= a;
		UV(i, 0) = w * (x + 1.0) / 2.0;
		UV(i, 1) = w * (y + 1.0) / 2.0;
	}

}



void sphereImg(const std::string& dataset,
	const std::string& shape_path, const std::string& shape_name, 
	double inv_ratio=4.0, int w=32, double r_=1.8) {

	Eigen::MatrixXd UV;
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	if (!igl::readOFF(shape_path + "/" + shape_name + ".off", V, F)) {
		cout << "unable to write file " << shape_path + "/" + shape_name + ".off" << endl;
	}

	std::string UV_path = dataset + "/" + "uv_coordinates";
	
	shphere_UV_coordinates(V, F, UV, w);

	save_matrix_(UV_path + "/" + shape_name + "_w=" + std::to_string(w) + ".txt", UV);

	int nv = V.rows();

	double inv_decay_ratio = inv_ratio;
	std::vector<double> ratios(3);
	ratios[0] = 1.0;
	std::vector<int> nrings(3);
	std::vector<int> ndirs(3);
	for (int i = 0; i < 3; i++) {
		nrings[i] = 2;
		ndirs[i] = 8;
	}
	for (int i = 0; i < 2; i++) {
		ratios[i + 1] = ratios[i] / inv_decay_ratio;
	}
	for (int i = 0; i < 3; i++) {
		cout << ratios[i] << endl;
	}

	double r = r_;


	std::vector<double> radius(3);
	for (int i = 0; i < 3; i++) {
		radius[i] = r;
		r *= sqrt(inv_decay_ratio);
	}

	computeFilterMaper_(ratios, radius, nrings, ndirs, shape_path, shape_name, dataset);//*/

}

/*
void saveOffMesh(const std::string& path, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F) {
	ofstream myfile;
	myfile.open(path);
	myfile << "OFF" << endl;
	myfile << V.rows() << " " << F.rows() << " " << 0 << endl;
	for (int i = 0; i < V.rows(); i++) {
		myfile << V(i, 0) << " " << V(i, 1) << " " << V(i, 2) << endl;
	}
	for (int i = 0; i < F.rows(); i++) {
		myfile << 3 << " " << F(i, 0) << " " << F(i, 1) << " " << F(i, 2) << endl;
	}
	myfile.close();
}
void permuteMesh(const std::string& src_path, const std::string& dst_path) {
	std::vector<int> idx;
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	if (!igl::readOFF(src_path.c_str(), V, F))
	{
		cout << "failed to load mesh" << endl;
		exit(666);
	}
	int nv = V.rows();
	idx.resize(nv);
	for (int i = 0; i < nv; i++) {
		idx[i] = i;
	}
	std::random_shuffle(idx.begin(), idx.end());
	Eigen::MatrixXd V_(V.rows(), 3);
	Eigen::MatrixXi F_(F.rows(), 3);
	for (int i = 0; i < nv; i++) {
		V_.row(idx[i]) = V.row(i);
	}
	for (int j = 0; j < F.rows(); j++) {
		F_(j, 0) = idx[F(j, 0)];
		F_(j, 1) = idx[F(j, 1)];
		F_(j, 2) = idx[F(j, 2)];
	}
	saveOffMesh(dst_path, V_, F_);
}*/

#endif // !IMG2SPHERE_H