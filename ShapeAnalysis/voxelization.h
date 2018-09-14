#pragma once
#ifndef VOXELIZATION_H
#define VOXELIZATION_H

#include <nanoflann.hpp>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <algorithm>    // std::sort
#include "visualize.h"
#include "utils.h"
#include "kdtree3d.h"


template <typename num_t>
class Tensor3D {
public:
	Tensor3D(){
		n_x = 0;
		n_y = 0;
		n_z = 0;
		n_x_ = 0;
		n_y_ = 0;
		n_z_ = 0;
		margin = 0;
		offset = 0;
	}
	Tensor3D(int nx, int ny, int nz) {
		n_x = nx;
		n_y = ny;
		n_z = nz;
		n_x_ = nx;
		n_y_ = ny;
		n_z_ = nz;
		margin = 0;
		offset = 0;
		T.resize(nx*ny*nz);
		std::fill(T.begin(), T.end(), 0);
	}
	Tensor3D(const Eigen::Vector3i& shape) {
		n_x = shape(0);
		n_y = shape(1);
		n_z = shape(2);
		n_x_ = n_x;
		n_y_ = n_y;
		n_z_ = n_z;
		margin = 0;
		offset = 0;
		T.resize(n_x*n_y*n_z);
		std::fill(T.begin(), T.end(), 0);
	}
	Tensor3D(const std::vector<num_t>& v, int nx, int ny, int nz) {
		n_x = nx;
		n_y = ny;
		n_z = nz;
		n_x_ = nx;
		n_y_ = ny;
		n_z_ = nz;
		margin = 0;
		offset = 0;
		T.resize(nx*ny*nz);
		if (v.size() == T.size()) {
			std::copy(v.begin(), v.end(), T.begin());
		}
		else {
			cout << "invalid input size" << endl;
		}
	}
	Tensor3D(const std::vector<num_t>& v, const Eigen::Vector3i& shape) {
		n_x = shape(0);
		n_y = shape(1);
		n_z = shape(2);
		n_x_ = nx;
		n_y_ = ny;
		n_z_ = nz;
		margin = 0;
		offset = 0;
		T.resize(nx*ny*nz);
		if (v.size() == T.size()) {
			std::copy(v.begin(), v.end(), T.begin());
		}
		else {
			cout << "invalid input size" << endl;
		}
	}
	~Tensor3D() {

	}
	

	Eigen::Vector3i getShape() const {
		Eigen::Vector3i shape;
		shape(0) = n_x;
		shape(1) = n_y;
		shape(2) = n_z;
		return shape;
	}
	void reshape(int nx, int ny, int nz) {
		n_x = nx;
		n_y = ny;
		n_z = nz;
		cout << "margin " << margin << endl;
		
		n_x_ = n_x + 2*margin;
		n_y_ = n_y + 2*margin;
		n_z_ = n_z + 2*margin;
		offset = n_z_*(n_y_*margin + margin) + margin;
		T.resize(n_x_*n_y_*n_z_);
		std::fill(T.begin(), T.end(), 0.);
	}
	void reshape(const Eigen::Vector3i& shape) {
		n_x = shape(0);
		n_y = shape(1);
		n_z = shape(2);
		n_x_ = n_x + 2 * margin;
		n_y_ = n_y + 2 * margin;
		n_z_ = n_z + 2 * margin;
		offset = n_z_*(n_y_*margin + margin) + margin;
		T.resize(n_x_*n_y_*n_z_);
		std::fill(T.begin(), T.end(), 0.);
	}

	void setMargin(int k) {
		int new_n_x_ = n_x + 2 * k;
		int new_n_y_ = n_y + 2 * k;
		int new_n_z_ = n_z + 2 * k;
		int new_margin = k;
		int new_offset = new_n_z_*(new_n_y_*new_margin + new_margin) + new_margin;
		 

		std::vector<num_t> T_tmp(new_n_x_*new_n_y_*new_n_z_);
		std::fill(T_tmp.begin(), T_tmp.end(), 0);

		for (int x = 0; x < n_x; x++) {
			for (int y = 0; y < n_y; y++) {
				for (int z = 0; z < n_z; z++) {
					T_tmp[new_n_z_*(new_n_y_*x + y) + z + new_offset] =
						T[(n_z_)*(n_y_*x + y) + z + offset];
				}
			}
		}

		T.resize(T_tmp.size());
		std::copy(T_tmp.begin(), T_tmp.end(), T.begin());
		margin += new_margin;
		offset = new_offset;
		n_x_ = new_n_x_;
		n_y_ = new_n_y_;
		n_z_ = new_n_z_;
	}

	int getMargin() const {
		return margin;
	}
	
	int getOffset() const {
		return offset;
	}

	num_t& at(int x, int y, int z) {
		return T[n_z_*(n_y_*x + y) + z + offset];
	}
	const num_t& at(int x, int y, int z) const {
		return T[n_z_*(n_y_*x + y) + z + offset];
	}
	num_t& at(const Eigen::Vector3i& p) {
		return T[n_z_*(n_y_*p(0) + p(1)) + p(2) + offset];
	}
	void at(const Eigen::Vector3i& p, num_t x) {
		//cout << p.transpose() << endl;
		cout << n_z_*(n_y_*p(0) + p(1)) + p(2) + offset << endl;
		T[n_z_*(n_y_*p(0) + p(1)) + p(2) + offset] = x;
		cout << T[n_z_*(n_y_*p(0) + p(1)) + p(2) + offset] << endl;
	}
	const num_t& at(const Eigen::Vector3i& p) const {
		return T[n_z_*(n_y_*p(0) + p(1)) + p(2) + offset];
	}
	num_t& at(int i) {
		return T[i];
	}
	const num_t& at(int i) const {
		return T[i];
	}

	int nbVoxels() const {
		return n_x*n_y*n_z;
	}

	int size() const {
		return T.size();
	}

	void normalize() {
		double x = -std::numeric_limits<double>::max();
		for (int i = 0; i < T.size(); i++) {
			x = std::max(fabs((double)(T[i])), x);
		}
		x = fabs(x);
		x += std::numeric_limits<double>::min();
		for (int i = 0; i < T.size(); i++) {
			T[i] = (num_t)((double)(T[i]) / x);
		}
	}

private:
	std::vector<num_t> T;
	int n_x;
	int n_y;
	int n_z;
	int n_x_;
	int n_y_;
	int n_z_;
	int offset;
	int margin;
};


template <typename num_t>
void conv3D(const Tensor3D<num_t>& T, const Tensor3D<num_t>& K, Tensor3D<num_t>& C) {
	Eigen::Vector3i sht = T.getShape();
	Eigen::Vector3i shk = K.getShape();
	int margin = T.getMargin();
	if ((shk(0) % 2 == 0) || (shk(1) % 2 == 0) || (shk(2) % 2 == 0)) {
		cout << "kernel size must be odd" << endl;
		return;
	}
	int rx = (shk(0) - 1) / 2;
	int ry = (shk(1) - 1) / 2;
	int rz = (shk(2) - 1) / 2;
	if ((rx > margin)||(ry > margin)||(rz > margin)) {
		cout << "kernel size out of bounds" << endl;
		return;
	}

	//for(int )
}




template <typename num_t>
void conv1D(const Tensor3D<num_t>& T, const std::vector<num_t>& k, int axis, Tensor3D<num_t>& C) {
	if (k.size() % 2 == 0) {
		cout << "kernel size must be odd" << endl;
		return;
	}
	int r = (k.size() - 1) / 2;
	if (r > T.getMargin()) {
		cout << "kernel size must be odd" << endl;
		return;
	}
	
	int d = k.size();
	Eigen::Vector3i sh = T.getShape();
	if ((C.getShape() != sh) || (C.getMargin() != T.getMargin())) {
		C.reshape(sh);
		C.setMargin(T.getMargin());
	}
	switch (axis) {
	case 0:
		for (int y = 0; y < sh(1); y++) {
			for (int z = 0; z < sh(2); z++) {
				for (int x = 0; x < sh(0); x++) {
					for (int i = 0; i < d; i++) {
						C.at(x, y, z) += k[i] * T.at(x + i - d, y, z);
					}
				}
			}
		}
		break;
	case 1:
		for (int x = 0; x < sh(0); x++) {
			for (int z = 0; z < sh(2); z++) {
				for (int y = 0; y < sh(1); y++) {
					for (int i = 0; i < d; i++) {
						C.at(x, y, z) += k[i] * T.at(x, y + i - d, z);
					}
				}
			}
		}
		break;
	case 2:
		for (int y = 0; y < sh(1); y++) {
			for (int x = 0; x < sh(0); x++) {
				for (int z = 0; z < sh(2); z++) {
					for (int i = 0; i < d; i++) {
						C.at(x, y, z) += k[i] * T.at(x, y, z + i - d);
					}
				}
			}
		}
		break;

	}
	
}
//*/
template <typename num_t>
void normalize_filter_1D(std::vector<num_t>& f) {
	num_t p = 0.0;
	num_t n = 0.0;
	for (int i = 0; i < f.size(); i++) {
		if (f[i] > 0) {
			p += f[i];
		}
		else {
			n -= f[i];
		}
	}
	p += std::numeric_limits<num_t>::min();
	n += std::numeric_limits<num_t>::min();
	num_t a = p + n;
	p /= a;
	n /= a;
	for (int i = 0; i < f.size(); i++) {
		if (f[i] > 0) {
			f[i] /= p;
		}
		else {
			f[i] /= n;
		}
	}
}

#define SIGMA_DIV 2.0
template <typename num_t>
void gaussianXX(int r, std::vector<num_t>& f) {
	f.resize(2 * r + 1);
	num_t s2 = r / SIGMA_DIV;
	num_t x;
	num_t y;
	for (int i = 0; i < f.size(); i++) {
		x = (i - r);
		y = x*x / s2;
		f[i] = (y - 1.)*exp(-y / 2.);
	}
	normalize_filter_1D(f);
}

template <typename num_t>
void gaussianX(int r, std::vector<num_t>& f) {
	f.resize(2 * r + 1);
	num_t s2 = r / SIGMA_DIV;
	num_t x;
	num_t y;
	for (int i = 0; i < f.size(); i++) {
		x = (i - r);
		y = x*x / s2;
		f[i] = (x/s2)*exp(-y / 2.);
	}
	normalize_filter_1D(f);
}

template <typename num_t>
void gaussian(int r, std::vector<num_t>& f) {
	f.resize(2 * r + 1);
	num_t s2 = r / SIGMA_DIV;
	num_t x;
	num_t y;
	for (int i = 0; i < f.size(); i++) {
		x = (i - r);
		y = x*x / s2;
		f[i] = exp(-y / 2.);
	}
	normalize_filter_1D(f);
}


template <typename num_t> 
void hessian(const Tensor3D<num_t>& T, int r, 
	Tensor3D<num_t>& hxx,
	Tensor3D<num_t>& hyy,
	Tensor3D<num_t>& hzz,
	Tensor3D<num_t>& hxy,
	Tensor3D<num_t>& hyz,
	Tensor3D<num_t>& hxz) {

	std::vector<num_t> g;
	gaussian(r, g);
	std::vector<num_t> gx;
	gaussianX(r, gx);
	std::vector<num_t> gxx;
	gaussianXX(r, gxx);


	conv1D(T, gxx, 0, hxx);
	Tensor3D<num_t> h_tmp(hxx.getShape());
	h_tmp.setMargin(hxx.getMargin());
	conv1D(hxx, g, 1, h_tmp);
	conv1D(h_tmp, g, 2, hxx);

	conv1D(T, gxx, 1, hyy);
	conv1D(hyy, g, 0, h_tmp);
	conv1D(h_tmp, g, 2, hyy);

	conv1D(T, gxx, 2, hzz);
	conv1D(hzz, g, 0, h_tmp);
	conv1D(h_tmp, g, 1, hzz);

	conv1D(T, gx, 0, hxy);
	conv1D(hxy, gx, 1, h_tmp);
	conv1D(h_tmp, g, 2, hxy);

	conv1D(T, gx, 1, hyz);
	conv1D(hyz, gx, 2, h_tmp);
	conv1D(h_tmp, g, 0, hyz);

	conv1D(T, gx, 0, hxz);
	conv1D(hxz, gx, 2, h_tmp);
	conv1D(h_tmp, g, 1, hzz);

}
//*/



void clean_point_cloud(const Eigen::MatrixXd& V_in, Eigen::MatrixXd& V_out, double r) {
	std::vector<bool> selected(V_in.rows());
	std::fill(selected.begin(), selected.end(), true);
	kdtree3d<double> kd(V_in);
	std::vector<int> ret_idx;
	std::vector<double> dist_sqr;
	//Eigen::Vector3d v;


	for (int i = 0; i < selected.size(); i++) {
		if (selected[i]) {
			kd.radiusSearch(V_in.row(i).transpose(), r, ret_idx);
			for (int i = 1; i < ret_idx.size(); i++) {
				selected[i] = false;
			}
		}
	}
	int nb_selected = 0;
	for (int i = 0; i < selected.size(); i++) {
		if (selected[i]) {
			nb_selected++;
		}
	}

	V_out.resize(nb_selected, 3);
	for (int i = 0; i < selected.size(); i++) {
		if (selected[i]) {
			V_out.row(i) = V_in.row(i);
		}
	}
}

double bounding_box(const Eigen::MatrixXd& V, Eigen::Matrix<double, 3, 2>& bnd_box) {
	for (int i = 0; i < 3; i++) {
		bnd_box(i, 0) = std::numeric_limits<double>::max();
		bnd_box(i, 1) = std::numeric_limits<double>::min();
	}
	
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < V.rows(); j++) {
			bnd_box(i, 0) = std::min(V(j,i), bnd_box(i, 0));
			bnd_box(i, 1) = std::max(V(j, i), bnd_box(i, 1));
		}
	}
	double dx = bnd_box(0, 1) - bnd_box(0, 0);
	double dy = bnd_box(1, 1) - bnd_box(1, 0);
	double dz = bnd_box(2, 1) - bnd_box(2, 0);


	/*cout << "dx: " <<  << endl;
	cout << "dy: " <<  << endl;
	cout << "dz: " <<  << endl;*/

	return std::max(dx, std::max(dy, dz));
}

inline Eigen::Vector3i voxel(const Eigen::Vector3d& v, 
	const Eigen::Vector3d& m,
	const Eigen::Vector3d& d,
	const Eigen::Vector3i shape) {
	Eigen::Vector3i res;
	for (int i = 0; i < 3; i++) {
		res(i) = int((shape(i)*(v(i) - m(i))) / d(i));
		res(i) = std::max(0, res(i));
		res(i) = std::min(shape(i), res(i));
	}
	return res;
}


void voxelize(const Eigen::MatrixXd& V, Tensor3D<float>& T, float delta) {
	Eigen::Matrix<double, 3, 2> bnd_box;
	bounding_box(V, bnd_box);
	Eigen::Vector3d d;
	Eigen::Vector3d m;
	for (int i = 0; i < 3; i++) {
		d(i) = (bnd_box(i, 1) - bnd_box(i, 0));
		m(i) = bnd_box(i, 0);
	}
	int nx = int(d(0) / delta) + 1;
	int ny = int(d(1) / delta) + 1;
	int nz = int(d(2) / delta) + 1;
	cout << "aa" << endl;
	T.reshape(nx, ny, nz);
	cout << "bb" << endl;
	bool n = true;
	cout << T.getShape().transpose() << endl;
	cout << T.getOffset() << endl;
	cout << T.getMargin() << endl;
	
	if (n) {
		for (int i = 0; i < V.rows(); i++) {
			//cout << voxel(V.row(i).transpose(), m, T.getShape(), delta).transpose() << endl;
			//cout << (float)(T.at(voxel(V.row(i).transpose(), m, delta))) << endl;
			T.at(voxel(V.row(i).transpose(), m, d, T.getShape()), 1.);
				//std::max((float)(T.at(voxel(V.row(i).transpose(), m, T.getShape(), delta))), float(1.0));
		}
	}
	else {
		for (int i = 0; i < V.rows(); i++) {
			T.at(voxel(V.row(i).transpose(), m, d, T.getShape())) = 
				T.at(voxel(V.row(i).transpose(), m, d, T.getShape()))+1.;
		}
	}
	T.normalize();
}

inline void symMat3d(double m00, double m11, double m22, double m01, double m12, double m02, Eigen::Matrix3d& M) {
	M << m00, m01, m02,
		m01, m11, m12,
		m02, m12, m22;
}
inline float veselness(const Eigen::Vector3d& eigs) {
	std::vector<double> l(3);
	for (int i = 0; i < 3; i++) {
		l[i] = fabs(eigs(i)) + std::numeric_limits<double>::min();
	}
	std::sort(l.begin(), l.end());
	double Ra = l[1] / l[2];
	double Rb = l[0] / sqrt(l[1] * l[2]);
	double S2 = eigs.squaredNorm();
	return (1. - exp(-2.*Ra*Ra))*exp(-2.*Rb*Rb)*(1. - exp(-2.*S2));
}


void Vesselness(int min_r, int max_r, int step, Tensor3D<float>& T, Tensor3D<unsigned char>& V) {
	Tensor3D<float> hxx(T.getShape());
	hxx.setMargin(max_r);
	Tensor3D<float> hyy(T.getShape());
	hyy.setMargin(max_r);
	Tensor3D<float> hzz(T.getShape());
	hzz.setMargin(max_r);
	Tensor3D<float> hxy(T.getShape());
	hxy.setMargin(max_r);
	Tensor3D<float> hyz(T.getShape());
	hyz.setMargin(max_r);
	Tensor3D<float> hxz(T.getShape());
	hxz.setMargin(max_r);
	Tensor3D<float> V_tmp(T.getShape());
	V_tmp.setMargin(max_r);
	T.setMargin(max_r);
	V.reshape(T.getShape());
	V.setMargin(T.getMargin());

	

	int nb_step = (max_r - min_r) / step;
	Eigen::Vector3d eigs;
	Eigen::Matrix3d eigvec;
	Eigen::Matrix3d H;
	for (int j = 0; j < nb_step; j++) {
		hessian<float>(T, min_r + j*step, hxx, hyy, hzz, hxy, hyz, hxz);
		for (int i = 0; i < T.size(); i++) {
			symMat3d(hxx.at(i), hyy.at(i), hzz.at(i), hxy.at(i), hyz.at(i), hxz.at(i), H);
			Eigs3D(H, eigs, eigvec);
			V_tmp.at(i) = std::max(V_tmp.at(i), veselness(eigs));
		}
	}
	V_tmp.normalize();

	for (int i = 0; i < T.size(); i++) {
		V.at(i) = (unsigned char)(255.*V_tmp.at(i));
	}
}//*/

template <typename num_t>
void save_Tensor3D(const std::string& path, const std::string& name, const Tensor3D<num_t>& T) {
	ofstream myfile;
	int nx = T.getShape()(0);
	int ny = T.getShape()(1);
	int nz = T.getShape()(2);
	myfile.open(path + "/" + name + "_" + std::to_string(nx) + "_" +
		std::to_string(ny) + "_" + std::to_string(nz) + ".raw" );
	int x;
	int y;
	int z;

	for (int i = 0; i < T.nbVoxels(); i++) {
		z = i % nz;
		y = ((i - z) / nz) % ny;
		x = (i - nz*y - z) / (nz*ny);
		myfile << T.at(x,y,z) << endl;
	}
	myfile.close();
}

template <typename num_t>
void save_Tensor3D_slices(const std::string& path, const std::string& name, const Tensor3D<num_t>& T) {
	ofstream myfile;
	int nx = T.getShape()(0);
	int ny = T.getShape()(1);
	int nz = T.getShape()(2);
	myfile.open(path + "/" + name + "_" + std::to_string(nx) + "_" +
		std::to_string(ny) + "_" + std::to_string(nz) + ".raw");
	int x;
	int y;
	int z;
	for (int z = 0; z < nz; z++) {
		for (int y = 0; y < ny; y++) {
			for (int x = 0; x < nx; x++) {
				myfile << T.at(x, y, z) << " ";
			}
			myfile << endl;
		}
		myfile << endl;
	}
	//myfile.close();
}

void test_vesselness(const std::string& path) {
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	if (!igl::readOFF(path, V, F))
	{
		cout << "failed to load mesh" << endl;
		exit(666);
	}
	Tensor3D<float> T;
	Eigen::Matrix<double, 3, 2> bnd_box;
	double delta = bounding_box(V, bnd_box)/60.;
	voxelize(V, T, delta);
	//cout << T.getShape().transpose() << endl;
	//T.reshape(3, 3, 3);
	//T.at(1, 1, 1) = 1.0;
	save_Tensor3D_slices("C:/Users/Adrien/Documents/fiji-win64/imgs", "test", T);
}







#endif // !VOXELIZATION_H
