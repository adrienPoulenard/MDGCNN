#pragma once
#ifndef UTILS_H
#define UTILS_H
#include <iostream>
#include <fstream>
#include <experimental/filesystem>
#include <windows.h>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <stdlib.h>     /* srand, rand */
#include "kdtree3d.h"


using namespace std;
namespace fs = std::experimental::filesystem;

inline double angle_(double c, double s) {
	//cout << "c: " << c << " s: " << s << endl;
	/*if (s >= 0.0) {
	return acos(c);
	}
	else {
	return 2.*M_PI - acos(c);
	}*/
	double a = atan2(s, c);
	if (a < 0.) {
		a += 2.*M_PI;
	}
	return a;
}


bool has_suffix(const string& s, const string& suffix)
{
	return (s.size() >= suffix.size()) && equal(suffix.rbegin(), suffix.rend(), s.rbegin());
}

std::string remove_last(const std::string& file_name, int n) {
	return file_name.substr(0, file_name.length() - n);
}

std::string convert_to_string(const wchar_t* c) {
	std::wstring ws(c);
	// your new String
	string str(ws.begin(), ws.end());
	return str;
}



// list files with a given extension
bool getFilesList_(const std::string& filePath, const std::string& extension, std::vector<std::string> & returnFileName)
{
	/*WIN32_FIND_DATA fileInfo;
	HANDLE hFind;
	string  fullPath = filePath + extension;
	hFind = FindFirstFile(fullPath.c_str(), &fileInfo);
	if (hFind != INVALID_HANDLE_VALUE) {
	returnFileName.push_back(fileInfo.cFileName);
	while (FindNextFile(hFind, &fileInfo) != 0) {
	returnFileName.push_back(fileInfo.cFileName);
	cout << fileInfo.cFileName << endl;
	}
	return true;
	}
	return false;*/
	std::string name;
	returnFileName.clear();
	for (auto& p : fs::recursive_directory_iterator(filePath))
	{
		if (p.path().extension() == extension) {
			//std::string name((p.path().c_str()));
			//name = std::string();
			name = remove_last(convert_to_string(p.path().filename().c_str()), extension.length());
			returnFileName.push_back(name);
			//convert_to_string(p.path().filename().c_str());

			//cout << remove_last(convert_to_string(p.path().filename().c_str()), extension.length()) << endl;
		}

	}
	return true;

}//*/

bool dirExists(const std::string& dirName_in)
{
	DWORD ftyp = GetFileAttributesA(dirName_in.c_str());
	if (ftyp == INVALID_FILE_ATTRIBUTES)
		return false;  //something is wrong with your path!

	if (ftyp & FILE_ATTRIBUTE_DIRECTORY)
		return true;   // this is a directory!

	return false;    // this is not a directory!
}

template <typename T>
void load_vector_(const std::string& path, std::vector< T >& v) {
	ifstream myfile;
	myfile.open(path);
	T value;
	v.clear();
	while (myfile >> value) {
		v.push_back(value);
	}
	myfile.close();
}

template <typename T>
bool save_vector_(const std::string& path, const std::vector< T >& v) {
	ofstream myfile;
	myfile.open(path);
	if (myfile.is_open())
	{
		for (int i = 0; i < v.size(); i++) {
			myfile << v[i] << endl;
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
void save_matrix_(const std::string& path, const Eigen::Matrix<T, -1, -1>& M) {
	ofstream myfile;
	myfile.open(path);
	for (int i = 0; i < M.rows(); i++) {
		for (int j = 0; j < M.cols(); j++) {
			myfile << std::to_string((T)(M(i, j))) << " ";
		}
		myfile << "\n";
	}
	myfile.close();
}

template <typename T>
void load_matrix_(const std::string& path, Eigen::Matrix<T, -1, -1>& M) {
	int nrows = M.rows();
	int ncols = M.cols();
	std::vector<T> v;
	load_vector_(path,v);
	for (int i = 0; i < M.rows(); i++) {
		for (int j = 0; j < M.cols(); j++) {
			M(i, j) = v[ncols*i + j];
		}	
	}

}

void load_matrix_(const std::string& path, int nrows, int ncols, Eigen::MatrixXd& M) {
	M.resize(nrows, ncols);
	std::vector<double> v;
	load_vector_(path, v);
	for (int i = 0; i < M.rows(); i++) {
		for (int j = 0; j < M.cols(); j++) {
			M(i, j) = v[ncols*i + j];
		}
	}
}

void display_matrix(const Eigen::MatrixXd& M) {
	for (int i = 0; i < M.rows(); i++) {
		for (int j = 0; j < M.cols(); j++) {
			cout << M(i, j) << " ";
		}
		cout << endl;
	}
}



void centerObject(Eigen::MatrixXd& V) {
	Eigen::Vector3d v(0.0, 0.0, 0.0);
	int nv = V.rows();
	for (int i = 0; i < nv; i++) {
		v += V.row(i);
	}
	v /= nv;
	/*double mean_rad = 0.0;
	for (int i = 0; i < nv; i++) {
	V.row(i) -= v;
	mean_rad += V.row(i).norm();
	}
	mean_rad /= nv;*/
	for (int i = 0; i < nv; i++) {
		V.row(i).normalize();
	}
	//return mean_rad;
}

void concat_matrices(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, Eigen::MatrixXd& M) {
	int nrows = std::min(A.rows(), B.rows());
	int ncolsA = A.cols();
	int ncolsB = B.cols();

	M.resize(nrows, ncolsA + ncolsB);

	for (int i = 0; i < nrows; i++) {
		for (int j = 0; j < ncolsA; j++) {
			M(i, j) = A(i, j);
		}
		for (int j = 0; j < ncolsB; j++) {
			M(i, j + ncolsA) = B(i, j);
		}
	}
}

float random_real() {
	float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
	return r;
}

inline void Eigs3D(const Eigen::Matrix3d& M, Eigen::Vector3d& eigs, Eigen::Matrix3d& eigvec) {
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(3);
	es.computeDirect(M);
	eigvec = es.eigenvectors();
	eigs = es.eigenvalues();
}

void loadRGB(const std::string& path, int nv, Eigen::MatrixXd& C) {
	Eigen::MatrixXd M;
	
	load_matrix_(path, 3*nv, 1, M);
	C.resize(nv, 3);
	for (int i = 0; i < nv; i++) {
		C(i, 0) = M(3 * i + 0, 0) / 255.0;
		C(i, 1) = M(3 * i + 1, 0) / 255.0;
		C(i, 2) = M(3 * i + 2, 0) / 255.0;
	}
}

/*
void bilinear_interpolation_RGB(Eigen::Vector3d& c, double x, double y,
	int w, int h,
	const std::vector<std::uint8_t>& img) {
	c.setZero();
	//double x_;
	//double y_;
	double a = 0;
	double b = 0;
	double c = 0;
	double d = 0;
	int X;
	int Y;
	double x1;
	double x2;
	double y1;

	if ((x >= 0) && (y >= 0.) && (x < w) && (y < h)) {
		X = int(std::floor(x));
		x -= X;
		x += 0.5;
		Y = int(std::floor(x));
		y -= Y;
		y += 0.5;
		a = 
	}
	
	if()
}*/

void normalize_shape(Eigen::MatrixXd& V) {
	int nv = V.rows();
	Eigen::Vector3d c;
	c.setZero();
	for (int i = 0; i < nv; i++) {
		c += V.row(i).transpose();
	}
	c /= nv;
	double d = 0.0;
	for (int i = 0; i < nv; i++) {
		V.row(i) -= c.transpose();
		d += V.row(i).squaredNorm();
	}
	d /= nv;
	d = sqrt(d);
	V /= d;
}

void display_progress(float progress) {
	int barWidth = 70;

	std::cout << "[";
	int pos = barWidth * progress;
	for (int i = 0; i < barWidth; ++i) {
		if (i < pos) std::cout << "=";
		else if (i == pos) std::cout << ">";
		else std::cout << " ";
	}
	std::cout << "] " << int(progress * 100.0) << " %\r";
	std::cout.flush();	
}

template <typename T>
void triangles_to_vertices(const std::vector<T>& t, const Eigen::MatrixXi& F, int nv, std::vector<T>& v) {
	v.resize(nv);
	for (int i = 0; i < F.rows(); i++) {
		v[F(i, 0)] = t[i];
		v[F(i, 1)] = t[i];
		v[F(i, 2)] = t[i];
	}
}

void find_nn_correspondances(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
	const Eigen::MatrixXd& NV, const Eigen::MatrixXi& NF, 
	std::vector<int>& idx) {

	kdtree3d<double> v_kd(V);

	std::vector<size_t> ret_indexes;
	std::vector<double> out_dists_sqr;
	idx.resize(NV.rows());
	for (int i = 0; i < NV.rows(); i++) {
		v_kd.knnSearch(NV.row(i).transpose(), 1, ret_indexes, out_dists_sqr);
		idx[i] = ret_indexes[0];
	}
}

template <typename T>
void find_nn_correspondances(const std::string& shape, const std::string& new_shape,
	const std::string& signal, const std::string& new_signal) {

	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	Eigen::MatrixXd NV;
	Eigen::MatrixXi NF;
	Eigen::Matrix<T, -1, -1> S;
	Eigen::Matrix<T, -1, -1> NS;
	std::vector<int> idx;

	if (!igl::readOFF(shape, V, F)) {
		cout << shape << " not found" << endl;
	}
	if (!igl::readOFF(new_shape, NV, NF)) {
		cout << new_shape << " not found" << endl;
	}

	find_nn_correspondances(V, F, NV, NF, idx);

	//cout << signal << endl;
	//load_matrix_(signal, S);
	std::vector<int> s;
	std::vector<int> ns;
	load_vector_(signal, s);
	//cout << S.rows() << " u " << s.size() << endl;
	//NS.resize(idx.size(), S.cols());
	ns.resize(idx.size());
	for (int i = 0; i < idx.size(); i++) {
		//NS.row(i) = S.row(idx[i]);
		ns[i] = s[idx[i]];
	}
	//save_matrix_(new_signal, NS);*/
	save_vector_(new_signal, ns);
}

#endif // !UTILS_H
