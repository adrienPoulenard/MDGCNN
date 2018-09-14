#pragma once
#ifndef GRID_MESH_H
#define GRID_MESH_H

#include <fstream>
#include <iostream>
#include <patch_op.h>
//#include <mnist/mnist_reader_less.hpp>
#include <cifar/cifar10_reader.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <igl/writeOFF.h>
#include "utils.h"
#include "create_dataset.h"
#include <igl/connect_boundary_to_infinity.h>
#include <igl/decimate.h>

using namespace std;

/*void create_grid(int w, int h, Eigen::MatrixXd& V, Eigen::MatrixXi& F, double ratio_x=1.0, double ratio_y=1.0) {
	//Eigen::MatrixXd V;
	//Eigen::MatrixXi F;

	V.resize((w+1)*(h+1), 3);
	F.resize((w)*(h) * 2, 3);
	int i_;
	int j_;

	for (int i = 0; i < w; i++) {
		for (int j = 0; j < h; j++) {
			i_ = (i + 1) % (w + 1);
			j_ = (j + 1) % (h + 1);
			if (j % 2 == 1) {
				F(2 * (h*i + j) + 0, 0) = (h + 1)*i + j;
				F(2 * (h*i + j) + 0, 1) = (h + 1)*i + j_;
				F(2 * (h*i + j) + 0, 2) = (h + 1)*(i_) + j_;
				
				F(2 * (h*i + j) + 1, 0) = (h + 1)*(i_) + j_;
				F(2 * (h*i + j) + 1, 1) = (h + 1)*(i_) + j;
				F(2 * (h*i + j) + 1, 2) = (h + 1)*i + j;	
			}
			else {
				F(2 * (h*i + j) + 0, 0) = (h + 1)*i + j;
				F(2 * (h*i + j) + 0, 1) = (h + 1)*i + j_;
				F(2 * (h*i + j) + 0, 2) = (h + 1)*(i_) + j;

				F(2 * (h*i + j) + 1, 0) = (h + 1)*(i) + j_;
				F(2 * (h*i + j) + 1, 1) = (h + 1)*(i_) + j_;
				F(2 * (h*i + j) + 1, 2) = (h + 1)*i_ + j;
			}
		}
	}

	for (int i = 0; i < w+1; i++) {
		for (int j = 0; j < h+1; j++) {
			if (j % 2 == 1) {
				V((h + 1)*i + j, 0) = (i+0.5)*ratio_x;
				V((h + 1)*i + j, 1) = j*ratio_y;
				V((h + 1)*i + j, 2) = 0.;
			}
			else {
				V((h + 1)*i + j, 0) = i*ratio_x;
				V((h + 1)*i + j, 1) = j*ratio_y;
				V((h + 1)*i + j, 2) = 0.;
			}
		}
	}


	//igl::connect_boundary_to_infinity(V, F, V_O, F_O);
}//*/

void create_grid(int w, int h, Eigen::MatrixXd& V, Eigen::MatrixXi& F, double ratio_x = 1.0, double ratio_y = 1.0) {
	//Eigen::MatrixXd V;
	//Eigen::MatrixXi F;

	V.resize((w + 1)*(h + 1), 3);
	F.resize((w)*(h) * 2, 3);
	int i_;
	int j_;

	for (int i = 0; i < w; i++) {
		for (int j = 0; j < h; j++) {
			i_ = (i + 1) % (w + 1);
			j_ = (j + 1) % (h + 1);
			
			F(2 * (h*i + j) + 0, 0) = (h + 1)*i + j;
			F(2 * (h*i + j) + 0, 1) = (h + 1)*i + j_;
			F(2 * (h*i + j) + 0, 2) = (h + 1)*(i_)+j + 1;

			F(2 * (h*i + j) + 1, 0) = (h + 1)*(i_)+j_;
			F(2 * (h*i + j) + 1, 1) = (h + 1)*(i_)+j;
			F(2 * (h*i + j) + 1, 2) = (h + 1)*i + j;
		}
	}

	for (int i = 0; i < w + 1; i++) {
		for (int j = 0; j < h + 1; j++) {
			V((h + 1)*i + j, 0) = i*ratio_x;
			V((h + 1)*i + j, 1) = j*ratio_y;
			V((h + 1)*i + j, 2) = 0.;
		}
	}
	//igl::connect_boundary_to_infinity(V, F, V_O, F_O);
}//*/

void save_grid_mesh(int w, int h, const std::string& path, double ratio_x=1.0, double ratio_y=1.0) {
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	create_grid(w, h, V, F, ratio_x, ratio_y);
	if (!igl::writeOFF(path, V, F)) {
		cout << "unable to write file " << path << endl;
	}
}


/*int GridUV(const std::string& shape_path, 
	const std::string& shape_name, 
	const std::string& UV_path,
	double offset) {
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	if (!igl::readOFF(shape_path + "/" + shape_name + ".off", V, F)) {
		cout << "unable to write file " << shape_path + "/" + shape_name + ".off" << endl;
	}
	Eigen::MatrixXd UV(V.rows(), 2);
	UV.setZero();

	double m = 0;
	for (int i = 0; i < V.rows(); i++) {
		if (std::max(V(i, 0), V(i, 0)) > m) {
			m = std::max(V(i, 0), V(i, 0));
		}
	}
	offset *= m;
	cout << "zzzzzzzzz" << endl;
	for (int i = 0; i < V.rows(); i++) {
		cout << "offset " << offset << endl;
		UV(i, 0) = V(i, 0) - offset;
		UV(i, 1) = V(i, 1) - offset;
	}
	save_matrix_(UV_path + "/" + shape_name, UV);
	return V.rows();
}*/


int GridUV(const Eigen::MatrixXd& V,
	const Eigen::MatrixXi& F,
	const std::string& path,
	double offset) {
	
	Eigen::MatrixXd UV(V.rows(), 2);
	UV.setZero();

	/*double m = 0;
	for (int i = 0; i < V.rows(); i++) {
		if (std::max(V(i, 0), V(i, 0)) > m) {
			m = std::max(V(i, 0), V(i, 0));
		}
	}
	
	offset *= m;*/
	for (int i = 0; i < V.rows(); i++) {
		UV(i, 0) = V(i, 0) - offset;
		UV(i, 1) = V(i, 1) - offset;
	}
	save_matrix_(path, UV);
	return V.rows();
}


void simplifyGrid(int w, int h, double X, double Y, double ratio, 
	Eigen::MatrixXd& U, Eigen::MatrixXi& G, Eigen::VectorXi& I) {
	int w_o = int( w * ratio );
	int h_o = int( h * ratio );
	double ratio_x = X / w_o;
	double ratio_y = Y / h_o;
	create_grid(w_o, h_o, U, G, ratio_x, ratio_y);

	
	// find parent vertices:
	I.resize((w_o+1)*(h_o+1));
	int i_;
	int j_;
	I.setZero();
	for (int i = 0; i < w_o + 1; i++) {
		i_ = std::min(w, int(i/ratio));
		for (int j = 0; j < h_o + 1; j++) {
			j_ = std::min(h, int(j/ratio));
			I((h_o + 1)*i + j) = (h + 1)*i_ + j_;
		}
	}
}


bool computeGridFilterMaper_(
	const std::vector<double>& ratios_,
	const std::vector<double>& radius,
	const std::vector<int>& nrings,
	const std::vector<int>& ndirs,
	int w,
	int h,
	double ratio_x,
	double ratio_y,
	const std::string& dataset_path) {

	PatchConvOperator* mapper1 = NULL;
	PatchConvOperator* mapper2 = NULL;
	std::string name;

	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	Eigen::MatrixXd U;
	Eigen::MatrixXi G;
	Eigen::VectorXi I;

	std::vector<double> angular_shift_;
	std::vector<int> parent_vertex;
	double ratio;

	int w_o = w;
	int h_o = h;

	std::vector<double> ratios(radius.size());
	std::vector<double> ratios_square(radius.size());
	ratios[0] = 1.0;
	if (ratios_.size() != ratios.size()) {
		//cout << "uuuuuuuu" << endl;
		//cout << ratios.size() << " " << ratios_.size() << endl;
		for (int i = 1; i < radius.size(); i++) {
			ratios[i] = ratios_[i - 1];
			ratios_square[i] = ratios_[i - 1] * ratios_[i - 1];
		}
	}
	else {
		for (int i = 0; i < radius.size(); i++) {
			ratios[i] = ratios_[i];
			ratios_square[i] = ratios_[i] * ratios_[i];
		}
	}
	std::string shape_name = "grid_" + std::to_string(w) + "x" + std::to_string(h);
	create_grid(w, h, V, F, ratio_x, ratio_y);
	double X = w * ratio_x;
	double Y = h * ratio_y;
	mapper1 = new  PatchConvOperator(V, F);
	mapper1->compute(ndirs[0], nrings[0], radius[0]);

	saveFilterMapper(radius[0], nrings[0], ndirs[0],
		shape_name + "_ratio=" + std::to_string(ratios_square[0])
		+ "_nv=" + std::to_string(V.rows()),
		dataset_path,
		*mapper1);

	for (int j = 0; j < ratios.size() - 1; j++) {
		ratio = ratios[j + 1] / ratios[j];
		// ratio = ratios[j + 1];
		// igl::decimate(V, F, (size_t)(F.rows()*ratio), U, G, J, I);

		simplifyGrid(w_o, h_o, X, Y, ratio, U, G, I);
		w_o *= ratio;
		h_o *= ratio;
		

		mapper2 = new PatchConvOperator(U, G);
		mapper2->compute(ndirs[j + 1], nrings[j + 1], radius[j + 1], true);
		saveFilterMapper(radius[j + 1], nrings[j + 1], ndirs[j + 1],
			shape_name + "_ratio=" + std::to_string(ratios_square[j + 1])
			+ "_nv=" + std::to_string(U.rows()),
			dataset_path,
			*mapper2);

		parent_vertex.resize(I.rows());
		for (int k = 0; k < parent_vertex.size(); k++) {
			parent_vertex[k] = I(k);
		}
		//index_shift(mapper1->Gpc(), mapper2->Gpc(), I, ndirs[j], angular_shift);
		angular_shift(mapper1->Gpc(), mapper2->Gpc(), I, angular_shift_);
		savePoolingOp(j, ratios_square, ndirs, shape_name, dataset_path, parent_vertex, angular_shift_);
		std::swap(mapper1, mapper2);
		delete mapper2;
		mapper2 = NULL;
		//V = U;
		//F = G;
	}
	delete mapper1;
	mapper1 = NULL;

	return true;
}//*/

void grid(const std::string& dataset, double r_, int nrings_, int ndirs_, double inv_ratio,
	const std::string& grid_path, int w, int h, double ratio_x, double ratio_y, double offset) {

	/*int nv = GridUV(shape_path, shape_name,
		"C:/Users/Adrien/Documents/Datasets/grid_mesh/uv_coordinates",
		0.05);*/

	Eigen::MatrixXd V;
	Eigen::MatrixXi F;

	create_grid(w, h, V, F, ratio_x, ratio_y);
	double X = w * ratio_x;
	double Y = w * ratio_y;
	std::string name = "grid_" + std::to_string(w) + "x" + std::to_string(h);
	if (!igl::writeOFF(grid_path + "/" + name + ".off", V, F)) {
		cout << "unable to write file at " << grid_path << endl;
	}

	GridUV(V, F, dataset + "/uv_coordinates/" + name + ".txt", offset);

	double inv_decay_ratio = inv_ratio;
	std::vector<double> ratios(3);
	ratios[0] = 1.0;
	std::vector<int> nrings(3);
	std::vector<int> ndirs(3);
	for (int i = 0; i < 3; i++) {
		nrings[i] = nrings_;
		ndirs[i] = ndirs_;
	}
	for (int i = 0; i < 2; i++) {
		ratios[i + 1] = ratios[i] / sqrt(inv_decay_ratio);
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

	computeGridFilterMaper_(ratios, radius, nrings, ndirs, w, h, ratio_x, ratio_y, dataset);

}




void test_bnd(const std::string& path) {
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	if (!igl::readOFF(path, V, F)) {
		cout << "unable to write file " << path << endl;
	}
	geodesic::Mesh mesh;
	if (!loadMesh(path, mesh)) {
		exit(666);
	}
	GPC gpc(mesh);
	int nv = V.rows();
	std::vector<double> func(nv);
	for (int i = 0; i < nv; i++) {
		if (gpc.isMeshBoundaryPoint(i)) {
			func[i] = 1.0;
		}
		else {
			func[i] = 0.0;
		}
	}
	visualize(V, F, func);
}

void test_pt_id(const std::string& path, int p) {
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	if (!igl::readOFF(path, V, F)) {
		cout << "unable to write file " << path << endl;
	}
	int nv = V.rows();
	std::vector<double> func(nv);
	for (int i = 0; i < nv; i++) {
		func[i] = 0.;
	}
	func[p] = 1.0;
	visualize(V, F, func);
}

#endif


