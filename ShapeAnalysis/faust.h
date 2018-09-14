#pragma once
#ifndef FAUST_H
#define FAUST_H
#include <iostream>
#include <fstream>

//#include <igl/viewer/Viewer.h>
//#include <igl/parula.h>
//#include <igl/read_triangle_mesh.h>
#include <igl/readOFF.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "cube_segmentation.h"
#include "create_dataset.h"
#include "permute_dataset.h"
#include "utils.h"



void symmetrize_labels(std::vector< int >& sym, std::vector< int >& labels) {
	int nv = labels.size();
	for (int i = 0; i < nv; i++) {
		if (labels[sym[i]] > labels[i]) {
			labels[sym[i]] = labels[i];
		}
	}
	squeeze_labels(labels);
}

void faustXYZ(const std::string& faust_path, const std::string& signal_path) {
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	std::string path = "C:/Users/Adrien/Documents/FAUST_shapes_off/tr_reg_000.off";
	if (!igl::readOFF(path, V, F))
	{
		cout << "failed to load mesh" << endl;
		exit(666);
	}
	std::vector< std::string > shapes;
	getFilesList_(faust_path, ".off", shapes);
	std::string name;
	cout << shapes.size() << endl;
	for (int i = 0; i < shapes.size(); i++) {
		//name = shapes[i].substr(0, shapes[i].length() - 4);
		//cout << name << endl;
		name = shapes[i];
		save_matrix_(signal_path + "/" + name + ".txt", V);
	}
}

void faust_vertices_labels(const std::string& faust_path, const std::string& labels_path) {
	std::vector< int > labels(6890);
	for (int i = 0; i < 6890; i++) {
		labels[i] = i;
	}
	std::vector< std::string > shapes;
	getFilesList_(faust_path, ".off", shapes);
	std::string name;
	cout << shapes.size() << endl;
	for (int i = 0; i < shapes.size(); i++) {
		//name = shapes[i].substr(0, shapes[i].length() - 4);
		//cout << name << endl;
		name = shapes[i];
		save_vector_<int>(labels_path + "/" + name + ".txt", labels);
	}
}

void faust_voxel_labels(const std::string& faust_path, const std::string& labels_path) {
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	std::string path = "C:/Users/Adrien/Documents/FAUST_shapes_off/tr_reg_070.off";
	if (!igl::readOFF(path, V, F))
	{
		cout << "failed to load mesh" << endl;
		exit(666);
	}

	float rx = 1.0;
	float ry = 1.0;
	float rz = 1.0;
	int nx = 10; //4
	int ny = 14;
	int nz = 1;
	float dx = 0.00;
	float eps = 0.000001;
	box3d bnd_box;
	get_bounding_box(V, bnd_box);
	cout << "Dx " << bnd_box.x_max - bnd_box.x_min << endl;
	cout << "Dy " << bnd_box.y_max - bnd_box.y_min << endl;
	std::vector< int > labels;
	voxelSegmentation(V, bnd_box.x_min - eps, bnd_box.x_max + dx + eps,
		bnd_box.y_min - eps, bnd_box.y_max + eps,
		bnd_box.z_min - eps, bnd_box.z_max + eps,
		nx, ny, nz,
		labels);

	/*std::vector< int > sym;
	load_vector_<int>(faust_path + "/" + "faust_sym.txt", sym);
	for (int i = 0; i < sym.size(); i++) {
		//cout << sym[i] << endl;
	}
	symmetrize_labels(sym, labels);*/

	std::vector< std::string > shapes;
	getFilesList_(faust_path, ".off", shapes);
	std::string name;
	cout << shapes.size() << endl;
	for (int i = 0; i < shapes.size(); i++) {
		//name = shapes[i].substr(0, shapes[i].length() - 4);
		//cout << name << endl;
		name = shapes[i];
		save_vector_<int>(labels_path + "/" + name + ".txt" , labels);
	}
}

void faust_left_right(const std::string& faust_path, const std::string& desc_path, 
					  const std::string& permutations_path) {
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	std::string path = "C:/Users/Adrien/Documents/FAUST_shapes_off/tr_reg_070.off";
	if (!igl::readOFF(path, V, F))
	{
		cout << "failed to load mesh" << endl;
		exit(666);
	}
	int nv = V.rows();
	Eigen::MatrixXd f(nv, 2);
	f.setZero();
	for (int i = 0; i < nv; i++) {
		if (V(i, 0) < 0.0) {
			f(i, 0) = 1.0;
		}
		else {
			f(i, 1) = 1.0;
		}
	}


	Eigen::MatrixXd f_perm(nv, 2);

	std::vector< std::string > shapes;
	getFilesList_(faust_path, ".off", shapes);
	std::string name;
	std::vector<int> permutation;
	for (int i = 0; i < shapes.size(); i++) {
		//name = shapes[i].substr(0, shapes[i].length() - 4);
		//cout << name << endl;
		name = shapes[i];
		load_vector_<int>(permutations_path + "/" + name + ".txt", permutation);
		permuteMatrix(permutation, f, f_perm);
		save_matrix_(desc_path + "/" + name + ".txt", f_perm);
	}
}

void faust_left_right(const std::string& faust_path, const std::string& desc_path) {
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	std::string path = faust_path + "/tr_reg_070.off";
	if (!igl::readOFF(path, V, F))
	{
		cout << "failed to load mesh" << endl;
		exit(666);
	}
	int nv = V.rows();
	std::vector<int> f(nv);

	for (int i = 0; i < nv; i++) {
		if (V(i, 0) < 0.0) {
			f[i] = 0;
		}
		else {
			f[i] = 1;
		}
	}

	std::vector< std::string > shapes;
	getFilesList_(faust_path, ".off", shapes);
	std::string name;
	
	for (int i = 0; i < shapes.size(); i++) {
		//name = shapes[i].substr(0, shapes[i].length() - 4);
		//cout << name << endl;
		name = shapes[i];
		save_vector_(desc_path + "/" + name + ".txt", f);
	}
}

void faust3d(const std::string& faust_path, const std::string& desc_path) {
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	std::string path = faust_path + "/tr_reg_070.off";

	std::vector< std::string > shapes;
	getFilesList_(faust_path, ".off", shapes);
	std::string name;

	for (int i = 0; i < shapes.size(); i++) {
		//name = shapes[i].substr(0, shapes[i].length() - 4);
		//cout << name << endl;
		name = shapes[i];
		path = faust_path + "/" + name + ".off";
		if (!igl::readOFF(path, V, F))
		{
			cout << "failed to load mesh" << endl;
			exit(666);
		}
		
		save_matrix_(desc_path + "/" + name + ".txt", V);
	}
}

void faust_train_test(const std::string& faust_path, const std::string& dataset_path) {
	int split = 80;
	std::vector< std::string > shapes;
	getFilesList_(faust_path, ".off", shapes);


	ofstream myfile;
	myfile.open(dataset_path + "/" + "train.txt");
	if (myfile.is_open())
	{
		for (int i = 0; i < split; i++) {
			myfile << shapes[i] << endl;
		}
		myfile.close();
	}
	else {
		cout << "Unable to open file: " << dataset_path + "/" + "train.txt" << endl;
	}

	myfile.open(dataset_path + "/" + "test.txt");
	if (myfile.is_open())
	{
		for (int i = split; i < shapes.size(); i++) {
			myfile << shapes[i] << endl;
		}
		myfile.close();
	}
	else {
		cout << "Unable to open file: " << dataset_path + "/" + "test.txt" << endl;
	}
}

void faustReconstruct3d(const std::string& permuted_funcs_path,
	const std::string& permutations_path) {
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	std::string path = "C:/Users/Adrien/Documents/FAUST_shapes_off/tr_reg_070.off";
	if (!igl::readOFF(path, V, F))
	{
		cout << "failed to load mesh" << endl;
		exit(666);
	}
	permuteFunc(V, permuted_funcs_path, permutations_path);
}



void faust(float radius) {
	std::string faust_shapes_path = "C:/Users/Adrien/Documents/FAUST_shapes_off";
	std::string labels_path = "C:/Users/Adrien/Documents/Datasets/Faust/labels_original_x=10_y=14";
	//std::string faust_dataset_path = "C:/Users/Adrien/Documents/Datasets/Faust";

	// train test
	/*ofstream myFile;
	myFile.open(faust_dataset_path + "/" + "FAUST_train.txt");
	for (int i = 0; i < 80; i++) {
		myFile << "tr_reg_0" + std::to_string(i) << endl;
	}
	myFile.close();

	myFile.open(faust_dataset_path + "/" + "FAUST_test.txt");
	for (int i = 0; i < 20; i++) {
		myFile << "tr_reg_0" + std::to_string(i+80) << endl;
	}
	myFile.close();*/
	// signals

	// faustXYZ(faust_shapes_path, "C:/Users/Adrien/Documents/Datasets/Faust/signals");

	// labels

	//faust_voxel_labels(faust_shapes_path, labels_path);
	//faust_vertices_labels(faust_shapes_path, labels_path);

	// filter mapper

	/*computeFilterMaper(radius,
		16,
		3,
		faust_shapes_path,
		faust_dataset_path);//*/
}

void non_permuted_labels(int nv, const std::string& shapes_path, const std::string& labels_path) {
	std::vector< std::string > names;
	getFilesList_(shapes_path, ".off", names);
	std::vector<int> labels(nv);
	for (int i = 0; i < nv; i++) {
		labels[i] = i;
	}
	for (int i = 0; i < names.size(); i++) {
		save_vector_<int>(labels_path + "/" + names[i] + ".txt", labels);
	}
}

void non_permuted_target_signal(const std::string& shapes_path, const std::string& target_path) {
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	std::string path = "C:/Users/Adrien/Documents/FAUST_shapes_off/tr_reg_070.off";
	if (!igl::readOFF(path, V, F))
	{
		cout << "failed to load mesh" << endl;
		exit(666);
	}
	std::vector< std::string > names;
	getFilesList_(shapes_path, ".off", names);
	for (int i = 0; i < names.size(); i++) {
		save_matrix_(target_path + "/" + names[i] + ".txt", V);
	}
}

void faust(const std::string& faust_path,
	const std::string& dataset_path, 
	double ratio_, double radius_, int nrings_, int ndirs_, int n) {

	double inv_decay_ratio = ratio_;
	std::vector<double> ratios(n);
	ratios[0] = 1.0;
	std::vector<int> nrings(n);
	std::vector<int> ndirs(n);
	for (int i = 0; i < n; i++) {
		nrings[i] = nrings_;
		ndirs[i] = ndirs_;
	}
	for (int i = 0; i < n-1; i++) {
		ratios[i + 1] = ratios[i] / inv_decay_ratio;
	}
	for (int i = 0; i < n; i++) {
		cout << ratios[i] << endl;
	}
	
	double r = radius_;
	

	std::vector<double> radius(n);
	for (int i = 0; i < n; i++) {
		radius[i] = r;
		r *= sqrt(inv_decay_ratio);
	}

	computeFilterMaper_(ratios, radius, nrings, ndirs, faust_path, "",
		dataset_path);
}

void remeshed_faust(const std::string& shapes_path, const std::string& faust_path) {
	std::vector< std::string > names;
	std::vector< int > labels;
	getFilesList_(shapes_path, ".off", names);
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	for (int i = 0; i < names.size(); i++) {
		// load mesh
		if (!igl::readOFF(shapes_path + "/" + names[i] + ".off", V, F)) {
			cout << "unable to open " << shapes_path + names[i] + ".off" << endl;
			system("pause");
		}
		// normalize
		normalize_shape(V);
		// save mesh
		if (!igl::writeOFF(faust_path + "/meshes/" + names[i] + ".off", V, F)) {
			cout << "unable to write " << faust_path + "/meshes/" + names[i] + ".off" << endl;
			system("pause");
		}
		load_vector_(shapes_path + "/corres/" + names[i] + ".vts", labels);
		for (int j = 0; j < labels.size(); j++) {
			labels[i] - 1;
		}
		save_vector_(faust_path + "/labels/" + names[i] + ".txt", labels);
	}
}

void remesh_faust(const std::string& shapes_path, 
	const std::string& labels_path,
	const std::string& tar_shapes_path) {
	Eigen::MatrixXd V0;
	Eigen::MatrixXi F0;
	Eigen::MatrixXd U0;
	Eigen::MatrixXi G0;
	Eigen::VectorXi I0;
	Eigen::VectorXi J0;
	if (!igl::readOFF(shapes_path + "/tr_reg_000.off", V0, F0)) {
		cout << "unable to open " << shapes_path + "/tr_reg_000.off" << endl;
		system("pause");
	}

	igl::decimate(V0, F0, (size_t)((F0.rows()*5000.0) / 6890.0), U0, G0, J0, I0);
	std::vector<int> idx;
	std::vector<int> idx_tar(5000);
	find_nn_correspondances(U0, G0, V0, F0, idx);

	std::vector< std::string > names;
	std::vector< int > labels;
	getFilesList_(shapes_path, ".off", names);
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	Eigen::MatrixXd U;
	Eigen::MatrixXi G;
	Eigen::VectorXi I;
	Eigen::VectorXi J;
	for (int i = 0; i < names.size(); i++) {
		// load mesh
		if (!igl::readOFF(shapes_path + "/" + names[i] + ".off", V, F)) {
			cout << "unable to open " << shapes_path + "/" + names[i] + ".off" << endl;
			system("pause");
		}

		igl::decimate(V, F, (size_t)((F.rows()*5000.0)/6890.0), U, G, J, I);
		// cout << names[i] << " nv: " << U.rows() << endl;

		for (int j = 0; j < idx_tar.size(); j++) {
			idx_tar[j] = idx[I(j)];
		}

		save_vector_(labels_path + "/" + names[i] + ".txt", idx_tar);
		normalize_shape(U);
		cout << names[i] << " nv= " << U.rows() << endl;
		// save mesh
		if (!igl::writeOFF(tar_shapes_path + "/" + names[i] + ".off", U, G)) {
			cout << "unable to save " << shapes_path + "/" + names[i] + ".off" << endl;
			system("pause");
		}

		/*find_nn_correspondances(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
			const Eigen::MatrixXd& NV, const Eigen::MatrixXi& NF,
			std::vector<int>& idx)*/

		
		// normalize
		//normalize_shape(V);
		/*// save mesh
		if (!igl::writeOFF(faust_path + "/meshes/" + names[i] + ".off", V, F)) {
			cout << "unable to write " << faust_path + "/meshes/" + names[i] + ".off" << endl;
			system("pause");
		}
		load_vector_(shapes_path + "/corres/" + names[i] + ".vts", labels);
		for (int j = 0; j < labels.size(); j++) {
			labels[i] - 1;
		}
		save_vector_(faust_path + "/labels/" + names[i] + ".txt", labels);*/
	}
}

#endif