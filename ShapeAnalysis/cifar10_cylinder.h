#pragma once
#ifndef CIFAR10_CYLINDER_H
#define CIFAR10_CYLINDER_H

#include <fstream>
#include <iostream>
#include <patch_op.h>
//#include <mnist/mnist_reader_less.hpp>
#include <cifar/cifar10_reader.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "utils.h"
#include "create_dataset.h"
#include "img2cylinder.h"
#include "visualize.h"
#include "img.h"

using namespace std;

void print_labels(int nb, bool test=true) {
	std::string cifar_path = "C:/Users/adrien/Documents/cpp libs/cifar-10-master/cifar-10/cifar-10-batches-bin";
	auto dataset = cifar::read_dataset_<std::vector, std::vector, uint8_t, uint8_t>(cifar_path);
	if (test) {
		for (int i = 0; i < nb;  i++) {
			cout << std::to_string(i) << " " << std::to_string(int(dataset.test_labels[i])) << endl;
		}
	}
	else {
		for (int i = 0; i < nb; i++) {
			cout << std::to_string(i) << " " << std::to_string(int(dataset.training_labels[i])) << endl;
		}
	}
}


bool computeCifar10CylinderDataset(const std::string& cifar_path,
	const std::string& cylinder_path,
	const std::string& cylinder_name,
	const std::string& train_path,
	const std::string& test_path,
	const std::string& labels_path,
	int nb_train = -1,
	int nb_test = -1,
	int nb_train_splits = 1,
	int nb_test_splits = 1,
	double margin = 0.0) {

	auto dataset = cifar::read_dataset_<std::vector, std::vector, uint8_t, uint8_t>(cifar_path);

	if (nb_test <= 0 || nb_test > dataset.test_labels.size()) {
		nb_test = dataset.test_labels.size();
	}

	if (nb_train <= 0 || nb_train > dataset.training_labels.size()) {
		nb_train = dataset.training_labels.size();
	}
	ofstream myfile;


	// save signals
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	if (!igl::readOFF(cylinder_path + "/" + cylinder_name + ".off", V, F))
	{
		cout << "failed to load cylinder mesh" << endl;
		return false;
	}
	//centerObject_(V);
	//Eigen::MatrixXd signal(V.rows(), 1);
	std::vector<uint8_t> signal(V.rows() * 3);

	std::ostringstream name;
	int percent = 0;

	int nb_train_batch = nb_train / nb_train_splits;
	std::vector<uint8_t> Signal(nb_train_batch*V.rows() * 3);

	// save labels
	dataset.training_labels;

	for (int l = 0; l < nb_train_splits; l++) {
		myfile.open(labels_path + "/" + "training_labels_" + std::to_string(nb_train)
			+ "_" + std::to_string(l) + ".txt");
		for (int i = 0; i < nb_train_batch; i++) {
			myfile << (int)(dataset.training_labels[i + nb_train_splits * l]) << endl;
		}
		myfile.close();
	}

	for (int l = 0; l < nb_train_splits; l++) {
		for (int i = 0; i < nb_train_batch; i++) {
			if (100.0*i*(l + 1) / nb_train > nb_train_splits * percent) {
				percent++;
				std::cout << percent << " percent of training images converted" << endl;
			}
			//img2sphere(signal, dataset.training_images[i], V, F, 2.0, -1, -1);
			rgbImg2cylinder(signal, dataset.training_images[i + nb_train_splits * l], V, F, margin, -1, -1);
			for (int k = 0; k < 3; k++) {
				for (int j = 0; j < V.rows(); j++) {
					Signal[3 * (V.rows()*i + j) + k] = signal[3 * j + k];
				}
			}
		}
		myfile.open(train_path + "/" + cylinder_name +
			"_training_" + std::to_string(nb_train) + "_" + std::to_string(l) + ".txt");
		percent = 0;
		for (int i = 0; i < Signal.size(); i++) {
			if (100.0*i / Signal.size() > percent) {
				percent++;
				std::cout << percent << " percent training saved" << endl;
			}
			myfile << (float)(Signal[i]) << endl;
		}
		myfile.close();
	}//*/

	std::cout << "training signal saved" << endl;



	percent = 0;
	int nb_test_batch = nb_test / nb_test_splits;
	Signal.resize(3 * V.rows()*nb_test_batch);

	for (int l = 0; l < nb_test_splits; l++) {
		myfile.open(labels_path + "/" + "test_labels_" + std::to_string(nb_test) +
			"_" + std::to_string(l) + ".txt");
		for (int i = 0; i < nb_test_batch; i++) {
			myfile << (int)(dataset.test_labels[i + l* nb_test_splits]) << endl;
		}
		myfile.close();
	}

	for (int l = 0; l < nb_test_splits; l++) {

		for (int i = 0; i < nb_test_batch; i++) {
			if (100.0*i*l / nb_test > nb_test_splits * percent) {
				percent++;
				std::cout << percent << " percent of test images converted" << endl;
			}

			rgbImg2cylinder(signal, dataset.test_images[i + nb_test_splits*l], V, F, margin, -1, -1);
			for (int k = 0; k < 3; k++) {
				for (int j = 0; j < V.rows(); j++) {
					Signal[3 * (V.rows()*i + j) + k] = signal[3 * j + k];
				}
			}
		}
		myfile.open(test_path + "/" + cylinder_name +
			"_test_" + std::to_string(nb_test) + "_" + std::to_string(l) + ".txt");
		percent = 0;
		for (int i = 0; i < Signal.size(); i++) {
			if (100.0*i / Signal.size() > percent) {
				percent++;
				std::cout << percent << " percent test saved" << endl;
			}
			myfile << (float)(Signal[i]) << endl;
		}
		myfile.close();
	}//*/
	std::cout << "test signal saved" << endl;

	return true;
}

bool computeCifar10CylinderDataset(
	const std::string& cylinder_path,
	const std::string& cylinder_name,
	const std::string& UV_path,
	double margin = 0.0) {
	Eigen::MatrixXd UV;
	// save signals
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	if (!igl::readOFF(cylinder_path + "/" + cylinder_name + ".off", V, F))
	{
		cout << "failed to load cylinder mesh" << endl;
		return false;
	}

	mapImage2cylinder(32, 32,
		V, F,
		UV,
		margin);

	save_matrix_(UV_path + "/" + cylinder_name + "_32x32.txt", UV);
}

bool test_rgb_to_cylinder(const std::string& cylinder_path, const std::string& cylinder_name) {
	auto dataset = cifar::read_dataset_<std::vector, std::vector, uint8_t, uint8_t>("C:/Users/adrien/Documents/cpp libs/cifar-10-master/cifar-10/cifar-10-batches-bin");
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	if (!igl::readOFF(cylinder_path + "/" + cylinder_name + ".off", V, F))
	{
		std::cout << "failed to load cylinder mesh" << endl;
		return false;
	}
	//centerObject(V);
	//Eigen::MatrixXd signal(V.rows(), 1);
	std::vector<uint8_t> signal(3 * V.rows());
	rgbImg2cylinder(signal, dataset.training_images[1565], V, F, 0.000, -1, -1);
	visualizeRGB(V, F, signal);
	return true;
}

void cifar10cylinder(const std::string& dataset,
	const std::string& shape_path, const std::string& shape_name,
	const std::string& signal_path, const std::string& labels_path,
	int nb_train, int nb_test,
	int nb_train_splits, int nb_test_splits,
	double margin) {

	/*
	permuteMesh("C:/Users/adrien/Documents/shapes/sphere/sphere_2002.off",
	"C:/Users/adrien/Documents/shapes/sphere/sphere_2002_shuf.off"); // */

	/*computeCifar10SpheresDataset("C:/Users/Adrien/Documents/cpp_libs/cifar-10-master/cifar-10/cifar-10-batches-bin",
	"C:/Users/Adrien/Documents/shapes/sphere", "sphere_3002",
	"C:/Users/Adrien/Documents/Datasets/cifar10_spheres/signals",
	"C:/Users/Adrien/Documents/Datasets/cifar10_spheres/signals",
	"C:/Users/Adrien/Documents/Datasets/cifar10_spheres/labels",
	nb_train,
	nb_test,
	2.0); */

	/*computeCifar10CylinderDataset(dataset,
	shape_path, shape_name,
	signal_path,
	signal_path,
	labels_path,
	nb_train,
	nb_test,
	nb_train_splits,
	nb_test_splits,
	margin);//*/


	/*int nrings = 2;
	int ndirs = 16;
	double a = 1.0;
	double diameter = 3*M_PI / 32;
	double radius = diameter / 2.0;

	std::string train_sphere = "sphere_3002";
	//std::string test_sphere = "sphere_2002_shuf";

	std::string spheres_path = "C:/Users/adrien/Documents/shapes/sphere";
	for (int j = 0; j < 3; j++) {
	for (int i = 0; i < 3; i++) {
	computeFilterMaper_(a*radius, nrings, ndirs, spheres_path, train_sphere,
	"C:/Users/adrien/Documents/Datasets/cifar10_spheres");
	radius *= 2.;
	}
	radius = diameter / 2.0;
	a += 0.1;
	}//*/

	computeCifar10CylinderDataset(
		shape_path, shape_name,
		"C:/Users/adrien/Documents/Datasets/cifar10_cylinder/uv_coordinates",
		margin);

	
	double inv_decay_ratio = 2.0;
	std::vector<double> ratios(3);
	ratios[0] = 1.0;
	std::vector<int> nrings(3);
	std::vector<int> ndirs(3);
	for (int i = 0; i < 3; i++) {
		nrings[i] = 2;
		ndirs[i] = 16;
	}
	for (int i = 0; i < 2; i++) {
		ratios[i + 1] = ratios[i] / inv_decay_ratio;
	}
	for (int i = 0; i < 3; i++) {
		cout << ratios[i] << endl;
	}
	int nv = 2000;
	double mesh_ratio_bnd = 2.0;
	double average_edge_length = 1. / sqrt(1725);
	double r_min = mesh_ratio_bnd * average_edge_length;
	double r = 4.0 / 64.0;
	if (r < r_min) {
		cout << "r < r_min !!!!" << endl;
		cout << r << " < " << r_min << endl;
		system("pause");
		r = r_min;
	}

	std::vector<double> radius(3);
	for (int i = 0; i < 3; i++) {
		radius[i] = r;
		r *= sqrt(inv_decay_ratio);
	}

	computeFilterMaper_(ratios, radius, nrings, ndirs, shape_path, shape_name,
		"C:/Users/adrien/Documents/Datasets/cifar10_cylinder");//*/

}

/*bool RGB_UV_cylinder(const std::string& cylinder_path, const std::string& cylinder,
	const std::vector<int> img_idx, const std::string& path, const std::string& cifar_path) {
	auto dataset = cifar::read_dataset_<std::vector, std::vector, uint8_t, uint8_t>(cifar_path);
	std::string name;
	Eigen::MatrixXd UV;

	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	if (!igl::readOFF(cylinder_path + "/" + cylinder + ".off", V, F))
	{
		std::cout << "failed to load cylinder mesh" << endl;
		return false;
	}
	std::vector<uint8_t> f(V.rows() * 3);

	for (int i = 0; i < img_idx.size(); i++) {
		name = cylinder + "_" + std::to_string(img_idx[i]);
		rgbImg2cylinder(f, UV, dataset.training_images[img_idx[i]], V, F, 0.025);
		cout << "uuuuuuu" << endl;
		write_image_ppm(dataset.training_images[img_idx[i]], 32, 32, path, name);
		cout << "saved" << endl;
	}
	save_matrix_(path + "/" + sphere + "_uv.txt", UV);
	return true;
}*/

#endif // !CIFAR10_CYLINDER_H
