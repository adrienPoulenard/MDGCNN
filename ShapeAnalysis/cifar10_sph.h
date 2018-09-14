#pragma once
#ifndef CIFAR10_SPH_H
#define CIFAR10_SPH_H

#include <fstream>
#include <iostream>
#include "img2sphere.h"
#include <patch_op.h>
//#include <mnist/mnist_reader_less.hpp>
#include <cifar/cifar10_reader.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "utils.h"
#include "create_dataset.h"
#include "img2sphere.h"
#include "visualize.h"
#include "img.h"

using namespace std;

bool computeCifar10SpheresDataset(const std::string& cifar_path,
	const std::string& sphere_path,
	const std::string& sphere_name,
	const std::string& train_path,
	const std::string& test_path,
	const std::string& labels_path,
	int nb_train = -1,
	int nb_test = -1,
	int nb_train_splits = 1,
	int nb_test_splits = 1,
	double ratio = 1.0) {

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
	if (!igl::readOFF(sphere_path + "/" + sphere_name + ".off", V, F))
	{
		cout << "failed to load sphere mesh" << endl;
		return false;
	}
	centerObject_(V);
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
			rgbImg2sphere(signal, dataset.training_images[i + nb_train_splits * l], V, F, 2.0, -1, -1);
			for (int k = 0; k < 3; k++) {
				for (int j = 0; j < V.rows(); j++) {
					Signal[3 * (V.rows()*i + j) + k] = signal[3 * j + k];
				}
			}
		}
		myfile.open(train_path + "/" + sphere_name + "_ratio=" + std::to_string(ratio) +
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

			rgbImg2sphere(signal, dataset.test_images[i + nb_test_splits*l], V, F, 2.0, -1, -1);
			for (int k = 0; k < 3; k++) {
				for (int j = 0; j < V.rows(); j++) {
					Signal[3 * (V.rows()*i + j) + k] = signal[3 * j + k];
				}
			}
		}
		myfile.open(test_path + "/" + sphere_name + "_ratio=" + std::to_string(ratio) +
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

bool test_rgb_to_sh(const std::string& sphere_path, const std::string& sphere_name) {
	auto dataset = cifar::read_dataset_<std::vector, std::vector, uint8_t, uint8_t>("C:/Users/Adrien/Documents/cpp_libs/cifar-10-master/cifar-10/cifar-10-batches-bin");
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	if (!igl::readOFF(sphere_path + "/" + sphere_name + ".off", V, F))
	{
		std::cout << "failed to load sphere mesh" << endl;
		return false;
	}
	centerObject(V);
	//Eigen::MatrixXd signal(V.rows(), 1);
	std::vector<uint8_t> signal(3 * V.rows());
	rgbImg2sphere(signal, dataset.training_images[1565], V, F, 2.0, -1, -1);
	visualizeRGB(V, F, signal);
	return true;
}

void cifar10sphere(const std::string& dataset, 
	const std::string& shape_path, const std::string& shape_name,
	const std::string& signal_path, const std::string& labels_path,
	int nb_train, int nb_test,
	int nb_train_splits, int nb_test_splits,
	double ratio) {

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

	/*computeCifar10SpheresDataset(dataset,
		shape_path, shape_name,
		signal_path,
		signal_path,
		labels_path,
		nb_train,
		nb_test,
		nb_train_splits,
		nb_test_splits,
		ratio);//*/

	
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

	/*double inv_decay_ratio = 2.0;
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
	int nv = 3002;
	double mesh_ratio_bnd = 2.0;
	double average_edge_length = sqrt(4 * M_PI) / sqrt(nv);
	double r_min = mesh_ratio_bnd * average_edge_length;
	double r = M_PI / 20.0;
	if (r < r_min) {
		
		cout << "r < r_min !!!!" << endl;
		cout << r << " < " << r_min << endl;
		system("pause");
		r = r_min;
	}
	
	std::vector<double> radius(3);
	for (int i = 0; i < 3; i++) {
		radius[i] = r * nrings[i];
		r *= sqrt(inv_decay_ratio);
	}*/

	std::vector<double> ratios(2);
	ratios[0] = 1.0;
	ratios[1] = 0.5;
	std::vector<double> radius(2);
	radius[0] = M_PI / 5.;
	radius[1] = M_PI / 5.;
	std::vector<int> nrings(2);
	nrings[0] = 8;
	nrings[1] = 8;
	std::vector<int> ndirs(2);
	ndirs[0] = 16;
	ndirs[1] = 16;
	computeFilterMaper_(ratios, radius, nrings, ndirs, shape_path, shape_name, 
		"C:/Users/Adrien/Documents/Datasets/cifar10_spheres");

}

bool RGB_UV_sphere(const std::string& sphere_path, const std::string& sphere,
	const std::vector<int> img_idx, const std::string& path, const std::string& cifar_path) {
	auto dataset = cifar::read_dataset_<std::vector, std::vector, uint8_t, uint8_t>(cifar_path);
	std::string name;
	Eigen::MatrixXd UV;
	
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	if (!igl::readOFF(sphere_path + "/" + sphere + ".off", V, F))
	{
		std::cout << "failed to load sphere mesh" << endl;
		return false;
	}
	std::vector<uint8_t> f(V.rows()*3);

	for (int i = 0; i < img_idx.size(); i++) {
		name = sphere + "_" + std::to_string(img_idx[i]);
		rgbImg2sphere(f, UV, dataset.training_images[img_idx[i]], V, F, 2.0);
		cout << "uuuuuuu" << endl;
		write_image_ppm(dataset.training_images[img_idx[i]], 32, 32, path, name);
		cout << "saved" << endl;
	}
	save_matrix_(path + "/" + sphere + "_uv.txt", UV);
	return true;
}




#endif