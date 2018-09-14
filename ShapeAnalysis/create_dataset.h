#pragma once
#ifndef CREATE_DATASET_H
#define CREATE_DATASET_H

#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <patch_op.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "utils.h"
#include "pooling.h"
#include "patch_conv_op.h"

void null_signal(const std::string& shapes_path, const std::string& signal_path, int nv) {
	std::vector< std::string > names;
	getFilesList_(shapes_path, ".off", names);
	std::vector<float> null(nv);
	for (int i = 0; i < nv; i++) {
		null[i] = 0.0;
	}
	
	for (int i = 0; i < names.size(); i++) {
		// permute labels 
		save_vector_<float>(signal_path + "/" + names[i] + ".txt", null);
	}
}


void computeConvOperatorNN(double radius,
	int nrings,
	int ndirs,
	const std::string& shape,
	const std::string& shapes_path,
	const std::string& dataset_path) {
	PatchOperator* mapper = NULL;
	std::string name;
	
	mapper = new PatchOperator(shapes_path + "/" + shape + ".off");
	mapper->compute(ndirs, nrings, radius, true);
	name = shape + "_rad=" + std::to_string(radius) +
		"_nrings=" + std::to_string(nrings) + "_ndirs=" + std::to_string(ndirs) + ".txt";
	mapper->saveConnectivity(dataset_path + "/" + "connectivity/" + name);
	mapper->saveTransport(dataset_path + "/" + "transport/" + name);
		//mapper->saveCoord3d(dataset_path + "/" + "coords3d/" + name);
	mapper->saveLocalFrame3d(dataset_path + "/" + "local_frames/" + shape + ".txt");
	mapper->saveBasePoint(dataset_path + "/" + "basepoints/" + shape + ".txt");

	delete mapper;
	mapper = NULL;	
}

void computeConvOperator(double radius,
	int nrings,
	int ndirs,
	const std::string& shape,
	const std::string& shapes_path,
	const std::string& dataset_path) {
	PatchConvOperator* mapper = NULL;
	std::string name;

	mapper = new PatchConvOperator(shapes_path + "/" + shape + ".off");
	mapper->compute(ndirs, nrings, radius);
	name = shape + "_rad=" + std::to_string(radius) +
		"_nrings=" + std::to_string(nrings) + "_ndirs=" + std::to_string(ndirs) + ".txt";
	std::string path_c = dataset_path + "/" + "bin_contributors/" + name;
	std::string path_b = dataset_path + "/" + "contributors_weights/" + name;
	std::string path_t = dataset_path + "/" + "transported_angle/" + name;
	mapper->savePatchOp(path_c, path_b, path_t);

	delete mapper;
	mapper = NULL;
}


bool computeFilterMaper(double radius, 
						int nrings,
						int ndirs,
						const std::string& shapes_path, 
						const std::string& dataset_path) {
	std::vector< std::string > shapes;
	getFilesList_(shapes_path, ".off", shapes);

	for (int i = 0; i < shapes.size(); i++) {
		computeConvOperator(radius,
			nrings,
			ndirs,
			shapes[i],
			shapes_path,
			dataset_path);
	}
}

bool computeFilterMaper_(double radius,
	int nrings,
	int ndirs,
	const std::string& shape_path,
	const std::string& shape_name,
	const std::string& dataset_path) {
	std::vector< std::string > shapes(1);
	shapes[0] = shape_name;
	//getFilesList_(shapes_path, ".off", shapes);

	PatchOperator* mapper = NULL;
	std::string name;

	for (int i = 0; i < shapes.size(); i++) {
		computeConvOperator(radius,
			nrings,
			ndirs,
			shapes[i],
			shape_path,
			dataset_path);
	}
	return true;
}

void saveFilterMapperNN(
	double radius,
	int nrings,
	int ndirs,
	const std::string& shape_name,
	const std::string& dataset_path,
	const PatchOperator& mapper) {
	std::string name;

	name = shape_name + "_rad=" + std::to_string(radius) +
		"_nrings=" + std::to_string(nrings) + "_ndirs=" + std::to_string(ndirs) + ".txt";
	mapper.saveConnectivity(dataset_path + "/" + "connectivity/" + name);
	mapper.saveTransport(dataset_path + "/" + "transport/" + name);
	//mapper->saveCoord3d(dataset_path + "/" + "coords3d/" + name);
	mapper.saveLocalFrame3d(dataset_path + "/" + "local_frames/" + shape_name + ".txt");
	mapper.saveBasePoint(dataset_path + "/" + "basepoints/" + shape_name + ".txt");
}

void saveFilterMapper(
	double radius,
	int nrings,
	int ndirs,
	const std::string& shape_name,
	const std::string& dataset_path,
	const PatchConvOperator& mapper) {
	std::string name;

	name = shape_name + "_rad=" + std::to_string(radius) +
		"_nrings=" + std::to_string(nrings) + "_ndirs=" + std::to_string(ndirs) + ".txt";

	std::string path_c = dataset_path + "/" + "bin_contributors/" + name;
	std::string path_b = dataset_path + "/" + "contributors_weights/" + name;
	std::string path_t = dataset_path + "/" + "transported_angles/" + name;
	if (!mapper.savePatchOp(path_c, path_b, path_t)) {
		//exit(666);
	}
}

void savePoolingOpNN(
	int j,
	const std::vector<double>& ratios,
	const std::vector<int>& ndirs,
	const std::string& shape_name,
	const std::string& dataset_path,
	const std::vector<int>& parent_vertex,
	const std::vector<int>& angular_shift) {
	
	int nv = parent_vertex.size();
	std::string file_name = shape_name + "_ratio_" + std::to_string(ratios[j]) +
		"_to_" + std::to_string(ratios[j + 1]);
	save_vector_(dataset_path + "/parent_vertices/" + file_name + "_nv=" +
		std::to_string(nv) + ".txt", parent_vertex);
	save_vector_(dataset_path + "/angular_shifts/" + file_name + "_nv=" + 
		std::to_string(nv) + "_ndirs=" + std::to_string(ndirs[j + 1]) + ".txt", angular_shift);
}

void savePoolingOp(
	int j,
	const std::vector<double>& ratios,
	const std::vector<int>& ndirs,
	const std::string& shape_name,
	const std::string& dataset_path,
	const std::vector<int>& parent_vertex,
	const std::vector<double>& angular_shift) {

	int nv = parent_vertex.size();
	std::string file_name = shape_name + "_ratio_" + std::to_string(ratios[j]) +
		"_to_" + std::to_string(ratios[j + 1]);
	save_vector_(dataset_path + "/parent_vertices/" + file_name + "_nv=" +
		std::to_string(nv) + ".txt", parent_vertex);
	save_vector_(dataset_path + "/angular_shifts/" + file_name + "_nv=" +
		std::to_string(nv) + ".txt", angular_shift);
}


bool computeFilterMaper_(
	const std::vector<double>& ratios_,
	const std::vector<double>& radius,
	const std::vector<int>& nrings,
	const std::vector<int>& ndirs,
	const std::string& shape_path,
	const std::string& shape_name,
	const std::string& dataset_path) {


	std::vector< std::string > shapes;
	if (shape_name == "") {
		getFilesList_(shape_path, ".off", shapes);
	}
	else {
		shapes.resize(1);
		shapes[0] = shape_name;
	}

	PatchConvOperator* mapper1 = NULL;
	PatchConvOperator* mapper2 = NULL;
	std::string name;

	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	Eigen::MatrixXd U;
	Eigen::MatrixXi G;
	Eigen::VectorXi J;
	Eigen::VectorXi I;

	std::vector<double> angular_shift_;
	std::vector<int> parent_vertex;
	
	double ratio;

	std::vector<double> ratios(radius.size());
	ratios[0] = 1.0;
	if (ratios_.size() != ratios.size()) {
		//cout << "uuuuuuuu" << endl;
		//cout << ratios.size() << " " << ratios_.size() << endl;
		for (int i = 1; i < radius.size(); i++) {
			ratios[i] = ratios_[i - 1];
		}
	}
	else {
		for (int i = 0; i < radius.size(); i++) {
			ratios[i] = ratios_[i];
		}
	}

	for (int i = 0; i < shapes.size(); i++) {

		display_progress(float(i) / float(shapes.size()));

		mapper1 = new  PatchConvOperator(shape_path + "/" + shapes[i] + ".off");
		mapper1->compute(ndirs[0], nrings[0], radius[0]);
		if (!igl::readOFF(shape_path + "/" + shapes[i] + ".off", V, F))
		{
			cout << "failed to load mesh" << endl;
			return false;
		}
		/*U = V;
		G = F;
		I.resize(V.rows());
		I.setZero();*/
		saveFilterMapper(radius[0], nrings[0], ndirs[0],
			shapes[i] + "_ratio=" + std::to_string(ratios[0])
			+ "_nv=" + std::to_string(V.rows()),
			dataset_path,
			*mapper1);
		
		for (int j = 0; j < ratios.size()-1; j++) {
			ratio = ratios[j+1] / ratios[j];
			
			igl::decimate(V, F, (size_t)(F.rows()*ratio), U, G, J, I);
			
			mapper2 = new PatchConvOperator(U, G);
			mapper2->compute(ndirs[j+1], nrings[j+1], radius[j+1], true);
			saveFilterMapper(radius[j+1], nrings[j+1], ndirs[j+1],
				shapes[i] + "_ratio=" + std::to_string(ratios[j+1])
				+ "_nv=" + std::to_string(U.rows()),
				dataset_path,
				*mapper2);

			parent_vertex.resize(I.rows());
			for (int k = 0; k < parent_vertex.size(); k++) {
				parent_vertex[k] = I(k);
			}
			//index_shift(mapper1->Gpc(), mapper2->Gpc(), I, ndirs[j], angular_shift);
			angular_shift(mapper1->Gpc(), mapper2->Gpc(), I, angular_shift_);
			savePoolingOp(j, ratios, ndirs, shapes[i], dataset_path, parent_vertex, angular_shift_);
			std::swap(mapper1, mapper2);
			delete mapper2;
			mapper2 = NULL;
			V = U;
			F = G;
		}
		delete mapper1;
		mapper1 = NULL;
	}
	return true;
}

void PrepareDataset(const std::string& shapes_path,
	const std::string& dataset_path,
	double ratio_, double radius_, int nrings_, int ndirs_, int n = 3) {

	double inv_decay_ratio = ratio_;
	std::vector<double> ratios(n);
	ratios[0] = 1.0;
	std::vector<int> nrings(n);
	std::vector<int> ndirs(n);
	for (int i = 0; i < n; i++) {
		nrings[i] = nrings_;
		ndirs[i] = ndirs_;
	}
	for (int i = 0; i < n - 1; i++) {
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

	computeFilterMaper_(ratios, radius, nrings, ndirs, shapes_path, "",
		dataset_path);
}

void test_validity(const std::string& path) {
	std::vector<std::string> shapes;
	getFilesList_(path, ".off", shapes);
	geodesic::Mesh mesh;
	for (int i = 0; i < shapes.size(); i++) {
		cout << "path: " << path << endl;
		cout << "shape:" << shapes[i] << endl;
		loadMesh(path + "/" + shapes[i] + ".off", mesh);
	}
	
}

void check_numeric_files(const std::string& path) {
	double m = 0.0;
	std::vector<std::string> names;
	getFilesList_(path, ".txt", names);
	ifstream file; 
	string line;

	unsigned int curLine = 0;
	bool found = false;
	std::vector<double> v;

	for (int i = 0; i < names.size(); i++) {
		found = false;
		file.open(path + "/" + names[i] + ".txt", ios::in);
			while (std::getline(file, line) && !found) {
				curLine++;
				if (line.find("nan", 0) != string::npos) {
					cout << names[i] << endl;
					found = true;
					//cout << "found: " << search << "line: " << curLine << endl;
				}
			}
		file.close();
		load_vector_(path + "/" + names[i] + ".txt", v);
		m = 0.0;
		for (int j = 0; j < v.size(); j++) {
			m = std::max(m, fabs(v[j]));
		}
		cout << "m = " << m << endl;
	}
}

#endif
