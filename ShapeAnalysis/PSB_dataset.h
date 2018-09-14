#pragma once
#ifndef PSB_dataset_H
#define PSB_dataset_H

#include "utils.h"
#include <igl/writeOFF.h>
#include <igl/readOFF.h>
#include <igl/decimate.h>
#include <igl/false_barycentric_subdivision.h>
#include "create_dataset.h"
#include "utils.h"
#include "clean_mesh.h"




bool reduce_mesh_and_labels(int max_m, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, 
	Eigen::MatrixXd& U, Eigen::MatrixXi& G, const std::vector<int>& labels,
	std::vector<int>& new_labels) {
	Eigen::VectorXi I;
	Eigen::VectorXi J;
	bool res = igl::decimate(V, F, max_m, U, G, J, I);
	new_labels.resize(U.rows());
	for (int i = 0; i < U.rows(); i++) {
		new_labels[i] = labels[I(i)];
	}
}

void refine_mesh_and_labels(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
	Eigen::MatrixXd& U, Eigen::MatrixXi& G, const std::vector<int>& labels,
	std::vector<int>& new_labels) {
	igl::false_barycentric_subdivision(V, F, U, G);
	int nv = V.rows();
	new_labels.resize(U.rows());
	for (int i = 0; i < F.rows(); i++) {
		for (int j = 0; j < 3; j++) {
			for (int k = 0; k < 3; k++) {
				if (G(3 * i + j, k) >= nv) {
					new_labels[G(3 * i + j, k)] = labels[F(i, 0)];
				}
				else {
					new_labels[G(3 * i + j, k)] = labels[G(3 * i + j, k)];
				}
			}
		}
	}
}

void remesh(int max_m, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
	Eigen::MatrixXd& U, Eigen::MatrixXi& G, const std::vector<int>& labels,
	std::vector<int>& new_labels) {

	Eigen::MatrixXd V_ = V;
	Eigen::MatrixXi F_ = F;
	std::vector<int> labels_(labels.size());
	
	int nv = V.rows();
	if (nv < max_m) {
		for (int i = 0; i < labels.size(); i++) {
			labels_[i] = labels[i];
		}
		std::swap(labels_, new_labels);
		while (nv < max_m) {
			std::swap(labels_, new_labels);
			refine_mesh_and_labels(V, F, U, G, labels_, new_labels);
			V_ = U;
			F_ = G;
			nv = U.rows();
		}

	}

	reduce_mesh_and_labels(max_m, V_, F_, U, G, labels, new_labels);

}

int min_nv() {

}
void prepare_psb(const std::string& src_path, const std::string& tar_path) {

	// mini psb

	/*category[0] = "Airplane";
	category[1] = "Ant";
	category[2] = "FourLeg";
	category[3] = "Hand";
	category[4] = "Octopus";
	category[5] = "Teddy";*/


	std::vector< std::string > category(19);
	category[0] = "Airplane";
	category[1] = "Ant";
	category[2] = "Armadillo";
	category[3] = "Bearing";
	category[4] = "Bird";
	category[5] = "Bust";
	category[6] = "Chair";
	category[7] = "Cup";
	category[8] = "Fish";
	category[9] = "FourLeg";
	category[10] = "Glasses";
	category[11] = "Hand";
	category[12] = "Human";
	category[13] = "Mech";
	category[14] = "Octopus";
	category[15] = "Plier";
	category[16] = "Table";
	category[17] = "Teddy";
	category[18] = "Vase";

	std::string path;

	std::vector< std::string > shapes_names;
	//getFilesList_(path, ".off", shapes_names);

	std::vector< std::string > labels_names;
	std::vector<int> labels;

	int nb_labels = 0;
	int nb_labels_tmp = 0;
	int nb_labels_cumul = 0;
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	std::vector<int> nb_labels_(category.size()+1);
	
	//myfile << "Writing this to a file.\n";
	//myfile.close();
	

	for (int i = 0; i < category.size(); i++) {
		display_progress(float(i) / float(category.size()));
		getFilesList_(src_path + "/" + category[i] + "/shapes", ".off", shapes_names);
		nb_labels = 0;
		for (int j = 0; j < shapes_names.size(); j++) {
			load_vector_(src_path + "/" + category[i] + "/labels/" + shapes_names[j] + ".txt", labels);
			nb_labels_tmp = *std::max_element(labels.begin(), labels.end());
			if (nb_labels < nb_labels_tmp) {
				nb_labels = nb_labels_tmp;
			}
		}
		nb_labels_[i] = nb_labels;
		for (int j = 0; j < shapes_names.size(); j++) {
			load_vector_(src_path + "/" + category[i] + "/labels/" + shapes_names[j] + ".txt", labels);
			for (int k = 0; k < labels.size(); k++) {
				labels[k] += nb_labels_cumul-1;
			}
			save_vector_(tar_path + "/labels/" + category[i] + "_" + shapes_names[j] + ".txt", labels);
			if (!igl::readOFF(src_path + "/" + category[i] + "/shapes/" + shapes_names[j] + ".off", V, F))
			{
				cout << "failed to load mesh" << endl;
			}
			normalize_shape(V);
			if (!igl::writeOFF(tar_path + "/shapes/" + category[i] + "_" + shapes_names[j] + ".off", V, F))
			{
				cout << "failed to save mesh" << endl;
			}
			save_matrix_(tar_path + "/descs/global_3d/" + category[i] + "_" + shapes_names[j] + ".txt", V);//*/
		}	
		nb_labels_cumul += nb_labels;
		nb_labels_[nb_labels_.size()-1] = nb_labels_cumul;
	}
	save_vector_(tar_path + "/nb_labels.txt", nb_labels_);//*/
	

	ofstream train;
	train.open(tar_path + "/train.txt");
	cout << train.is_open() << endl;
	ofstream test;
	test.open(tar_path + "/test.txt");

	for (int i = 0; i < category.size(); i++) {
		display_progress(float(i) / float(category.size()));
		shapes_names.clear();
		getFilesList_(src_path + "/" + category[i] + "/shapes", ".off", shapes_names);
		for (int j = 0; j < shapes_names.size()-4; j++) {
			cout << j << " " << shapes_names.size() << " " << category[i] << endl;
			train << category[i] + "_" + shapes_names[j] << endl;
		}
		for (int j = 0; j < 4; j++) {
			test << category[i] + "_" + shapes_names[j + shapes_names.size() - 4] << endl;
		}
	}

	train.close();
	test.close();

	//PrepareDataset(tar_path + "/shapes", tar_path, 4.0, 0.12, 2, 8, 2);
}

void singular_psb(const std::string& src_path, const std::string& tar_path) {
	std::vector< std::string > category(19);
	category[0] = "Airplane";
	category[1] = "Ant";
	category[2] = "Armadillo";
	category[3] = "Bearing";
	category[4] = "Bird";
	category[5] = "Bust";
	category[6] = "Chair";
	category[7] = "Cup";
	category[8] = "Fish";
	category[9] = "FourLeg";
	category[10] = "Glasses";
	category[11] = "Hand";
	category[12] = "Human";
	category[13] = "Mech";
	category[14] = "Octopus";
	category[15] = "Plier";
	category[16] = "Table";
	category[17] = "Teddy";
	category[18] = "Vase";

	int nb_singular_meshes = 0;
	std::string path;

	std::vector< std::string > shapes_names;
	//getFilesList_(path, ".off", shapes_names);
	std::vector<int> labels;
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	Eigen::Matrix<bool, -1, -1> B;
	for (int i = 0; i < category.size(); i++) {
		getFilesList_(src_path + "/" + category[i] + "/shapes", ".off", shapes_names);
		for (int j = 0; j < shapes_names.size(); j++) {
			igl::readOFF(src_path + "/" + category[i] + "/shapes/" + shapes_names[j] + ".off", V, F);
			if (!(igl::is_edge_manifold(F) && igl::is_vertex_manifold(F, B))) {
				std::cout << category[i] + "_" + shapes_names[j] << " is non manifold" << endl;
				// copy mesh
				igl::writeOFF(tar_path + "/" + category[i] + "/shapes/" + shapes_names[j] + ".off", V, F);
				// load labels
				load_vector_(src_path + "/" + category[i] + "/labels/" + shapes_names[j] + ".txt", labels);
				// save labels
				save_vector_(tar_path + "/" + category[i] + "/labels/" + shapes_names[j] + ".txt", labels);
				nb_singular_meshes++;
			}
		}
	}
	std::cout << nb_singular_meshes << "singular meshes" << endl;
}

void test_PSB_validity(const std::string& path) {
	std::vector< std::string > category(19);
	category[0] = "Airplane";
	category[1] = "Ant";
	category[2] = "Armadillo";
	category[3] = "Bearing";
	category[4] = "Bird";
	category[5] = "Bust";
	category[6] = "Chair";
	category[7] = "Cup";
	category[8] = "Fish";
	category[9] = "FourLeg";
	category[10] = "Glasses";
	category[11] = "Hand";
	category[12] = "Human";
	category[13] = "Mech";
	category[14] = "Octopus";
	category[15] = "Plier";
	category[16] = "Table";
	category[17] = "Teddy";
	category[18] = "Vase";

	for (int i = 0; i < category.size(); i++) {
		cout << "-------------------------------" << endl;
		cout << category[i] << endl;
		cout << "-------------------------------" << endl;
		test_validity(path + "/" + category[i] + "/shapes");
	}
}


void PSB_labels_conversion(const std::string& PSB1, const std::string& PSB2) {
	std::vector< std::string > category(19);
	category[0] = "Airplane";
	category[1] = "Ant";
	category[2] = "Armadillo";
	category[3] = "Bearing";
	category[4] = "Bird";
	category[5] = "Bust";
	category[6] = "Chair";
	category[7] = "Cup";
	category[8] = "Fish";
	category[9] = "FourLeg";
	category[10] = "Glasses";
	category[11] = "Hand";
	category[12] = "Human";
	category[13] = "Mech";
	category[14] = "Octopus";
	category[15] = "Plier";
	category[16] = "Table";
	category[17] = "Teddy";
	category[18] = "Vase";

	std::vector<std::string> shapes_names;
	for (int i = 0; i < category.size(); i++) {
		display_progress(float(i) / category.size());
		getFilesList_(PSB2 + "/" + category[i] + "/shapes", ".off", shapes_names);
		for (int j = 0; j < shapes_names.size(); j++) {
			find_nn_correspondances<int>(PSB1 + "/" + category[i] + "/shapes/" + shapes_names[j] + ".off", 
				PSB2 + "/" + category[i] + "/shapes/" + shapes_names[j] + ".off",
				PSB1 + "/" + category[i] + "/labels/" + shapes_names[j] + ".txt",
				PSB2 + "/" + category[i] + "/labels/" + shapes_names[j] + ".txt");
		}
	}
}

#endif