#pragma once
#ifndef HUMAN_SEG_SIG_2017_TORIC_COVER_H
#define HUMAN_SEG_SIG_2017_TORIC_COVER_H

#include <iostream>
#include <fstream>

//#include <igl/viewer/Viewer.h>
//#include <igl/parula.h>
//#include <igl/read_triangle_mesh.h>
#include <igl/readOFF.h>
#include <igl/writeOFF.h>
#include <igl/readOBJ.h>
#include <igl/readPLY.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "cube_segmentation.h"
#include "create_dataset.h"
#include "permute_dataset.h"
#include "utils.h"

using namespace std;

void SIG_17_shapes(const std::string& src_path, const std::string& tar_path) {
	std::vector<std::string> mit_folders(8);
	mit_folders[0] = "bouncing";
	mit_folders[1] = "crane";
	mit_folders[2] = "handstand";
	mit_folders[3] = "jumping";
	mit_folders[4] = "march1";
	mit_folders[5] = "march2";
	mit_folders[6] = "squat1";
	mit_folders[7] = "squat2";

	std::string faust_folder = "faust";
	std::string adobe_folder = "adobe";
	std::string scape_folder = "scape";

	std::string path;
	std::vector<std::string> names;
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	std::vector< int > labels;

	ofstream train_file(tar_path + "/train.txt");
	// MIT shapes
	
	for (int i = 0; i < 8; i++) {
		getFilesList_(src_path + "/meshes/train/mit/" + mit_folders[i], ".off", names);
		for (int j = 0; j < names.size(); j++) {
			// load labels
			//load_vector_(src_path + "/segs/train/mit/" + mit_folders[i] + "/" + names[j] + ".txt", labels);
			// save labels
			//save_vector_(tar_path + "/labels/mit_" + mit_folders[i] + "_" + names[j] + ".txt", labels);
			// load mesh
			if (!igl::readOFF(src_path + "/meshes/train/mit/" + mit_folders[i] + "/" + names[j] + ".off", V, F)) {
				cout << "unable to open " << src_path + "/meshes/train/mit/" + mit_folders[i] + "/" + names[j] + ".off" << endl;
				system("pause");
			}

			save_matrix_(tar_path + "/descs/global_3d/mit_" + mit_folders[i] + "_" + names[j] + ".txt", V);
			// save mesh
			/*if (!igl::writeOFF(tar_path + "/meshes/mit_" + mit_folders[i] + "_" + names[j] + ".off", V, F)) {
				cout << "unable to write " << tar_path + "/meshes/mit_" + mit_folders[i] + "_" + names[j] + ".off" << endl;
				system("pause");
			}*/
			train_file << "mit_" + mit_folders[i] + "_" + names[j] << endl;
		}
	}

	std::vector< std::string > train_folders(3);
	train_folders[0] = "faust";
	train_folders[1] = "adobe";
	train_folders[2] = "scape";


	
	
	for (int i = 0; i < 3; i++) {
		getFilesList_(src_path + "/meshes/train/" + train_folders[i], ".off", names);
		for (int j = 0; j < names.size(); j++) {
			// load labels
			//load_vector_(src_path + "/segs/train/" + train_folders[i] + "/" + names[j] + ".txt", labels);
			// save labels
			//save_vector_(tar_path + "/labels/" + train_folders[i] + "_" + names[j] + ".txt", labels);
			// load mesh
			if (!igl::readOFF(src_path + "/meshes/train/" + train_folders[i] + "/" + names[j] + ".off", V, F)) {
				cout << "unable to open " << src_path + "/meshes/train/" + train_folders[i] + "/" + names[j] + ".off" << endl;
				system("pause");
			}
			save_matrix_(tar_path + "/descs/global_3d/" + train_folders[i] + "_" + names[j] + ".txt", V);
			// save mesh
			/*if (!igl::writeOFF(tar_path + "/meshes/" + train_folders[i] + "_" + names[j] + ".off", V, F)) {
				cout << "unable to write " << tar_path + "/meshes/" + train_folders[i] + "_" + names[j] + ".off" << endl;
				system("pause");
			}*/
			train_file << train_folders[i] + "_" + names[j] << endl;
		}
	}
	train_file.close();

	// test
	ofstream test_file(tar_path + "/test.txt");
	getFilesList_(src_path + "/meshes/test/shrec", ".off", names);
	for (int j = 0; j < names.size(); j++) {
		// load labels
		//load_vector_(src_path + "/segs/test/shrec/" + names[j] + ".txt", labels);
		// save labels
		//save_vector_(tar_path + "/labels/shrec_" + names[j] + ".txt", labels);
		// load mesh
		if (!igl::readOFF(src_path + "/meshes/test/shrec/" + names[j] + ".off", V, F)) {
			cout << "unable to open " << src_path + "/meshes/test/shrec/" + names[j] + ".off" << endl;
			system("pause");
		}
		save_matrix_(tar_path + "/descs/global_3d/" + "shrec_" + names[j] + ".txt", V);
		// save mesh
		/*if (!igl::writeOFF(tar_path + "/meshes/shrec_" + names[j] + ".off", V, F)) {
			cout << "unable to write " << tar_path + "/meshes/shrec_" + names[j] + ".off" << endl;
			system("pause");
		}*/
		test_file << "shrec_" + names[j] << endl;
	}
	test_file.close();
}


void human_seg_sig2017_toric_cover_off(const std::string& src_path, const std::string& tar_path, bool normalize=true) {

	int nf = 13776;
	std::vector<std::string> mit_folders(8);
	mit_folders[0] = "bouncing";
	mit_folders[1] = "crane";
	mit_folders[2] = "handstand";
	mit_folders[3] = "jumping";
	mit_folders[4] = "march1";
	mit_folders[5] = "march2";
	mit_folders[6] = "squat1";
	mit_folders[7] = "squat2";

	std::string faust_folder = "faust";
	std::string adobe_folder = "adobe";
	std::string scape_folder = "scape";

	std::string path;
	std::vector<std::string> names;
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;

	Eigen::MatrixXd N;
	Eigen::MatrixXd UV;

	Eigen::VectorXi I;
	Eigen::VectorXi J;
	Eigen::MatrixXd U;
	Eigen::MatrixXi G;

	std::vector< int > labels;
	std::vector< int > labels_t;
	std::vector< int > labels_v;

	// MIT shapes
	cout << "converting mit" << endl;
	for (int i = 0; i < 8; i++) {
		path = src_path + "/meshes/train/mit/" + mit_folders[i] + "/meshes";
		getFilesList_(path, ".obj", names);
		// load labels
		load_vector_(src_path + "/segs/train/mit/" + mit_folders[i] + "_corrected.txt", labels_t);
		
		for (int j = 0; j < names.size(); j++) {
			// load mesh
			if (!igl::readOBJ(path + "/" + names[j] + ".obj", V, F)) {
				cout << "unable to open " << path + "/" + names[j] + ".obj" << endl;
				system("pause");
			}
			triangles_to_vertices(labels_t, F, V.rows(), labels_v);
			for (int k = 0; k < labels_v.size(); k++) {
				labels_v[k] -= 1;
			}
			if (normalize) {
				normalize_shape(V);
			}
			if (F.rows() > nf) {
				igl::decimate(V, F, nf, U, G, J, I);
				// convert labels
				labels.resize(U.rows());
				for (int k = 0; k < U.rows(); k++) {
					labels[k] = labels_v[I(k)];
				}
				// save mesh
				if (!igl::writeOFF(tar_path + "/meshes/train/mit/" + mit_folders[i] + "/" + names[j] + ".off", U, G)) {
					cout << "unable to write " << tar_path + "/meshes/train/mit/" + mit_folders[i] + "/" + names[j] + ".off" << endl;
				}
				// save labels
				save_vector_(tar_path + "/segs/train/mit/" + mit_folders[i] + "/" + names[j] + ".txt",
					labels);
			}
			else {
				// save mesh
				igl::writeOFF(tar_path + "/meshes/train/mit/" + mit_folders[i] + "/" + names[j] + ".off", V, F);
				// save labels
				save_vector_(tar_path + "/segs/train/mit/" + mit_folders[i] + "/" + names[j] + ".txt", labels_v);
			}
		}
	}//*/


	// Faust shapes
	cout << "converting faust" << endl;
	path = src_path + "/meshes/train/faust";
	getFilesList_(path, ".off", names);

	// load labels
	load_vector_(src_path + "/segs/train/faust/faust_corrected.txt", labels_t);
	

	for (int i = 0; i < names.size(); i++) {

		// load mesh
		//system("pause");
		if (!igl::readOFF(path + "/" + names[i] + ".off", V, F)) {
			cout << "unable to open " << path + "/" + names[i] + ".off" << endl;
			system("pause");
		}
		triangles_to_vertices(labels_t, F, V.rows(), labels_v);
		for (int k = 0; k < labels_v.size(); k++) {
			labels_v[k] -= 1;
		}
		if (normalize) {
			normalize_shape(V);
		}
		if (F.rows() > nf) {
			igl::decimate(V, F, nf, U, G, J, I);
			// convert labels
			labels.resize(U.rows());
			for (int k = 0; k < U.rows(); k++) {
				labels[k] = labels_v[I(k)];
			}
			cout << "yyyyyyyyyyyy" << endl;
			// save mesh
			igl::writeOFF(tar_path + "/meshes/train/faust/" + names[i] + ".off", U, G);
			// save labels
			save_vector_(tar_path + "/segs/train/faust/" + names[i] + ".txt",
				labels);
		}
		else {
			// save mesh
			igl::writeOFF(tar_path + "/meshes/train/faust/" + names[i] + ".off", V, F);
			// save labels
			save_vector_(tar_path + "/segs/train/faust/" + names[i] + ".txt", labels_v);
		}
	}//*/



	// Adobe shapes
	cout << "converting adobe" << endl;
	path = src_path + "/meshes/train/adobe";
	getFilesList_(path, ".off", names);

	for (int i = 0; i < names.size(); i++) {
		// load labels
		load_vector_(src_path + "/segs/train/adobe/" + names[i] + ".txt", labels_t);
		// load mesh
		if (!igl::readOFF(path + "/" + names[i] + ".off", V, F)) {
			cout << "unable to open " << path + "/" + names[i] + ".off" << endl;
			system("pause");
		}
		triangles_to_vertices(labels_t, F, V.rows(), labels_v);
		for (int k = 0; k < labels_v.size(); k++) {
			labels_v[k] -= 1;
		}
		if (normalize) {
			normalize_shape(V);
		}
		if (F.rows() > nf) {
			igl::decimate(V, F, nf, U, G, J, I);
			// convert labels
			labels.resize(U.rows());
			for (int k = 0; k < U.rows(); k++) {
				labels[k] = labels_v[I(k)];
			}
			// save mesh
			igl::writeOFF(tar_path + "/meshes/train/adobe/" + names[i] + ".off", U, G);
			// save labels
			save_vector_(tar_path + "/segs/train/adobe/" + names[i] + ".txt",
				labels);
		}
		else {
			// save mesh
			igl::writeOFF(tar_path + "/meshes/train/adobe/" + names[i] + ".off", V, F);
			// save labels
			save_vector_(tar_path + "/segs/train/adobe/" + names[i] + ".txt", labels_v);
		}
	}//*/

	//int nf_ = nf;
	//nf = 5000000;
	// SCAPE shapes
	cout << "converting scape" << endl;
	path = src_path + "/meshes/train/scape";
	getFilesList_(path, ".off", names);
	// load labels
	cout << names.size() << endl;
	load_vector_(src_path + "/segs/train/scape/scape_corrected.txt", labels_t);
	for (int i = 0; i < names.size(); i++) {
		// load mesh
		if (!igl::readOFF(path + "/" + names[i] + ".off", V, F)) {
			cout << "unable to open " << path + "/" + names[i] + ".off" << endl;
			system("pause");
		}
		triangles_to_vertices(labels_t, F, V.rows(), labels_v);
		for (int k = 0; k < labels_v.size(); k++) {
			labels_v[k] -= 1;
		}
		if (normalize) {
			normalize_shape(V);
		}
		if (F.rows() > nf) {
			igl::decimate(V, F, nf, U, G, J, I);
			// convert labels
			labels.resize(U.rows());
			for (int k = 0; k < U.rows(); k++) {
				labels[k] = labels_v[I(k)];
			}
			// save mesh
			//cout << names[i] << endl;
			igl::writeOFF(tar_path + "/meshes/train/scape/" + names[i] + ".off", U, G);
			// save labels
			save_vector_(tar_path + "/segs/train/scape/" + names[i] + ".txt",
				labels);
		}
		else {
			// save mesh
			igl::writeOFF(tar_path + "/meshes/train/scape/" + names[i] + ".off", V, F);
			// save labels
			save_vector_(tar_path + "/segs/train/scape/" + names[i] + ".txt", labels_v);
		}
	}
	//nf = nf_;//*/
	// SHREC shapes
	cout << "converting shrec" << endl;
	path = src_path + "/meshes/test/shrec";
	getFilesList_(path, ".off", names);

	for (int i = 0; i < names.size(); i++) {
		// load labels
		load_vector_(src_path + "/segs/test/shrec/" + names[i] + ".txt", labels_t);
		// load mesh
		if (!igl::readOFF(path + "/" + names[i] + ".off", V, F)) {
			cout << "unable to open " << path + "/" + names[i] + ".off" << endl;
			system("pause");
		}
		triangles_to_vertices(labels_t, F, V.rows(), labels_v);
		for (int k = 0; k < labels_v.size(); k++) {
			labels_v[k] -= 1;
		}
		if (normalize) {
			normalize_shape(V);
		}
		if (F.rows() > nf) {
			igl::decimate(V, F, nf, U, G, J, I);
			// convert labels
			labels.resize(U.rows());
			cout << labels.size() << endl;
			for (int k = 0; k < U.rows(); k++) {
				labels[k] = labels_v[I(k)];
			}
			// save mesh
			igl::writeOFF(tar_path + "/meshes/test/shrec/" + names[i] + ".off", U, G);
			// save labels
			save_vector_(tar_path + "/segs/test/shrec/" + names[i] + ".txt",
				labels);
		}
		else {
			// save mesh
			igl::writeOFF(tar_path + "/meshes/test/shrec/" + names[i] + ".off", V, F);
			// save labels
			save_vector_(tar_path + "/segs/test/shrec/" + names[i] + ".txt", labels_v);
		}
	}//*/
}

void human_seg_sig2017_dataset(const std::string& src_path, const std::string& tar_path, double rad, int nring, int ndir) {
	std::vector<std::string> mit_folders(8);
	mit_folders[0] = "bouncing";
	mit_folders[1] = "crane";
	mit_folders[2] = "handstand";
	mit_folders[3] = "jumping";
	mit_folders[4] = "march1";
	mit_folders[5] = "march2";
	mit_folders[6] = "squat1";
	mit_folders[7] = "squat2";

	std::string faust_folder = "faust";
	std::string adobe_folder = "adobe";
	std::string scape_folder = "scape";

	int npools = 1;
	double ratio = 0.25;
	std::vector<double> ratios(npools + 1);
	ratios[0] = 1.0;
	//double rad = 0.05;
	std::vector<double> radius(npools + 1);
	radius[0] = rad;
	for (int i = 0; i < npools; i++) {
		ratios[i + 1] = ratio * ratios[i];
		radius[i + 1] = sqrt(1. / ratio)*radius[i];
	}

	std::vector<int> nrings(npools + 1);

	std::vector<int> ndirs(npools + 1);

	for (int i = 0; i <= npools; i++) {
		ndirs[i] = ndir;
		nrings[i] = nring;
	}

	cout << "compute mit shapes patch op" << endl;
	for (int i = 0; i < 8; i++) {
		computeFilterMaper_(
			ratios,
			radius,
			nrings,
			ndirs,
			src_path + "/meshes/train/mit/" + mit_folders[i],
			"",
			tar_path + "/patch_ops/train/mit/" + mit_folders[i]);
	}

	cout << "compute adobe shapes patch op" << endl;
	computeFilterMaper_(
		ratios,
		radius,
		nrings,
		ndirs,
		src_path + "/meshes/train/adobe",
		"",
		tar_path + "/patch_ops/train/adobe");

	cout << "compute faust shapes patch op" << endl;
	computeFilterMaper_(
		ratios,
		radius,
		nrings,
		ndirs,
		src_path + "/meshes/train/faust",
		"",
		tar_path + "/patch_ops/train/faust");//*/

	cout << "compute scape shapes patch op" << endl;
	computeFilterMaper_(
		ratios,
		radius,
		nrings,
		ndirs,
		src_path + "/meshes/train/scape",
		"",
		tar_path + "/patch_ops/train/scape");

	cout << "compute shrec shapes patch op" << endl;
	computeFilterMaper_(
		ratios,
		radius,
		nrings,
		ndirs,
		src_path + "/meshes/test/shrec",
		"",
		tar_path + "/patch_ops/test/shrec");//*/

}

#endif // !HUMAN_SEG_SIG_2017_TORIC_COVER_H
