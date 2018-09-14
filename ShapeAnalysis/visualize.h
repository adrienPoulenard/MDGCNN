#pragma once
#ifndef VISUALIZE_H
#define VISUALIZE_H
#include <iostream>
#include <fstream>

#include <igl/viewer/Viewer.h>
#include <igl/parula.h>

#include <igl/readOFF.h>
#include <Eigen/Dense>
#include "utils.h"
#include <igl/bounding_box_diagonal.h>

void visualize(const std::string& shape) {


	Eigen::MatrixXd V;
	Eigen::MatrixXi F;

	if (!igl::readOFF(shape, V, F))
	{
		cout << "failed to load mesh" << endl;
		exit(666);
	}

	igl::viewer::Viewer viewer;
	viewer.callback_key_down = [&](igl::viewer::Viewer & viewer, unsigned char key, int)->bool
	{
		switch (key)
		{
		default:
			return false;
		case ' ':
		{
			viewer.data.set_mesh(V, F);
			viewer.data.compute_normals();
			//viewer.data.set_colors(C);
			//viewer.data.add_edges(BC, BC + X, black);
			return true;
		}
		}
	};
	viewer.callback_key_down(viewer, ' ', 0);
	viewer.core.show_lines = false;
	viewer.launch();// */
}


void visualize(const std::string& shape, const std::string& func) {


	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	
	if (!igl::readOFF(shape, V, F))
	{
		cout << "failed to load mesh" << endl;
		exit(666);
	}

	Eigen::MatrixXd C(V.rows(), 1);
	load_matrix_(func, C);

	igl::viewer::Viewer viewer;
	viewer.callback_key_down = [&](igl::viewer::Viewer & viewer, unsigned char key, int)->bool
	{
		switch (key)
		{
		default:
			return false;
		case ' ':
		{
			viewer.data.set_mesh(V, F);
			viewer.data.compute_normals();
			viewer.data.set_colors(C);
			//viewer.data.add_edges(BC, BC + X, black);
			return true;
		}
		}
	};
	viewer.callback_key_down(viewer, ' ', 0);
	viewer.core.show_lines = false;
	viewer.launch();// */
}

void visualize(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F) {



	igl::viewer::Viewer viewer;
	viewer.callback_key_down = [&](igl::viewer::Viewer & viewer, unsigned char key, int)->bool
	{
		switch (key)
		{
		default:
			return false;
		case ' ':
		{
			viewer.data.set_mesh(V, F);
			viewer.data.compute_normals();
			//viewer.data.add_edges(BC, BC + X, black);
			return true;
		}
		}
	};
	viewer.callback_key_down(viewer, ' ', 0);
	viewer.core.show_lines = false;
	viewer.launch();// */
}

template <typename T>
void visualize(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const std::vector<T>& func) {


	Eigen::MatrixXd C(V.rows(), 1);

	for (int i = 0; i < V.rows(); i++) {
		C(i, 0) = func[i];
	}

	igl::viewer::Viewer viewer;
	viewer.callback_key_down = [&](igl::viewer::Viewer & viewer, unsigned char key, int)->bool
	{
		switch (key)
		{
		default:
			return false;
		case ' ':
		{
			viewer.data.set_mesh(V, F);
			viewer.data.compute_normals();
			viewer.data.set_colors(C);
			//viewer.data.add_edges(BC, BC + X, black);
			return true;
		}
		}
	};
	viewer.callback_key_down(viewer, ' ', 0);
	viewer.core.show_lines = false;
	viewer.launch();// */
}

void visualize_labels(const std::string& shape, const std::string& labels) {
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	if (!igl::readOFF(shape, V, F)) {
		exit(666);
	}
	std::vector<int> Labels;
	load_vector_(labels, Labels);
	visualize(V, F, Labels);
}


void visualize(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::MatrixXd& C) {
	igl::viewer::Viewer viewer;
	viewer.callback_key_down = [&](igl::viewer::Viewer & viewer, unsigned char key, int)->bool
	{
		switch (key)
		{
		default:
			return false;
		case ' ':
		{
			viewer.data.set_mesh(V, F);
			viewer.data.compute_normals();
			viewer.data.set_colors(C);
			//viewer.data.add_edges(BC, BC + X, black);
			return true;
		}
		}
	};
	viewer.callback_key_down(viewer, ' ', 0);
	viewer.core.show_lines = false;
	viewer.launch();// */
}

void visualize(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::MatrixXd& C, const Eigen::MatrixXd& X) {
	igl::viewer::Viewer viewer;
	viewer.callback_key_down = [&](igl::viewer::Viewer & viewer, unsigned char key, int)->bool
	{
		switch (key)
		{
		default:
			return false;
		case ' ':
		{
			viewer.data.set_mesh(V, F);
			viewer.data.compute_normals();
			viewer.data.set_colors(C);
			viewer.data.add_edges(V, V + X, Eigen::RowVector3d(1, 0, 0));
			return true;
		}
		}
	};
	viewer.callback_key_down(viewer, ' ', 0);
	viewer.core.show_lines = false;
	viewer.launch();// */
}

void visualize3D(const std::string& shape, const std::string& coords) {
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	if (!igl::readOFF(shape, V, F))
	{
		cout << "failed to load mesh" << endl;
		exit(666);
	}
	load_matrix_(coords, V);
	visualize(V, F);
}

void visualizeRGB(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const std::vector<uint8_t>& img) {
	Eigen::MatrixXd C(V.rows(), 3);
	for (int j = 0; j < 3; j++) {
		for (int i = 0; i < V.rows(); i++) {
			C(i, j) = (double)(img[3 * i + j]) / 255.0;
		}
	}
	igl::viewer::Viewer viewer;
	viewer.callback_key_down = [&](igl::viewer::Viewer & viewer, unsigned char key, int)->bool
	{
		switch (key)
		{
		default:
			return false;
		case ' ':
		{
			viewer.data.set_mesh(V, F);
			viewer.data.compute_normals();
			viewer.data.set_colors(C);
			//viewer.data.add_edges(BC, BC + X, black);
			return true;
		}
		}
	};
	viewer.callback_key_down(viewer, ' ', 0);
	viewer.core.show_lines = false;
	viewer.launch();// */
}

void visualizeRGB(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::MatrixXd& C) {

	igl::viewer::Viewer viewer;
	viewer.callback_key_down = [&](igl::viewer::Viewer & viewer, unsigned char key, int)->bool
	{
		switch (key)
		{
		default:
			return false;
		case ' ':
		{
			viewer.data.set_mesh(V, F);
			viewer.data.compute_normals();
			viewer.data.set_colors(C);
			//viewer.data.add_edges(BC, BC + X, black);
			return true;
		}
		}
	};
	viewer.callback_key_down(viewer, ' ', 0);
	viewer.core.show_lines = false;
	viewer.launch();// */
}

void visualizeRGB(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const std::string& path) {
	Eigen::MatrixXd C;
	loadRGB(path, V.rows(), C);
	visualizeRGB(V, F, C);
}

void visualizeRGB(const std::string& path, const std::string& shape, const std::string& img_path) {
	Eigen::MatrixXd C;
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;

	if (!igl::readOFF(path + "/" + shape + ".off" , V, F))
	{
		cout << "failed to load mesh" << endl;
		exit(667);
	}

	loadRGB(img_path, V.rows(), C);

	for (int i = 0; i < V.rows(); i++) {
		cout << C.row(i) << endl;
	}
	visualizeRGB(V, F, C);
}





#endif // !VISUALIZE_H

