#pragma once
#ifndef CUBE_SEGMENTATION_H
#define CUBE_SEGMENTATION_H
#include <iostream>
#include <fstream>

//#include <igl/viewer/Viewer.h>
//#include <igl/parula.h>
//#include <igl/read_triangle_mesh.h>
#include <igl/readOFF.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>

using namespace std;


template <typename T>
vector<size_t> sort_indexes(const vector<T> &v) {

	// initialize original index locations
	vector<size_t> idx(v.size());
	iota(idx.begin(), idx.end(), 0);

	// sort indexes based on comparing values in v
	sort(idx.begin(), idx.end(),
		[&v](size_t i1, size_t i2) {return v[i1] < v[i2]; });

	return idx;
}

inline int vertex_label(float x, float y, float z, 
						float x_min, float x_max,
						float y_min, float y_max,
						float z_min, float z_max,
						int nx, int ny, int nz) {
	//int x_ = int(floor((nx*(x - x_min)) / (x_max - x_min)));
	int x_ = int(floor((nx*(x - x_min)) / (x_max - x_min)));
	if (2*x_ >= nx) {
		//cout << "zz" << endl;
		x_ = nx - 1 - x_;
		//x_ = 0;
	}
	int y_ = int(floor((ny*(y - y_min)) / (y_max - y_min)));
	int z_ = int(floor((nz*(z - z_min)) / (z_max - z_min)));
	
	return nz*(ny*x_ + y_) + z_;
}

void squeeze_labels(std::vector<int>& labels) {
	int nv = labels.size();
	std::vector< int > sorted_labels(nv);
	for (int i = 0; i < nv; i++) {
		sorted_labels[i] = labels[i];
	}
	std::sort(sorted_labels.begin(), sorted_labels.end());
	std::map<int, int> labels_set;
	int l = sorted_labels[0];
	int nb_labels = 1;
	labels_set.insert(std::make_pair(l, 0));
	for (int i = 0; i < nv; i++) {
		if (sorted_labels[i] > l) {
			l = sorted_labels[i];
			labels_set.insert(std::make_pair(l, nb_labels));
			//cout << nb_labels << endl;
			nb_labels++;
		}
	}
	//cout << labels_set.size() << endl;
	for (int i = 0; i < nv; i++) {
		labels[i] = labels_set[labels[i]];
	}
}

bool voxelSegmentation(const Eigen::MatrixXd& V, float x_min, float x_max,
										        float y_min, float y_max,
												float z_min, float z_max,
												int nx, int ny, int nz,
												std::vector< int >& labels) {
	// save signals
	/*Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	if (!igl::readOFF(path, V, F))
	{
		cout << "failed to load sphere mesh" << endl;
		return false;
	}*/
	int nv = V.rows();
	labels.resize(nv);
	float x;
	float y;
	float z;
	for (int i = 0; i < nv; i++) {
		x = V(i, 0);
		y = V(i, 1);
		z = V(i, 2);
		labels[i] = vertex_label(x, y, z,
			x_min, x_max,
			y_min, y_max,
			z_min, z_max,
			nx, ny, nz);
	}
	squeeze_labels(labels);
	return true;
}

struct box3d {
	float x_min;
	float x_max;
	float y_min;
	float y_max;
	float z_min;
	float z_max;
};

void get_bounding_box(const Eigen::MatrixXd& V, box3d& bnd_box) {
	bnd_box.x_min = std::numeric_limits<float>::max();
	bnd_box.x_max = -std::numeric_limits<float>::max();
	bnd_box.y_min = std::numeric_limits<float>::max();
	bnd_box.y_max = -std::numeric_limits<float>::max();
	bnd_box.z_min = std::numeric_limits<float>::max();
	bnd_box.z_max = -std::numeric_limits<float>::max();

	for (int i = 0; i < V.rows(); i++) {
		if (V(i, 0) < bnd_box.x_min) {
			bnd_box.x_min = V(i, 0);
		}
		if (V(i, 0) > bnd_box.x_max) {
			bnd_box.x_max = V(i, 0);
		}

		if (V(i, 1) < bnd_box.y_min) {
			bnd_box.y_min = V(i, 1);
		}
		if (V(i, 1) > bnd_box.y_max) {
			bnd_box.y_max = V(i, 1);
		}

		if (V(i, 2) < bnd_box.z_min) {
			bnd_box.z_min = V(i, 2);
		}
		if (V(i, 2) > bnd_box.z_max) {
			bnd_box.z_max = V(i, 2);
		}
	}
}

#endif // !CUBE_SEGMENTATION_H
