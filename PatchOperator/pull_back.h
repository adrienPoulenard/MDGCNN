#pragma once
#ifndef PULL_BACK_H
#define PULL_BACK_H

#include <geodesic_mesh.h>
#include "matrixContainer.h"
#include <array>

#include <nanoflann.hpp>

#include <ctime>
#include <cstdlib>
#include <iostream>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "local_coordinates.h"



struct {
	bool operator()(const Eigen::Vector2d& a, const Eigen::Vector2d& b) const
	{
		return a(1) < b(1);
	}
} ycomp;

struct {
	bool operator()(const Eigen::Vector2d& a, const Eigen::Vector2d& b) const
	{
		return a(0) < b(0);
	}
} xcomp;

inline double truncate(double x) {
	if (x >= 0.) {
		return floor(x);
	}
	else {
		return ceil(x);
	}
}

inline void barycentricCoordinates(
	const Eigen::Vector2d& x,
	Eigen::Vector3d& b_coord,
	const Eigen::Vector2d& a,
	const Eigen::Vector2d& b,
	const Eigen::Vector2d& c) {
	Eigen::Matrix2d T;
	T.col(0) = a - c;
	T.col(1) = b - c;
	b_coord.head(2) = T.inverse()*(x - c);
	b_coord[2] = 1.0 - b_coord[0] - b_coord[1];
}

inline void barycentricCoordinates(
	const Eigen::Vector2d& x,
	Eigen::Vector3d& b_coord,
	const Eigen::Matrix<double, 3, 2>& vert) 
{
	Eigen::Matrix2d T;
	T.col(0) = (vert.row(0) - vert.row(2)).transpose();
	T.col(1) = (vert.row(1) - vert.row(2)).transpose();
	b_coord.head(2) = T.inverse()*(x - vert.row(2).transpose());
	b_coord[2] = 1.0 - b_coord[0] - b_coord[1];
}

inline int segmentTruncate(int i, int a, int b) {
	return std::max(std::min(i, b), a);
}

// ineficient
template <int k>
inline void pullBackTriangle(const Eigen::Matrix<double, -1, k>& functions, block_matrix<double, k, 1>& grid, int n0, int m0, double pixel_size,
	const Eigen::Vector2d& a,
	const Eigen::Vector2d& b,
	const Eigen::Vector2d& c,
	int a_id,
	int b_id,
	int c_id) {

	//compute bounding box:
	double Xmin = std::min(std::min(a(0), b(0)), c(0)) / pixel_size;
	double Xmax = std::max(std::max(a(0), b(0)), c(0)) / pixel_size;
	double Ymin = std::min(std::min(a(1), b(1)), c(1)) / pixel_size;
	double Ymax = std::max(std::max(a(1), b(1)), c(1)) / pixel_size;


	Eigen::Vector2d pt;

	int xmin = segmentTruncate(truncate(Xmin) - 1, -n0, grid.cols() - n0 - 1);
	int xmax = segmentTruncate(truncate(Xmax) + 1, -n0, grid.cols() - n0 - 1);
	int ymin = segmentTruncate(truncate(Ymin) - 1, -m0, grid.cols() - m0 - 1);
	int ymax = segmentTruncate(truncate(Ymax) + 1, -m0, grid.cols() - m0 - 1);

	for (int px = xmin; px < xmax; px++) {
		pt(0) = px*pixel_size;
		for (int py = ymin; py < ymax; py++) {
			pt(1) = py*pixel_size;
			Eigen::Vector3d b_coord;
			barycentricCoordinates(pt, b_coord, a, b, c);
			if ((b_coord(0) >= 0.0) & (b_coord(1) >= 0.0) & (b_coord(2) >= 0.0)) {
				for (int i = 0; i < k; i++) {
					grid(px + n0, py + m0) =
						(b_coord(0)*functions.row(a_id) +
							b_coord(1)*functions.row(b_id) +
							b_coord(2)*functions.row(c_id)).transpose();
					//cout << grid(x + n0, y + m0) << endl;
				}
			}
		}
	}
}




// pull back class for meshes
class meshLocalPullBack: public localPullBack< geodesic::Mesh > {
public:
	// a window model is given as a set of cells
	// each cell is a quadrilateral
	// the coordinates of the 4 vertices of a cell are given as the rows of a model matrix
	meshLocalPullBack(const Eigen::Matrix<double, -1, 8>& model) : window_model(model), resultSet(std::size_t(1)) {
		nb_cells = model.rows();
		//cells_vertices_triangles_id.resize(nb_cells, 5);
		//cells_vertices_bar_coord.resize(nb_cells, 15);
		search_tree = NULL;
		nn_idx.resize(1);
		nn_dist.resize(1);
		resultSet.init(&nn_idx[0], &nn_dist[0]);
		query_pt.resize(2);
		//triplets.resize();
	}
	~meshLocalPullBack() {
		if (search_tree != NULL) {
			delete search_tree;
		}
	}
	//unsigned getNbCells() const { return window_model.rows(); }
	// compute the pullBack operator in a local system of coordinates
	// centered at the point "vertex_id"
	// the local pullBack is stored in a sparse matrix
	virtual void computeLocalPullBackOperatorAt(
		unsigned int pt_id,
		const Eigen::MatrixXd& coord_mat, 
		const std::vector<bool>& valid_idx,
		const std::vector<bool>& valid_idx_inverse,
		geodesic::Mesh& mesh, 
		std::vector< Eigen::Triplet<double> >& triplets) {
		
		
		nv = valid_idx.size();
		nb_valid = coord_mat.rows();
		// build kd-tree
		if (search_tree != NULL) {
			delete search_tree;
		}
		search_tree = new kdTree(2, coord_mat, 10);
		search_tree->index->buildIndex();
		// each cell is divided in 4 triangles.
		// for each triangle of the cell
		// for each vertex of the triangle
		// find the triangle of mesh contaning this vertex
		// compute barycentric coordintes 
		double T_area = 0.0;
		Eigen::Matrix<double, 5, 3> b_coord;
		Eigen::Matrix<double, 5, 3> e_coord;
		Eigen::Vector3d b_coord_tmp;
		Eigen::Vector3d e_coord_tmp;
		Eigen::Vector2d x;
		bool valid_cell = true;
		std::vector< geodesic::face_pointer > f(5);
		std::vector< int > v_id(3);

		for (unsigned cell_id = 0; cell_id < nb_cells; cell_id++) {
			valid_cell = true;
			// central point
			x.setZero();
			for (unsigned i = 0; i < 4; i++) {
				x(0) += window_model(cell_id, 2 * i);
				x(1) += window_model(cell_id, 2 * i + 1);
			}
			x /= 4.0;
			query_pt[0] = x(0);
			query_pt[1] = x(1);
			search_tree->index->findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));
			
			valid_cell =
				(valid_cell &
					findTriangle(x, nn_idx[0], coord_mat, valid_idx, valid_idx_inverse, mesh,
						e_coord_tmp, b_coord_tmp, f[4]));
			b_coord.row(4).transpose() = b_coord_tmp;
			e_coord.row(4).transpose() = e_coord_tmp;

			// loop through vertices of the cell
			for (unsigned i = 0; i < 4; i++) {
				x(0) = window_model(cell_id, 2 * i);
				x(1) = window_model(cell_id, 2 * i + 1);
				query_pt[0] = x(0);
				query_pt[1] = x(1);
				search_tree->index->findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));

				valid_cell =
					(valid_cell &
						findTriangle(x, nn_idx[0], coord_mat, valid_idx, valid_idx_inverse, mesh,
							e_coord_tmp, b_coord_tmp, f[4]));
				b_coord.row(4).transpose() = b_coord_tmp;
				e_coord.row(4).transpose() = e_coord_tmp;
			}
			// compute the pullBack to the cell
			if (valid_cell) {
				// loop through triangles composing the cell
				for (unsigned i = 0; i < 4; i++) {
					T_area =
						(e_coord.row(i) - e_coord.row(4)).transpose().cross(
						(e_coord.row((i + 1) % 4) - e_coord.row(4)).transpose()).norm() / 2.0;

					unsigned int I = pt_id*nb_cells + cell_id;
					unsigned int J = 0;
					double V = 0.0;
					// loop through vertices of the triangle
					v_id[0] = i;
					v_id[1] = (i + 1) % 4;
					v_id[2] = 4;
					for (unsigned j = 0; j < 3; j++) {
						// loop through vertices of the mesh triangle containing the vertex
						for (unsigned k = 0; k < 3; k++) {
							J = f[v_id[j]]->adjacent_vertices()[k]->id();
							V = T_area*b_coord(v_id[j], k) / 3.0;
							triplets.push_back(Eigen::Triplet<double>(I, J, V));
						}// mesh triangle vertices
					}// cell triangle vertices	
				}// cell triangles
			}// if(valid_cell)
		}// cells
	}

protected:

private:
	unsigned int nv;
	unsigned int nb_valid;
	//unsigned int nb_cells;
	const Eigen::Matrix<double, -1, 8>& window_model;

	//Eigen::Matrix<int, -1, 5> cells_vertices_triangles_id;
	//Eigen::Matrix<double, -1, 15> cells_vertices_bar_coord;

	// kd-tree for quick mesh to cells correspondance search
	typedef nanoflann::KDTreeEigenMatrixAdaptor< Eigen::MatrixXd >  kdTree;
	kdTree* search_tree;
	std::vector<double> query_pt;
	std::vector<std::size_t> nn_idx;
	std::vector<double> nn_dist;
	nanoflann::KNNResultSet<double> resultSet;
	// create triplet list for sparse matrix of pull back operator
	//typedef std::vector< Eigen::Triplet<double> > tripletList;
	//std::vector< Eigen::Triplet<double> > triplets;

	// find the triangle "f" adjactent to "x_nn_idx" which contains "x"
	// compute the barycentric coordinates "b_coord" of "x"
	// compute the corresponding 3d point "X"
	inline bool findTriangle(
		const Eigen::Vector2d& x,
		int x_nn_idx,
		const Eigen::MatrixXd& coord_mat,
		const std::vector<bool>& valid_idx,
		const std::vector<bool>& valid_idx_inverse,
		geodesic::Mesh& mesh,
		Eigen::Vector3d& X,
		Eigen::Vector3d& b_coord,
		geodesic::face_pointer& f
	) {

		if (valid_idx[x_nn_idx] == -1) {
			return false;
		}

		//geodesic::face_pointer f;
		geodesic::vertex_pointer nn = &(mesh.vertices()[valid_idx[x_nn_idx]]);
		geodesic::vertex_pointer v;
		Eigen::Matrix<double, 3, 2> verts;
		//Eigen::Vector3d b_coord;

		for (unsigned i = 0; i < nn->adjacent_faces().size(); i++) {
			f = nn->adjacent_faces()[i];
			for (unsigned j = 0; j < 3; j++) {
				v = f->adjacent_vertices()[j];
				if (valid_idx_inverse[v->id()] == -1) {
					return false;
				}
				verts.row(j) = coord_mat.row(valid_idx_inverse[v->id()]);
			}
			barycentricCoordinates(x, b_coord, verts);

			if ((b_coord(0) >= 0.0) & (b_coord(1) >= 0.0) & (b_coord(2) >= 0.0)) {
				// update triplets
				X.setZero();
				for (unsigned j = 0; j < 3; j++) {
					v = f->adjacent_vertices()[j];
					X(0) += b_coord(j)*v->x();
					X(1) += b_coord(j)*v->y();
					X(2) += b_coord(j)*v->z();
				}
				//unsigned int I = vertex_id*vertex_id + cell_id;
				//unsigned int J = 

				//triplets.push_back(Eigen::Triplet<double>(I,J,V));
				return true;
			}
		}
		return false;
	}
};

// no need for this class a simple function taking a locall coord and a local pull back object as input 
// is enougth   (take a pointer to a sparse matrix and update it)
// 
/*
template<class T>
inline constexpr T pow(const T base, unsigned const exponent)
{
	// (parentheses not required in next line)
	return (exponent == 0) ? 1 : (base * pow(base, exponent - 1));
}*/

template <class DataStruct, unsigned dim>
void computePullBackLayer(
	localPullBack< DataStruct >& local_pullback,
	localCoordinates< DataStruct >& local_coords,
	double radius_limit,
	int stop_criterion,
	double precision,
	Eigen::SparseMatrix<double>* & pbMat);

template<>
void computePullBackLayer< geodesic::Mesh, 2>(
	localPullBack< geodesic::Mesh >& local_pullback,
	localCoordinates< geodesic::Mesh >& local_coords,
	double radius_limit,
	int stop_criterion,
	double precision,
	Eigen::SparseMatrix<double>* & pbMat) {
	if (pbMat != NULL) {
		delete pbMat;
	}
	unsigned int nb_cells = local_pullback.getNbCells();
	unsigned int nb_vertices = local_coords.getNbVertices();
	std::vector< Eigen::Triplet<double> > triplets;
	triplets.reserve(nb_cells * 15 * nb_vertices); // the 15 here might vary depending on the data structure we are working with

	int percent = 0;
	double percent_tmp = 0.0;
	for (unsigned i = 0; i < nb_vertices; i++) {
		percent_tmp = 100.0*i / nb_vertices;
		if (floor(percent_tmp - percent) > 0) {
			percent++;
			cout << percent << endl;
		}
		local_coords.compute(i, radius_limit, stop_criterion, precision);
		local_coords.pullBack(local_pullback, triplets);
	}
	pbMat = new Eigen::SparseMatrix<double>(nb_cells*nb_vertices, nb_vertices);
	pbMat->setFromTriplets(triplets.begin(), triplets.end());
	triplets.clear();
}

#endif // !PULL_BACK_H
