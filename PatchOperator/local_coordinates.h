#pragma once
#ifndef LOCAL_COORDINATES_H
#define LOCAL_COORDINATES_H

//#include "pull_back.h"
// interface for local pullBack classes

#include <Eigen/Dense>
#include <Eigen/Sparse>

// interface for local alignment class
class localAlignment {
public:
	localAlignment() {}

	~localAlignment() {}

	virtual void computeLocalAlignmentOperatorAt(
		unsigned int pt_id,
		const Eigen::MatrixXd& transport_mat,
		const std::vector<int>& valid_idx,
		const std::vector<int>& valid_idx_inverse,
		std::vector< Eigen::Triplet<int> >& triplets) = 0;
};
template <class DataStruct>
class localPullBack {
public:
	localPullBack() {}
	~localPullBack() {}
	unsigned getNbCells() const { return nb_cells; }
	virtual void computeLocalPullBackOperatorAt(
		unsigned int pt_id,
		const Eigen::MatrixXd& coord_mat,
		const std::vector<int>& valid_idx,
		const std::vector<int>& valid_idx_inverse,
		DataStruct& manifold,
		std::vector< Eigen::Triplet<double> >& triplets) = 0;
protected:
	unsigned nb_cells;
};

// abstract template class for building local coordinates on a manifold of type "T" data format 

template <class DataStruct>
class localCoordinates {
public:
	localCoordinates(DataStruct& manifold): myManifold(manifold) {}
	~localCoordinates() {}
	virtual double compute(int basepoint_id, double rad, int stop_criterion = 0, double precision = 0.000001 ) = 0;
	const Eigen::MatrixXd& getCoords() const { return coordMat; }
	const Eigen::MatrixXd& parallelTransport() const { return pTransport; }
	const std::vector<int>& getValidIdx() const { return valid_idx; }
	const std::vector<int>& getValidIdxInverse() const { return valid_idx_inverse; }
	double getRadius() const { return radius; }
	void pullBack(localPullBack<DataStruct>& pull_back_op, std::vector< Eigen::Triplet<double> >& triplets) const {
		pull_back_op.computeLocalPullBackOperatorAt(bp_id, coordMat, valid_idx, valid_idx_inverse, myManifold, triplets);
	}
	virtual int getNbVertices() const { return nv; }
protected:
	DataStruct& myManifold; // mesh in GPC case
	Eigen::MatrixXd coordMat;
	Eigen::MatrixXd pTransport;
	std::vector<int> valid_idx; // ids of the points corresponding to coordMat rows
	std::vector<int> valid_idx_inverse; // indices of rows of coordMat corresponding to valid points
	double radius;
	int bp_id;
	int nv;
private:
};

#endif // !LOCAL_COORDINATES_H
