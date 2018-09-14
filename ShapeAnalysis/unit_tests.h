#pragma once
#ifndef UNIT_TESTS_H
#define UNIT_TESTS_H

#include <GPC.h>
#include <patch_op.h>
#include "visualize.h"
#include <igl/edges.h>


void compute_tangent_vec(const GPC& gpc, int x_id, double angle, Eigen::Vector3d& t) {

	Eigen::Vector3d x = gpc.getVertices().row(x_id).transpose();
	int p_id = gpc.getOneRingVertices(x_id)[0];
	int q_id = gpc.getOneRingVertices(x_id)[1];
	Eigen::Vector3d p = gpc.getVertices().row(p_id).transpose();
	Eigen::Vector3d q = gpc.getVertices().row(q_id).transpose();
	Eigen::Vector3d u = p - x;
	u.normalize();
	Eigen::Vector3d v = q - x;
	v.normalize();
	v -= v.dot(u)*u;
	v.normalize();
	t = cos(angle)*u + sin(angle)*v;
}

void compute_transported_vec(const GPC& gpc, const Eigen::Vector3d& t,
	double& tx,
	const Eigen::Vector3d& bar_coords,
	const Eigen::Vector3i& contrib_idx,
	Eigen::Matrix3d& v_to_contrib,
	Eigen::Matrix3d& t_v,
	Eigen::Vector3d& t_x) {



	if (gpc.isMeshBoundaryPoint(contrib_idx(0)) ||
		gpc.isMeshBoundaryPoint(contrib_idx(1)) ||
		gpc.isMeshBoundaryPoint(contrib_idx(2))) {
		tx = 0;
		t_v.setZero();
		t_x.setZero();
		cout << "zzzzz" << endl;
		return;
	}
	
	// transport 

	int p_id;
	int q_id;
	
	Eigen::Vector3d p;
	Eigen::Vector3d q;
	Eigen::Vector3d u;
	Eigen::Vector3d v;
	Eigen::Vector3d x;

	for (int i = 0; i < 3; i++) {
		x = gpc.getVertices().row(contrib_idx(i)).transpose();
		p_id = gpc.getOneRingVertices(contrib_idx(i))[0];
		q_id = gpc.getOneRingVertices(contrib_idx(i))[1];
		p = gpc.getVertices().row(p_id).transpose();
		q = gpc.getVertices().row(q_id).transpose();
		u = p - x;
		u.normalize();
		v = q - x;
		v.normalize();
		v -= v.dot(u)*u;
		v.normalize();
		t_v.col(i) = cos(t(i))*u + sin(t(i))*v;
	}

	

	// contributors
	x.setZero();
	for (int i = 0; i < 3; i++) {

		x += bar_coords(i)*gpc.getVertices().row(contrib_idx(i)).transpose();
	}
	for (int i = 0; i < 3; i++) {
		v_to_contrib.col(i) = x - gpc.getVertices().row(contrib_idx(i)).transpose();
	}
	for (int i = 0; i < 3; i++) {
		if (bar_coords(i) > 0.95) {
			t_x.setZero();
			return;
		}
	}

	//p_id = contrib_idx(0);
	//q_id = contrib_idx(1);
	u = gpc.getVertices().row(contrib_idx(0)).transpose() - x;
	u.normalize();

	p = gpc.getVertices().row(contrib_idx(0)).transpose();
	q = gpc.getVertices().row(contrib_idx(1)).transpose();
	v = gpc.getVertices().row(contrib_idx(2)).transpose();
	Eigen::Vector3d w = (q - p).cross(v - p);
	w.normalize();



	//v = gpc.getVertices().row(contrib_idx(1)).transpose() - x;
	//v.normalize();
	//v -= u.dot(v)*u;
	//v.normalize();

	v = w.cross(u);

	t_x = cos(tx)*u + sin(tx)*v;
}


void test_gpc_interpolation(const std::string& shape, int bp_id, double radius, int nrings, int ndirs) {
	// load mesh
	geodesic::Mesh mesh;
	GPC* gpc = NULL;
	if (loadMesh(shape, mesh)) {
		gpc = new GPC(mesh);
	}
	else {
		exit(666);
		// throw error
	}
	gpc = new GPC(mesh);

	gpc->compute(bp_id, 1.5*radius, 0, 0.000001);

	


	int nv = gpc->getNbVertices();
	/*for (int i = 0; i < nv; i++) {
		cout << mesh.vertices()[i].adjacent_vertices().size() << endl;
	}*/
	std::vector<double> contrib(nv);
	std::fill(contrib.begin(), contrib.end(), 0.0);
	double r;
	double t;
	Eigen::Vector3i contrib_idx;
	Eigen::Vector3d bar_coords;
	Eigen::Vector3d transport;
	double tx;
	Eigen::Matrix3d v_to_contrib;
	Eigen::Vector3d t_x;
	Eigen::Matrix3d t_v;

	Eigen::MatrixXd V;
	Eigen::MatrixXi F;

	if (!igl::readOFF(shape, V, F))
	{
		cout << "failed to load mesh" << endl;
		exit(666);
	}
	Eigen::MatrixXd BPC(nrings*ndirs * 3, 3);
	BPC.setZero();
	Eigen::MatrixXd BPT(nrings*ndirs * 3, 3);
	BPT.setZero();
	Eigen::MatrixXd C(V.rows(), 1);
	C.setZero();
	Eigen::MatrixXd VT(nrings*ndirs*3, 3);
	VT.setZero();
	Eigen::MatrixXd VC(nrings*ndirs*3, 3);
	VC.setZero();
	Eigen::MatrixXd VX(nrings*ndirs, 3);
	VX.setZero();
	Eigen::MatrixXd BPX(nrings*ndirs, 3);
	BPX.setZero();

	for (int i = 0; i < nrings; i++) {
		r = (i + 1)*radius / (nrings);
		for (int j = 0; j < ndirs; j++) {
			t = 2.*M_PI*j / ndirs;
			gpc->getPatchBin(r, t, contrib_idx, bar_coords, transport, tx);

			for (int k = 0; k < 3; k++) {
				//transport(k) = gpc->getU1()[contrib_idx(k)];
			}
			compute_transported_vec(*gpc, transport, tx, bar_coords, contrib_idx,
				v_to_contrib,
				t_v, t_x);
			for (int k = 0; k < 3; k++) {
				VT.row(3 * (ndirs*i + j) + k) = t_v.col(k).transpose();
				BPT.row(3 * (ndirs*i + j) + k) = V.row(contrib_idx(k));
				VC.row(3 * (ndirs*i + j) + k) = v_to_contrib.col(k).transpose();
				BPC.row(3 * (ndirs*i + j) + k) = V.row(contrib_idx(k));
				BPX.row((ndirs*i + j)) += bar_coords(k)*V.row(contrib_idx(k));
			}
			VX.row((ndirs*i + j)) = t_x.transpose();
		}
	}

	// display


	for (int i = 0; i < V.rows(); i++) {
		C(i, 0) = gpc->getAngle()[i];
		if (C(i, 0) < 2.*M_PI - 0.1) {
			//C(i, 0) = 0.0;
		}
		if (C(i, 0) > 0.1) {
			//C(i, 0) = 0.0;
		}
		//C(i, 0) = gpc->getDist()[i];
	}




	const Eigen::RowVector3d black(0, 0, 0);
	const Eigen::RowVector3d white(255, 255, 255);
	const Eigen::RowVector3d red(255, 0, 0);
	const Eigen::RowVector3d green(0, 255, 0);
	const Eigen::RowVector3d grey(100, 100, 100);

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
			//viewer.data.add_edges(BPX, BPX + 0.25*VX, red);
			viewer.data.add_edges(BPT, BPT + 0.02*VT, grey);
			viewer.data.add_edges(BPC, BPC + VC, black);
			//viewer.data.add_edges(BC, BC + X, black);
			return true;
		}
		}
	};
	viewer.callback_key_down(viewer, ' ', 0);
	viewer.core.show_lines = false;
	viewer.launch();// */

	delete gpc;
}

void display_GPC_transport(const std::string& shape, int bp_id, double radius) {
	// load mesh
	geodesic::Mesh mesh;
	GPC* gpc = NULL;
	if (loadMesh(shape, mesh)) {
		gpc = new GPC(mesh);
	}
	else {
		exit(666);
		// throw error
	}
	gpc = new GPC(mesh);

	gpc->compute(bp_id, radius, 0, 0.000001);

	int nv = gpc->getNbVertices();


	Patch patch;

	gpc->getPatch(patch);

	/*for (int i = 0; i < nv; i++) {
	cout << mesh.vertices()[i].adjacent_vertices().size() << endl;
	}*/
	std::vector<double> contrib(nv);
	std::fill(contrib.begin(), contrib.end(), 0.0);
	double r;

	Eigen::Vector3i contrib_idx;
	Eigen::Vector3d bar_coords;
	Eigen::Vector3d transport;
	double tx;
	Eigen::Matrix3d v_to_contrib;
	Eigen::Vector3d t_x;
	Eigen::Matrix3d t_v;

	Eigen::MatrixXd V;
	Eigen::MatrixXi F;

	if (!igl::readOFF(shape, V, F))
	{
		cout << "failed to load mesh" << endl;
		exit(666);
	}
	
	Eigen::MatrixXd C(V.rows(), 1);
	C.setZero();
	Eigen::MatrixXd T(V.rows(), 3);
	T.setZero();

	int nv_patch = patch.r().size();

	Eigen::Vector3d t;
	for (int i = 0; i < nv_patch; i++) {
		compute_tangent_vec(*gpc, patch.l()[i], patch.u1()[i], t);
		C(patch.l()[i], 0) = patch.theta()[i];
		T.row(patch.l()[i]) = t.transpose();
	}

	// display


	/*for (int i = 0; i < V.rows(); i++) {
		C(i, 0) = gpc->getAngle()[i];
		if (C(i, 0) < 2.*M_PI - 0.1) {
			//C(i, 0) = 0.0;
		}
		if (C(i, 0) > 0.1) {
			//C(i, 0) = 0.0;
		}
		//C(i, 0) = gpc->getDist()[i];
	}*/




	const Eigen::RowVector3d black(0, 0, 0);
	const Eigen::RowVector3d white(255, 255, 255);
	const Eigen::RowVector3d red(255, 0, 0);
	const Eigen::RowVector3d green(0, 255, 0);
	const Eigen::RowVector3d grey(100, 100, 100);

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
			//viewer.data.add_edges(BPX, BPX + 0.25*VX, red);
			viewer.data.add_edges(V, V + 0.04*T, grey);
			return true;
		}
		}
	};
	viewer.callback_key_down(viewer, ' ', 0);
	viewer.core.show_lines = false;
	viewer.launch();// */

	delete gpc;
}

#endif // !UNIT_TESTS_H
