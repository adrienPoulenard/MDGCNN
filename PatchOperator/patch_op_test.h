#pragma once
#ifndef PATCH_OP_TEST_H
#define PATCH_OP_TEST_H

#include"sync_conv.h"


void setWeightsCheckerboard(int ndir, int nrings, double radius, Eigen::MatrixXf& weights) {
	weights.resize(ndir, nrings);
	weights.setZero();
	float sgn = 1.0;
	for (int j = 0; j < nrings; j++) {
		for (int i = 0; i < ndir; i++) {
			weights(i,j) = pow(-1.0, int(8 * i / ndir) +j);
			//weights(i, j) = pow(-1.0, int(4 * i / ndir));
		}

	}

}

void setWeightsQuadrans(int ndir, int nrings, double radius, Eigen::MatrixXf& weights) {
	weights.resize(ndir, nrings);
	weights.setZero();
	
	for (int j = 0; j < nrings; j++) {

		for (int i = 0; i < ndir; i++) {
			weights(i, j) = pow(-1.0, int(4*i / ndir));
		}
	}

}

void setWeightsWindow248(int ndir, double radius, Eigen::MatrixXf& weights) {
	// 1st ring
	int nrings = 3;
	weights.resize(ndir, 3);
	weights.setZero();
	int shift = ndir;
	int shift2 = 0;
	for (int j = 0; j < nrings; j++) {
		if (j == 1) {
			shift = ndir / 8;
		}
		else if( j == 2 ){
			shift = ndir / 16;
			shift2 = 0;
		} else {
			shift = 0;
		}
		
		for (int i = 0; i < ndir; i++) {
			weights((i + shift) % ndir, j) = pow(-1., (j+1))*pow(-1.0, int(pow(2,(j+1+shift2)) * i / ndir) + j);
			//weights(i, j) = pow(-1.0, int(4 * i / ndir));
		}

	}
}



void setWeightsCenter(int ndir, int nrings, double radius, Eigen::MatrixXf& weights) {
	weights.resize(ndir, nrings);
	weights.setZero();
		for (int i = 0; i < ndir; i++) {
			weights(i, 0) = 1.;
		}
	
}



void setWeightsOne(int ndir, int nrings, double radius, Eigen::MatrixXf& weights) {
	weights.resize(ndir, nrings);
	for (int j = 0; j < nrings; j++) {
		for (int i = 0; i < ndir; i++) {
			weights(i, j) = 1.;
		}
	}
}


void setWeights(int ndir, int nrings, double radius, double(*g)(double, double), Eigen::MatrixXf& weights) {
	weights.resize(ndir, nrings);
	double sigma = M_PI / 5.;
	double ratio = 5.;
	double x = 0.;
	double y = 0.;
	double r = 0.;
	double theta = 0.;
	for (int j = 0; j < nrings; j++) {
		for (int i = 0; i < ndir; i++) {
			theta = 2.*i*M_PI / ndir;
			r = (j + 1)*radius / nrings;
			weights((i)%ndir, j) = g(theta, r);
			//x = r*cos(theta);
			//y = r*sin(theta);
			//exp(-(x*x + ratio*ratio*y*y) / sigma);
		}
	}
}

// util
template< typename T>
T tensorAcces(const Eigen::Matrix<T, -1, -1>& M, const std::vector<int>& idx, const std::vector<int>& size) {
	int Idx = 0;
	int ndim = idx.size();
	for (int i = 0; i < ndim-1; i++) {
		Idx += idx[ndim - 1 - i];
		Idx *= size[ndim - 2 - i];
	}
	Idx += idx[ndim - 1 - i];
	return M.data()[Idx];
}

template< typename T>
T tensor3DAcces(const Eigen::Matrix<T, -1, -1>& M, int i, int j, int k, int n1, int n2) {
	return M.data()[n1*(k*n2+j) + i];
}

void circularConv(const Eigen::MatrixXf& weights, const Eigen::MatrixXf& Y, Eigen::MatrixXf& C, const bool oriented = false) {
	
	
	int ndir = weights.rows();
	int nrings = weights.cols();
	int nframes = ndir;
	if (!oriented) {
		nframes = 2 * ndir;
	}
	int nv = (Y.rows()*Y.cols())/(nframes*nrings);
	 
	C.resize(nv, nframes);
	C.setZero();
	float t = 0.0;
	for (int l = 0; l < ndir; l++) {
		for (int k = 0; k < nrings; k++) {
			for (int i = 0; i < nv; i++) {
				// direct frames
				for (int j = 0; j < ndir; j++) {
					
					//C(i, l) += weights((j - l + ndir) % ndir, k)*Y(nrings*(nframes*i + j) + k, 0);
					C(i, l) += weights((j - l + ndir) % ndir, k)*Y(nframes*(nrings*i + k) + j, 0);
					//C(i, j) += Y(nrings*(2 * ndir*i + j) + k, 0);
				}
				if (!oriented) {
					// reversed frames
					for (int j = 0; j < ndir; j++) {
						//C(i, l+ndir) += weights((j - l + ndir) % ndir, k)*tensor3DAcces(Y, i, j+ndir, k, nv, 2 * ndir);
						//C(i, l+ndir) += weights((j - l + ndir) % ndir, k)*Y((nv*(k * 2 * ndir + j + ndir) + i), 0);
						//C(i, l + ndir) += weights((j - l + ndir) % ndir, k)*Y(nrings*(nframes*i + j + ndir) + k, 0);
						C(i, l + ndir) += weights((j - l + ndir) % ndir, k)*Y(nframes*(nrings*i + k) + j + ndir, 0);
					}
				}
			}
		}
	}
	/*for (int i = 0; i < nv; i++) {
		for (int l = 0; l < ndir; l++) {
			for (int k = 0; k < nrings; k++) {
				for (int j = 0; j < ndir; j++) {
					C(i, l) += Y(nrings*(2 * ndir*i + j) + k, 0);
				}
			}
		}
	}*/
}

void testPopConvolution(int bp, int ndir, int nrings, double radius, geodesic::Mesh& myMesh, 
						Eigen::MatrixXf& C, const bool oriented = false) {
	
	
	int nv = myMesh.vertices().size();
	int nframes = ndir;
	if (!oriented) {
		nframes = 2 * ndir;
	}
	Eigen::MatrixXf F(nv, nframes);

	float val = 1.0;
	Eigen::MatrixXf dirac(nv, 1);
	dirac.setZero();
	dirac(bp, 0) = val;
	Eigen::MatrixXd D;
	frameBundlePullBack(dirac, F, nframes);

	// exponential average for soft bin max pooling
	Eigen::MatrixXf xExp(F.rows(), F.cols());
	Eigen::MatrixXf Exp(F.rows(), F.cols());
	double nu = 10.0;
	for (int i = 0; i < F.rows(); i++) {
		for (int j = 0; j < F.cols(); j++) {
			xExp(i, j) = F(i, j)*exp(nu*F(i, j)*F(i, j));
			Exp(i, j) = exp(nu*F(i, j)*F(i, j));
		}
	}


	PatchOperator P(myMesh);
	//double sd = 1.0;
	//GaussianModel model(radius*0.9, 1.0, ndir, nrings);
	//P.compute(model, false, radius, 0.0);
	P.compute(ndir, nrings, radius, oriented);
	
	
	Eigen::Matrix<float, -1, -1, Eigen::RowMajor> Fxexp(xExp);
	Eigen::Matrix<float, -1, -1, Eigen::RowMajor> Fexp(Exp);
	Eigen::Matrix<float, -1, -1, Eigen::RowMajor> F2(F);

	Eigen::Map<Eigen::RowVectorXf> fxexp(Fxexp.data(), Fxexp.size());
	Eigen::Map<Eigen::RowVectorXf> fexp(Fexp.data(), Fexp.size());
	Eigen::Map<Eigen::RowVectorXf> f2(F2.data(), F2.size());

	Eigen::MatrixXf Yxexp = P.getPatchOpMat()*(fxexp.transpose());
	Eigen::MatrixXf Yexp = P.getPatchOpMat()*(fexp.transpose());
	
	//Eigen::MatrixXf Y = P.getPatchOpMat()*(f2.transpose());
	Eigen::MatrixXf Y(Yexp.rows(), Yexp.cols());
	Y.setZero();
	for (int i = 0; i < Yexp.rows(); i++) {
		Y(i, 0) = Yxexp(i, 0) / (Yexp(i, 0)+std::numeric_limits<float>::min());
	}


	cout << Y.rows() << " - " << Y.cols() << endl;
	Eigen::MatrixXf weights(ndir, nrings);
	//setWeights(ndir, nrings, radius, &flatGaussian, weights);
	//setWeightsOne(ndir, nrings, radius, weights);
	//setWeightsCenter(ndir, nrings, radius, weights);
	//setWeightsCheckerboard(ndir, nrings, radius, weights);
	//setWeightsWindow248(ndir, radius, weights);
	setWeightsQuadrans(ndir, nrings, radius, weights);
	//gaussianFrameWeights(ndir, nrings, weights);
	circularConv(weights, Y, C, oriented);
	//Map<RowVectorXf> v2(M2.data(), M2.size());
	//cout << "v2:" << endl << v2 << endl;*/

}


bool patchOpTest(int bp, int ndir, int nrings, int frame, double radius, const bool oriented = false) {
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	if (!igl::readOFF("C:/Users/Adrien/Documents/shapes/sphere/sphere_2002_shuf.off", V, F))
	{
		cout << "failed to load mesh" << endl;
		return false;
	}


	int nv = V.rows();


	Eigen::MatrixXf C;
	std::vector<double> points;
	std::vector<unsigned> facets;
	convert(V, F, points, facets);
	geodesic::Mesh mesh;
	mesh.initialize_mesh_data(points, facets);
	GPC gpc(mesh);
	Eigen::MatrixXf ff(nv, 1);
	ff.setZero();
	//testConvolution(bp, ndir, radius, gpc, C);
	testPopConvolution(bp, ndir, nrings, radius, mesh, C, oriented);
	Synchronize(bp, frame, gpc, C, ff);

	gpc.compute(bp, 100.0);

	Eigen::MatrixXd f(nv, 1);

	for (int i = 0; i < nv; i++) {
		f(i, 0) = (double)(ff(i, 0));
	}

	int k = 0;
	Eigen::MatrixXd angle(gpc.getDist().size(), 1);
	Eigen::MatrixXd U1_approx(gpc.getDist().size(), 3);
	U1_approx.setZero();
	Eigen::MatrixXd U2_approx(gpc.getDist().size(), 3);
	U2_approx.setZero();
	Eigen::MatrixXd U1(gpc.getDist().size(), 3);
	U1.setZero();
	Eigen::MatrixXd U2(gpc.getDist().size(), 3);
	U2.setZero();
	Eigen::MatrixXd BaseDir(gpc.getDist().size(), 3);
	BaseDir.setZero();

	for (int i = 0; i < gpc.nbValid(); i++) {
		Eigen::Vector3d u;
		Eigen::Vector3d v;
		//Eigen::Vector3d w;
		int l = gpc.getValidIdx()[i];

		int u_id = gpc.getOneRingVertices(l)[0];
		int v_id = gpc.getOneRingVertices(l)[1];


		u = gpc.getVertexCoord(u_id);
		u -= gpc.getVertexCoord(l);
		u.normalize();

		v = gpc.getVertexCoord(v_id);
		v -= gpc.getVertexCoord(l);
		v.normalize();
		v -= v.dot(u)*u;
		v.normalize();
		double sz = 0.07;
		double angle1 = (2.*M_PI*angular_bin(gpc.getU1()[l], ndir)) / ndir;
		double angle2 = angle1 + M_PI / 2.;
		U1_approx.row(l) = sz*(cos(angle1)*u + sin(angle1)*v);
		U2_approx.row(l) = sz*(cos(angle2)*u + sin(angle2)*v);

		U1.row(l) = sz*(cos(gpc.getU1()[l])*u + sin(gpc.getU1()[l])*v);
		U2.row(l) = sz*(cos(gpc.getU2()[l])*u + sin(gpc.getU2()[l])*v);
		BaseDir.row(l) = sz*u;

		//U1.row(l) = gecAlgorithm.pTransport1(i);
	}
	for (int i = 0; i < gpc.getDist().size(); i++) {
		if (gpc.getState()[i]) {

			angle(i, 0) = gpc.getAngle()[i];

			//C(i, 0) = 1.0;
		}
		else {
			angle(i, 0) = 0.0;
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
			viewer.data.set_colors(f);
			//viewer.data.set_colors(angle);
			viewer.data.add_edges(V, V + 0.07*gpc.getNormals(), Eigen::RowVector3d(0, 0, 0));
			viewer.data.add_edges(V, V + U1, Eigen::RowVector3d(0, 0, 0));
			viewer.data.add_edges(V, V + U1_approx, Eigen::RowVector3d(255, 255, 255));
			//viewer.data.add_edges(V, V + BaseDir, Eigen::RowVector3d(255, 255, 255));
			return true;
		}
		}
	};
	viewer.callback_key_down(viewer, ' ', 0);
	viewer.core.show_lines = false;
	viewer.launch();//*/
	return true;

}

#endif // !PATCH_OP_TEST_H
