

//#include <igl/viewer/Viewer.h>
#include <iostream>
#include <fstream>

#include <geodesic_algorithm_exact.h>
//#include <SFML/Graphics.hpp>
#include <igl/viewer/Viewer.h>
#include <igl/parula.h>
//#include <igl/read_triangle_mesh.h>
#include <igl/readOFF.h>
//#include <igl/readOFF.h>

#include "GPC.h"
//#include "display.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "patch_op.h"
#include "sync_conv.h"
#include "patch_op_test.h"



void convert_(const std::vector<double>& v, const std::vector<unsigned>& f, Eigen::MatrixXd& V, Eigen::MatrixXi& F) {
	V.resize(v.size() / 3, 3);
	for (unsigned i = 0; i < V.rows(); i++) {
		V(i, 0) = v[3 * i];
		V(i, 1) = v[3 * i + 1];
		V(i, 2) = v[3 * i + 2];
	}
	F.resize(f.size() / 3, 3);
	for (unsigned i = 0; i < F.rows(); i++) {
		F(i, 0) = f[3 * i];
		F(i, 1) = f[3 * i + 1];
		F(i, 2) = f[3 * i + 2];
	}
}

void diamond(Eigen::MatrixXd& V, Eigen::MatrixXi& F, int k, double r) {
	V.resize(k+2, 3);
	V.setZero();

	for (int i = 0; i < k; i++) {
		V.row(i) << cos(i * 2 * PI / k), sin(i * 2 * PI / k), 0.0;
	}
	V *= r;
	V.row(k) << 0.0, 0.0, 1.0;
	V.row(k+1) << 0.0, 0.0, -1.0;
	V *= 0.01;
	cout << V << endl;

	F.resize(2*k, 3);
	for (int i = 0; i < k; i++) {
		F.row(2*i) << (i + 1) % k, k, i;
		F.row(2*i+1) << k+1, (i + 1) % k, i;
	}

}

void simplex(Eigen::MatrixXd& V, Eigen::MatrixXi& F) {

	V.resize(4, 3);
	V << 0.0, 0.0, 0.0,
		1.0, 0.0, 0.0,
		0.0, 1.0, 0.0,
		1.0, 1.0, 0.0;

	/*V << 0.0, 0.0, 0.5,
	1.0, 0.0, 0.0,
	-1.0, 1.0, 0.0,
	-1.0, -1.0, 0.0;

	//V *= 1.0 / sqrt(2);
	//cout << V << endl; */


	F.resize(2, 3);
	F << 0, 1, 2,
		1, 2, 3;
		//1, 3, 0;
	    //1, 2, 3;

	/*F.resize(2, 3);
	F << 0, 2, 1,
	1, 3, 0;*/

	/*F.resize(2, 3);
	F << 0, 3, 2,
	2, 1, 0;*/

	/*F.resize(2, 3);
	F << 2, 0, 3,
	3, 1, 2;*/

	/*F.resize(1, 3);
	F << 1, 0, 2;*/

}







int main() {
	
	/*std::vector<double> points;
	std::vector<unsigned> faces;
	std::string myfile = "C:/Users/Adrien/Documents/cpp libs/geodesic_cpp_03_02_2008/flat_triangular_mesh.txt";

	geodesic::Mesh mesh;
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;




	if (!igl::readOFF("C:/Users/adrien/Documents/shape_collections/spheres/sphered.off", V, F))
	{
		cout << "failed to load mesh" << endl;
		exit(666);
	}
	

	
	convert( V,  F,  points,  faces);
	cout << points.size() << endl;
	mesh.initialize_mesh_data(points, faces);		//create internal mesh data structure including edges
	GPC gecAlgorithm(mesh);

	// choose basepoint
	//unsigned source_vertex_index = 0;
	unsigned source_vertex_index = 47;
	//unsigned source_vertex_index = 750;
	geodesic::SurfacePoint source(&mesh.vertices()[source_vertex_index]);
	//geodesic::SurfacePoint source(&mesh.edges()[source_edge_index]);
	// compute
	double radius = 1000.0;
	int n = 500;
	radius = gecAlgorithm.compute(source_vertex_index, radius, ORIENTATION_GLOBAL, 0.000001);
	//radius = gecAlgorithm.compute(source_vertex_index, 100, 0, 0.000001);
	double eps = radius / n;
	gecAlgorithm.displayDuration();
	
	


	






	

	//cout << "dist " << dist.size() << C.rows() << endl;

	int k = 0;
	Eigen::MatrixXd C(gecAlgorithm.getDist().size(), 1);
	Eigen::MatrixXd U1(gecAlgorithm.getDist().size(), 3);
	U1.setZero();
	Eigen::MatrixXd U2(gecAlgorithm.getDist().size(), 3);
	U2.setZero();
	Eigen::MatrixXd BaseDir(gecAlgorithm.getDist().size(), 3);
	BaseDir.setZero();
	
	for (int i = 0; i < gecAlgorithm.nbValid(); i++) {
		Eigen::Vector3d u;
		Eigen::Vector3d v;
		//Eigen::Vector3d w;
		int l = gecAlgorithm.getValidIdx()[i];
		
		int u_id = gecAlgorithm.getOneRingVertices(l)[0];
		int v_id = gecAlgorithm.getOneRingVertices(l)[1];
		
		
		u = gecAlgorithm.getVertexCoord(u_id);
		u -= gecAlgorithm.getVertexCoord(l);
		u.normalize();

		v = gecAlgorithm.getVertexCoord(v_id);
		v -= gecAlgorithm.getVertexCoord(l);
		v.normalize();
		v -= v.dot(u)*u;
		v.normalize();
		double sz = 10.0;
		U1.row(l) = sz*(cos(gecAlgorithm.getU1()[l])*u + sin(gecAlgorithm.getU1()[l])*v);
		U2.row(l) = sz*(cos(gecAlgorithm.getU2()[l])*u + sin(gecAlgorithm.getU2()[l])*v);
		BaseDir.row(l) = sz*u;

		//U1.row(l) = gecAlgorithm.pTransport1(i);
	}
	for (int i = 0; i < gecAlgorithm.getDist().size(); i++) {
		if (gecAlgorithm.getState()[i]) {



			C(i, 0) = gecAlgorithm.getAngle()[i];
			
			//C(i, 0) = 1.0;
		}
		else {
			C(i, 0) = 0.0;
		}
	}

	//
	//Eigen::SparseMatrix<double>* pbMat;

	
	

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
			viewer.data.add_edges(V, V + U1, Eigen::RowVector3d(0, 0, 0));
			viewer.data.add_edges(V, V + BaseDir, Eigen::RowVector3d(255, 255, 255));
			return true;
		}
		}
	};
	viewer.callback_key_down(viewer, ' ', 0);
	viewer.core.show_lines = false;
	viewer.launch();*/

	/*unsigned int nb_dir = 8;
	unsigned int nb_rings = 5;
	//double r = M_PI / 4.0;
	double r = M_PI / 4.0;
	double s_t = (nb_dir / M_PI)*(nb_dir / M_PI)*log(2.0);
	double s_r = (nb_rings / (2.0*r))*(nb_rings / (2.0*r))*log(2.0);
	double margin_ratio = 0.2;
	double delta = r / nb_rings;
	double sd = 0.5;
	//cout << "sd " << sd << endl;
	GaussianModel gm(r, sd, nb_dir, nb_rings);
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;

	PatchOperator Pop("C:/Users/Adrien/Documents/shapes/sphere/sphere_7002.off");



	//cout << "margin ratio = " << (r + margin / r) << endl;
	//cout << "angular standard deviation = " << 2*nb_rings / (2 * M_PI*sqrt(s_t)) << endl;
	//cout << "radial standard deviation = " << 1. / (r*sqrt(s_r)) << endl;
	Pop.compute(gm, false, r*(1.+margin_ratio), 0.0001);
	Pop.savePatchOp("C:/Users/Adrien/Documents/SGCNN/patch_op_tests/test.txt");
	//Pop.compute(gm, false, 0.1, 0.0001);*/
	//unitTest(100, 16, 28, M_PI/2.0);
	patchOpTest(100, 16, 3, 0, M_PI / 6.0);
	system("pause");
}




/*
int main(int argc, char **argv)
{


	std::vector<double> points;
	std::vector<unsigned> faces;
	std::string myfile = "C:/Users/Adrien/Documents/cpp libs/geodesic_cpp_03_02_2008/flat_triangular_mesh.txt";
	bool success = geodesic::read_mesh_from_file("C:/Users/Adrien/Documents/cpp libs/geodesic_cpp_03_02_2008/flat_triangular_mesh.txt", points, faces); // in cst and simple func
	if (!success)
	{
		std::cout << "something is wrong with the input file" << std::endl;
		system("pause");
		return 0;
	}

	geodesic::Mesh mesh;
	mesh.initialize_mesh_data(points, faces);		//create internal mesh data structure including edges

	geodesic::GeodesicAlgorithmExact algorithm(&mesh);	//create exact algorithm for the mesh

	unsigned source_vertex_index = 0;//(argc == 2) ? 0 : atol(argv[2]);

	geodesic::SurfacePoint source(&mesh.vertices()[source_vertex_index]);		//create source 
	std::vector<geodesic::SurfacePoint> all_sources(1, source);					//in general, there could be multiple sources, but now we have only one

	if (false)	//target vertex specified, compute single path
	{
		unsigned target_vertex_index = atol(argv[3]);
		geodesic::SurfacePoint target(&mesh.vertices()[target_vertex_index]);		//create source 

		std::vector<geodesic::SurfacePoint> path;	//geodesic path is a sequence of SurfacePoints

		bool const lazy_people_flag = false;		//there are two ways to do exactly the same
		if (lazy_people_flag)
		{
			algorithm.geodesic(source, target, path); //find a single source-target path
		}
		else		//doing the same thing explicitly for educational reasons
		{
			double const distance_limit = geodesic::GEODESIC_INF;			// no limit for propagation
			std::vector<geodesic::SurfacePoint> stop_points(1, target);	//stop propagation when the target is covered
			algorithm.propagate(all_sources, distance_limit, &stop_points);	//"propagate(all_sources)" is also fine, but take more time because covers the whole mesh

			algorithm.trace_back(target, path);		//trace back a single path 
		}

		print_info_about_path(path);
		for (unsigned i = 0; i<path.size(); ++i)
		{
			geodesic::SurfacePoint& s = path[i];
			std::cout << s.x() << "\t" << s.y() << "\t" << s.z() << std::endl;
		}
	}
	else		//target vertex is not specified, print distances to all vertices
	{
		algorithm.propagate(all_sources);	//cover the whole mesh

		for (unsigned i = 0; i<mesh.vertices().size(); ++i)
		{
			geodesic::SurfacePoint p(&mesh.vertices()[i]);

			double distance;
			unsigned best_source = algorithm.best_source(p, distance);		//for a given surface point, find closets source and distance to this source

			std::cout << distance << " ";		//print geodesic distance for every vertex
		}
		std::cout << std::endl;
	}

	system("pause");
	return 0;
}

*/