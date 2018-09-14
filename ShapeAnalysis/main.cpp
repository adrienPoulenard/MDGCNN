#include <iostream>
#include <fstream>

#include <igl/viewer/Viewer.h>
#include <igl/parula.h>
//#include <igl/read_triangle_mesh.h>
#include <igl/readOFF.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "cube_segmentation.h"
#include "faust.h"
#include "permute_dataset.h"
#include "create_dataset.h"
#include "visualize_patch.h"
#include "signed_sphere.h"
#include "cross_sphere.h"
#include "concat_descs.h"
#include "visualize.h"
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include "polygons_on_spheres.h"
#include "relative_orientation_test.h"
#include "local_pca.h"
#include "hessian_filter.h"
#include "local_frames.h"
#include "voxelization.h"
#include "cifar10_sph.h"
#include "img.h"
#include "img2cylinder.h"
#include "cifar10_cylinder.h"
#include "unit_tests.h"
#include "grid_mesh.h"
#include "shot.h"
#include "human_seg_sig_2017_toric_cover.h"
#include "landmarks.h"
#include "load_mesh.h"
#include "PSB_dataset.h"
#include "format_conv.h"


using namespace std;

int main() {
	/* initialize random seed: */
	srand(time(NULL));

	/*Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	//std::string path = "C:/Users/Adrien/Documents/FAUST_shapes_off/tr_reg_070.off";
	//std::string path = "C:/Users/Adrien/Documents/shapes/Faust_permuted/shapes/tr_reg_070.off";
	//std::string path = "C:/Users/Adrien/Documents/shapes/sphere/sphere_3002.off";
	//std::string path = "C:/Users/Adrien/Documents/shapes/sphere_reconstruct_permuted/sphere_3002.off";
	//std::string path = "C:/Users/Adrien/Documents/shapes/Faust_permuted/shapes/tr_reg_087.off";
	std::string path = "C:/Users/Adrien/Documents/FAUST_shapes_off/tr_reg_087.off";
	if (!igl::readOFF(path, V, F))
	{
		cout << "failed to load mesh" << endl;
		exit(666);
	}

	float rx = 1.0;
	float ry = 1.0;
	float rz = 1.0;
	int nx = 10; //4
	int ny = 14;
	int nz = 1;
	float dx = 0.00; //0.07
	float eps = 0.000001;
	box3d bnd_box;
	get_bounding_box(V, bnd_box);
	cout << "Dx " << bnd_box.x_max - bnd_box.x_min << endl;
	cout << "Dy " << bnd_box.y_max - bnd_box.y_min << endl;
	std::vector< int > labels;
	voxelSegmentation(V, bnd_box.x_min-eps, bnd_box.x_max+dx+eps,
		bnd_box.y_min-eps, bnd_box.y_max+eps,
		bnd_box.z_min-eps, bnd_box.z_max+eps,
		nx, ny, nz,
		labels);
		
	Eigen::MatrixXd C(V.rows(), 1);
	//load_vector_<int>("C:/Users/Adrien/Documents/Datasets/Faust/labels_x=12_y=26_75/tr_reg_070.txt", labels);
	std::vector<double> v;
	//signedSphere_(path, v);
	//crossSphere_(path, v);
	//load_matrix_("C:/Users/Adrien/Documents/Datasets/spheres/reconstructed/reconstructed.txt", V);
	load_matrix_("C:/Users/Adrien/Documents/Datasets/Faust/reconstructed/reconstructed.txt", V);
	//visualize_patch(0, 0.1, path, v);
	for (int i = 0; i < V.rows(); i++) {
		//C(i, 0) = v[i];
		//C(i, 0) = (1024*labels[i])%97;
		C(i, 0) = 0.0;
		//cout << labels[i] << endl;
	}
	//C(0, 0) = 10.0;

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

	//faust(0.066666);

	/*permuteDataset("E:/Users/Adrien/Documents/shapes/FAUST",
		"E:/Users/Adrien/Documents/shapes/Faust_permuted/shapes",
		"E:/Users/Adrien/Documents/shapes/Faust_permuted/permutations",
		"E:/Users/Adrien/Documents/shapes/Faust_permuted/reverse_permutations");//*/

	/*permuteLabels("C:/Users/Adrien/Documents/Datasets/Faust/labels_original_x=10_y=14_35",
		"C:/Users/Adrien/Documents/Datasets/Faust/labels_x=10_y=14_35",
		"C:/Users/Adrien/Documents/shapes/Faust_permuted/permutations"); // */

	/*refShapePermuteXYZ("C:/Users/Adrien/Documents/FAUST_shapes_off/tr_reg_000.off",
		"C:/Users/Adrien/Documents/Datasets/Faust/signals",
		"C:/Users/Adrien/Documents/shapes/Faust_permuted/permutations"); // */

	/*computeFilterMaper(0.15, 4, 16,
		"C:/Users/Adrien/Documents/shapes/Faust_permuted/shapes",
		"C:/Users/Adrien/Documents/Datasets/Faust");//*/

	/*null_signal("C:/Users/Adrien/Documents/shapes/Faust_permuted/shapes",
		"C:/Users/Adrien/Documents/Datasets/Faust/null_signal", 6890);*/

	/*permuteDataset("C:/Users/Adrien/Documents/shapes/sphere_reconstruct", 
		"C:/Users/Adrien/Documents/shapes/sphere_reconstruct_permuted", 
		"C:/Users/Adrien/Documents/shapes/trash");*/

	/*crossSphere("C:/Users/Adrien/Documents/shapes/sphere_reconstruct_permuted",
		"C:/Users/Adrien/Documents/Datasets/spheres");//*/

	/*computeFilterMaper(M_PI/12.0, 2, 16,
		"C:/Users/Adrien/Documents/shapes/sphere_reconstruct_permuted",
		"C:/Users/Adrien/Documents/Datasets/spheres");//*/

	/*faust_left_right("C:/Users/Adrien/Documents/shapes/Faust_permuted/shapes", 
		"C:/Users/Adrien/Documents/Datasets/Faust/left_right",
		"C:/Users/Adrien/Documents/shapes/Faust_permuted/permutations");*/
	/*faustReconstruct3d("C:/Users/Adrien/Documents/Datasets/Faust/3d_reconstruct_70",
		"C:/Users/Adrien/Documents/shapes/Faust_permuted/permutations");*/

	/*concat_descs(6890, 15, 2, 
		"C:/Users/Adrien/Documents/Datasets/Faust/matlab_descs_15wks",
		"C:/Users/Adrien/Documents/Datasets/Faust/left_right",
		"C:/Users/Adrien/Documents/Datasets/Faust/15wks_left_right");*/
	/*dotedSphere("C:/Users/Adrien/Documents/shapes/sphere_reconstruct_permuted/sphere_3002.off",
	"C:/Users/Adrien/Documents/Datasets/spheres/dirac_test/sphere_3002_dot.txt");*/
	/*visualize("C:/Users/Adrien/Documents/shapes/sphere_reconstruct_permuted/sphere_3002.off",
		"C:/Users/Adrien/Documents/Datasets/spheres/dirac_test/sphere_3002_dot_conv.txt"); //*/
	
	/*generate_permuted_spheres(1100, "sphere_2002",
		"C:/Users/Adrien/Documents/shapes/sphere", 
		"C:/Users/Adrien/Documents/shapes/sphere_2002");*/


	/*rand_polygon_test("C:/Users/Adrien/Documents/shapes/sphere_2002/sphere_2002_0.off");*/

	/*sphere_polygons_labels("C:/Users/Adrien/Documents/shapes/sphere_2002",
						   "C:/Users/Adrien/Documents/Datasets/star_spheres");*/

	/*visualize("C:/Users/Adrien/Documents/shapes/sphere_2002/sphere_2002_3.off",
		"C:/Users/Adrien/Documents/Datasets/star_spheres/labels/sphere_2002_3.txt");

	visualize("C:/Users/Adrien/Documents/shapes/sphere_2002/sphere_2002_3.off",
		"C:/Users/Adrien/Documents/Datasets/star_spheres/signals/sphere_2002_3.txt");//*/
	
	//visualize_patch(0, M_PI / 10., "C:/Users/Adrien/Documents/shapes/sphere_2002/sphere_2002_1.off");
	
	
	/*computeFilterMaper(M_PI / 15.0, 2, 16,
		"C:/Users/Adrien/Documents/shapes/sphere_2002",
		"C:/Users/Adrien/Documents/Datasets/star_spheres");*/
	
	//train_test_txt("C:/Users/Adrien/Documents/Datasets/star_spheres", 600, 150);

	/*computeFilterMaper(0.1, 3, 16,
		"C:/Users/Adrien/Documents/FAUST_shapes_off",
		"C:/Users/Adrien/Documents/Datasets/Faust non permuted");*/

	/*non_permuted_labels(6890,
	"C:/Users/Adrien/Documents/FAUST_shapes_off",
	"C:/Users/Adrien/Documents/Datasets/Faust non permuted/labels");*/

	/*non_permuted_target_signal("C:/Users/Adrien/Documents/FAUST_shapes_off", 
		"C:/Users/Adrien/Documents/Datasets/Faust non permuted/3d_reconstruct_70");*/

	//cout << visualize_lines("C:/Users/Adrien/Documents/shapes/sphere_2002/sphere_2002_1.off", 0*M_PI/10.) << endl;

	/*local_patterns_("C:/Users/Adrien/Documents/shapes/sphere_2002",
		"C:/Users/Adrien/Documents/Datasets/star_spheres/patterns_orth_signal",
		"C:/Users/Adrien/Documents/Datasets/star_spheres/patterns_orth_labels", 0 * M_PI / 10.);*/
	
	/*visualize("C:/Users/Adrien/Documents/shapes/sphere_2002/sphere_2002_1.off", 
		"C:/Users/Adrien/Documents/Datasets/star_spheres/patterns_orth_signal/sphere_2002_1.txt"); //*/

	//double radius = 0.1;
	//test_grid_hessian("C:/Users/Adrien/Documents/FAUST_shapes_off/tr_reg_070.off", radius);
	//gaussian_test("C:/Users/Adrien/Documents/FAUST_shapes_off/tr_reg_070.off", 700, radius);
	//test_kd_tree("C:/Users/Adrien/Documents/FAUST_shapes_off/tr_reg_070.off", 700, radius);
	//test_local_hessian_paterns("C:/Users/Adrien/Documents/shapes/sphere_2002/sphere_2002_1.off", 0, radius);
	//test_local_pca_paterns("C:/Users/Adrien/Documents/FAUST_shapes_off/tr_reg_070.off", 0, radius);
	//test_local_pca_paterns("C:/Users/Adrien/Documents/FAUST_shapes_off/tr_reg_070.off", 1, radius);
	//test_local_pca_paterns("C:/Users/Adrien/Documents/FAUST_shapes_off/tr_reg_070.off", 2, radius);
	//test_local_hessian_paterns("C:/Users/Adrien/Documents/FAUST_shapes_off/tr_reg_070.off", 3, radius);//*/
	
	//test_vesselness("C:/Users/Adrien/Documents/FAUST_shapes_off/tr_reg_070.off");
	
	//test_rgb_to_sh("C:/Users/Adrien/Documents/shapes/sphere", "sphere_3002");
	//test_rgb_to_cylinder("C:/Users/Adrien/Documents/shapes/cylinder", "cylinder_2000");
	
	/*visualizeRGB("C:/Users/adrien/Documents/shapes/sphere", "sphere_3002",
		"C:/Users/adrien/Documents/Datasets/cifar10_spheres/test/test3.txt");*/

	//visualize("C:/Users/adrien/Documents/shapes/sphere/sphere_3002.off");
    /*cifar10sphere("C:/Users/Adrien/Documents/cpp_libs/cifar-10-master/cifar-10/cifar-10-batches-bin",
	"C:/Users/adrien/Documents/shapes/sphere", "sphere_3002",
	"C:/Users/adrien/Documents/Datasets/cifar10_spheres/signals",
	"C:/Users/adrien/Documents/Datasets/cifar10_spheres/labels",
	50000,
	10000,
	5, 5,
	2.0);
	system("pause");//*/

	/*cifar10cylinder("C:/Users/adrien/Documents/cpp libs/cifar-10-master/cifar-10/cifar-10-batches-bin",
	"C:/Users/adrien/Documents/shapes/cylinder", "cylinder_2000",
	"C:/Users/adrien/Documents/Datasets/cifar10_cylinder/signals",
	"C:/Users/adrien/Documents/Datasets/cifar10_cylinder/labels",
	50000,
	10000,
	5, 5,
	0.0);
	system("pause");//*/

	//print_labels(100, false);

	/*std::vector<int> img_idx(30);
	for (int i = 0; i < 30; i++) {
		img_idx[i] = 100 * i;
	}
	RGB_UV_sphere("C:/Users/Adrien/Documents/shapes/sphere", 
	"sphere_3002",
	img_idx, "C:/Users/Adrien/Desktop/screens phd/cifar_uv", 
	"C:/Users/Adrien/Documents/cpp_libs/cifar-10-master/cifar-10/cifar-10-batches-bin");//*/
	//test_img("C:/Users/Adrien/Desktop/screens phd/cifar_uv");

	//shift_test("C:/Users/adrien/Documents/shapes/sphere/sphere_3002.off", 0.5);
	//sphere2cylinder("C:/Users/adrien/Documents/shapes/sphere/sphere21k.off");
	/*std::vector<double> r(4);
	r[0] = 0.9443*1 / 10.;
	r[1] = 0.9443*0.75 / 10.;
	r[2] = 0.9443*0.85 / 10.;
	r[3] = 0.9443*1.25 / 10.;
reducedCylinders("C:/Users/adrien/Documents/shapes/sphere/sphere21k.off",
	"C:/Users/adrien/Documents/shapes/cylinder", r);//*/
	// visualize("C:/Users/Adrien/Documents/shapes/cylinder/cylinder_2000.off");
	

	/*reduce_shape(0.5, "sphere",
		"C:/Users/Adrien/Documents/shapes/sphere/sphere_3002.off", 
		"C:/Users/Adrien/Documents/shapes/sphere");//*/

	//test_gpc_interpolation("C:/Users/adrien/Documents/shapes/sphere/sphere_1002.off", 123, 2.*M_PI / 10., 2, 8);
	
	// test_gpc_interpolation("C:/Users/Adrien/Documents/shapes/cylinder/cylinder_2000.off", 40, 0.1, 2, 4);
	
	
	//test_gpc_interpolation("E:/Users/Adrien/Documents/shapes/sphere/sphere_3002.off", 0, (M_PI*1.8) / 32., 2, 8);
	//test_gpc_interpolation("C:/Users/Adrien/Documents/Datasets/grid_mesh/grid/grid_38x38.off", 39 * 19 + 19, 1.8, 2, 8);

	//

	//test_bnd("C:/Users/Adrien/Documents/shapes/grid/grid.off");
	//
	//save_grid_mesh(19, 19, "C:/Users/Adrien/Documents/shapes/grid/grid.off", 1.0, 1.0);
	//test_gpc_interpolation("C:/Users/Adrien/Documents/shapes/grid/grid.off", 20 * 11 + 10, 2.*1.8, 2, 8);
	//test_pt_id("C:/Users/Adrien/Documents/shapes/grid/grid.off", 41 * 20 + 20);
	// visualize("C:/Users/Adrien/Documents/shapes/grid/grid.off");
	/*grid("C:/Users/Adrien/Documents/Datasets/grid_mesh/grid.off",
	"C:/Users/Adrien/Documents/shapes/grid", "grid");//*/
	
	/*sphereImg("E:/Users/Adrien/Documents/Datasets/sphere",
	"E:/Users/Adrien/Documents/shapes/sphere",
	"sphere_3002", 4.0, 32, (M_PI*2.5)/32.);//*/
	//test_gpc_interpolation("E:/Users/Adrien/Documents/shapes/sphere/sphere_3002.off", 0, (M_PI*2.5) / 32., 2, 8);

	/*visualizeRGB("E:/Users/Adrien/Documents/shapes/sphere", "sphere_3002",
	"E:/Users/Adrien/Documents/Keras/Gcnn/unit tests/cifar10_2355_sphere_3002_w=32.txt");//*/

	/*grid("C:/Users/adrien/Documents/Datasets/grid_mesh", 1.8, 2, 8, 4.0,
	"C:/Users/adrien/Documents/Datasets/grid_mesh/grid", 38, 38, 1.0, 1.0, 4.0);//*/
	
    /*visualize("C:/Users/Adrien/Documents/Datasets/grid_mesh/grid/grid_38x38.off",
	"C:/Users/Adrien/Documents/Keras/Gcnn/unit tests/dirac_0_grid_38x38.txt");//*/

	/*visualizeRGB("C:/Users/adrien/Documents/Datasets/grid_mesh/grid", "grid_38x38",
	"C:/Users/adrien/Documents/Keras/Gcnn/unit tests/cifar10_2355_grid_38x38.txt");//*/
	
	//test_gpc_interpolation("E:/Users/Adrien/Documents/shapes/Faust_permuted/shapes/tr_reg_000.off", 500, 0.05, 2, 16);
	/*faust_left_right("C:/Users/adrien/Documents/shapes/Faust", 
	"C:/Users/adrien/Documents/Datasets/Faust/labels/left_right");
	
	faust3d("C:/Users/adrien/Documents/shapes/Faust",
		"C:/Users/adrien/Documents/Datasets/Faust/signals/global_3d");
	
	faust_train_test("C:/Users/adrien/Documents/shapes/Faust", 
		"C:/Users/adrien/Documents/Datasets/Faust");//*/

	/*faust("E:/Users/Adrien/Documents/shapes/Faust", 
		"E:/Users/Adrien/Documents/Datasets/Faust", 
		4.0, 0.05, 2, 8, 2);//*/


	/*compute_shot("E:/Users/Adrien/Documents/shapes/Faust",
		"E:/Users/Adrien/Documents/Datasets/Faust/signals/shot_1_bin_24", 1, 24.0);//*/

	/*compute_shot("E:/Users/Adrien/Documents/Datasets/PSB/shapes",
		"E:/Users/Adrien/Documents/Datasets/PSB/descs/shot_1_bin_24", 1, 24.0);//*/

	
	/*compute_shot("E:/Users/Adrien/Documents/shapes/Faust_permuted/shapes",
		"E:/Users/Adrien/Documents/Datasets/Faust_permuted/signals/shot_1_bin", 1);//*/

	/*human_seg_sig2017_toric_cover_off("E:/Users/Adrien/Documents/Datasets/sig17_seg_benchmark",
		"E:/Users/Adrien/Documents/Datasets/SIG2017_toriccov_seg_bench", true);//*/
	
	//test_gpc_interpolation("E:/Users/Adrien/Documents/shapes/Faust/tr_reg_000.off", 0, 0.05, 2, 8);
	/*test_gpc_interpolation("E:/Users/Adrien/Documents/Datasets/SIG2017_toriccov_seg_bench/meshes/train/faust/tr_reg_000.off", 
		0, 0.1, 2, 8);//*/

	/*test_gpc_interpolation("E:/Users/Adrien/Documents/Datasets/SIG2017_toriccov_seg_bench/meshes/train/scape/mesh000.off",
		12011, 0.1, 2, 8);*/

	/*test_gpc_interpolation("C:/Users/adrien/Documents/Datasets/SIG2017_toriccov_seg_bench/meshes/test/shrec/shrec_12.off",
		2473, 0.07, 2, 8);//*/
	
	/*human_seg_sig2017_dataset("E:/Users/Adrien/Documents/Datasets/SIG2017_toriccov_seg_bench", 
		"E:/Users/Adrien/Documents/Datasets/SIG2017_toriccov_seg_bench", 0.1, 2, 8);*/

	/*visualize("C:/Users/adrien/Documents/Datasets/SIG2017_toriccov_seg_bench/meshes/test/shrec/shrec_13.off",
		"C:/Users/adrien/Documents/Datasets/SIG2017_toriccov_seg_bench/segs/test/shrec/shrec_13.txt");

	visualize("C:/Users/adrien/Documents/Datasets/SIG2017_toriccov_seg_bench/meshes/test/shrec/shrec_13.off",
		"C:/Users/adrien/Documents/Datasets/SIG2017_toriccov_seg_bench/preds_async_1/shrec_13.txt");//*/
	
	/*PrepareDataset("C:/Users/adrien/Documents/Datasets/ruqis_regression/val/shapes",
	"C:/Users/adrien/Documents/Datasets/ruqis_regression/val",
	4.0, 0.05, 2, 8, 2);//*/
	
	/*test_gpc_interpolation("C:/Users/adrien/Documents/Datasets/ruqis_regression/train/shapes/1.off",
		1000, 0.05, 2, 8);//*/

	/*visualize3D("E:/Users/Adrien/Documents/shapes/Faust/tr_reg_000.off", 
		"E:/Users/Adrien/Documents/Datasets/Faust/results/shape_matching_preds/sync/tr_reg_098.txt");//*/

	//visualize_landmark(5459, 0.12, "E:/Users/Adrien/Documents/shapes/Faust/tr_reg_000.off");
	
	/*std::vector<int> idx(1);
	idx[0] = 5459;
	landmarks_on_dataset(idx, 0.12, "E:/Users/Adrien/Documents/shapes/Faust",
		"E:/Users/Adrien/Documents/Datasets/Faust/signals/right_hand");*/

	/*std::vector<int> idx(1);
	//idx[0] = 5459;
	//idx[1] = 6765;
	//idx[2] = 5126;
	//idx[3] = 5284;
	//idx[3] = 4530;
	idx[0] = 1326;
	//idx[5] = 414;
	//idx[6] = 2212;
	//idx[7] = 3365;
	//idx[8] = 1736;
	//idx[10] = 1881;
	//idx[9] = 1046;

	//idx[0] = 3028;
	//idx[1] = 3158;

	visualize_landmarks(idx, 0.05, "E:/Users/Adrien/Documents/shapes/Faust/tr_reg_000.off");

	landmarks_on_dataset(idx, 0.05, "E:/Users/Adrien/Documents/shapes/Faust",
		"E:/Users/Adrien/Documents/Datasets/Faust/signals/sparse_signal_1");//*/

	/*prepare_psb("E:/Users/Adrien/Documents/shapes/PSB_4000", 
				"E:/Users/Adrien/Documents/Datasets/PSB");//*/

	//test_gpc_interpolation("E:/Users/Adrien/Documents/Datasets/PSB/shapes/Airplane_61.off", 0, 0.12, 2, 8);

	//test_gpc_interpolation("E:/Users/Adrien/Documents/shapes/PSB_4000/Human/shapes/2.off", 1200, 0.05, 2, 8);
	//test_validity("E:/Users/Adrien/Documents/shapes/PSB_4000/Vase/shapes");
	/*SIG_17_shapes("C:/Users/adrien/Documents/Datasets/SIG2017_toriccov_seg_bench", 
		"C:/Users/adrien/Documents/Datasets/SIG 17");//*/
	/*test_gpc_interpolation("C:/Users/adrien/Documents/Datasets/SIG 17/meshes/adobe_FemaleFitA_tri_fixed.off", 
		150, 0.1, 2, 8);*/
	/*PrepareDataset("C:/Users/adrien/Documents/Datasets/SIG 17/meshes",
	"C:/Users/adrien/Documents/Datasets/SIG 17",
	4.0, 0.05, 2, 8, 2);//*/
	/*compute_shot("C:/Users/adrien/Documents/Datasets/SIG 17/meshes",
	"C:/Users/adrien/Documents/Datasets/SIG 17/descs/shot_1_bin_12", 1, 12.0);//*/

	/*remeshed_faust("E:/Users/Adrien/Documents/Datasets/Remeshed Faust/forAdrien/FAUST/vtx_5k",
	"E:/Users/Adrien/Documents/Datasets/Remeshed Faust");//*/

	/*test_gpc_interpolation("E:/Users/Adrien/Documents/Datasets/Remeshed Faust/meshes/tr_reg_000.off",
		150, 0.1, 2, 8); //*/

	/*PrepareDataset("E:/Users/adrien/Documents/Datasets/Remeshed Faust/meshes",
		"E:/Users/adrien/Documents/Datasets/Remeshed Faust",
		4.0, 0.1, 2, 8, 2);//*/

	/*compute_shot("C:/Users/adrien/Documents/Datasets/Faust_5k/shapes",
		"C:/Users/adrien/Documents/Datasets/Faust_5k/descs/shot_1_bin_6", 1, 6.0);//*/
	
	/*singular_psb("E:/Users/Adrien/Documents/shapes/PSB_4000", 
		"E:/Users/Adrien/Documents/shapes/PSB_4000_singular");*/
	
	/*test_gpc_interpolation("E:/Users/Adrien/Documents/shapes/PSB_4000_singular_fixed/Armadillo/shapes/284_.off",
		150, 0.1, 2, 8); //*/

	/*display_GPC_transport("C:/Users/adrien/Documents/shapes/sphere/sphere_5002.off", 
		0, M_PI/4.0);*/

	/*test_gpc_interpolation("C:/Users/adrien/Documents/shapes/sphere/sphere_5002.off",
		0, M_PI / 4.0, 2, 8);*/

	/*cout << is_vertex_manifold("E:/Users/Adrien/Documents/shapes/PSB_4000_singular_fixed/Armadillo/shapes/284") << endl;
	cout << is_edge_manifold("E:/Users/Adrien/Documents/shapes/PSB_4000_singular_fixed/Armadillo/shapes/284.off") << endl;*/

	//test_PSB_validity("E:/Users/Adrien/Documents/shapes/PSB_simplified_segmentation_benchmark_5k2");
	
	/*PSB_labels_conversion("E:/Users/Adrien/Documents/shapes/PSB_full",
		"E:/Users/Adrien/Documents/shapes/PSB_5k"); //*/

	/*visualize_labels("E:/Users/Adrien/Documents/shapes/PSB_5k/Plier/shapes/210.off",
		"E:/Users/Adrien/Documents/shapes/PSB_5k/Plier/labels/210.txt");//*/

	/*prepare_psb("E:/Users/Adrien/Documents/shapes/PSB_5k",
	"E:/Users/Adrien/Documents/Datasets/PSB");//*/
	
	//test_gpc_interpolation("E:/Users/Adrien/Documents/Datasets/PSB/shapes/Plier_217.off", 1751, 0.30, 2, 8);

	/*PrepareDataset("E:/Users/adrien/Documents/Datasets/PSB/shapes",
		"E:/Users/adrien/Documents/Datasets/PSB",
		4.0, 0.12, 2, 8, 2);//*/
	
	//check_numeric_files("E:/Users/adrien/Documents/Datasets/PSB/transported_angles");

	/*compute_shot("E:/Users/adrien/Documents/Datasets/PSB/shapes",
		"E:/Users/adrien/Documents/Datasets/PSB/descs/shot_1_bin_9", 1, 6.0);//*/

	/*remesh_faust("C:/Users/Adrien/Documents/shapes/Faust",
		"C:/Users/adrien/Documents/Datasets/Faust_5k/labels",
		"C:/Users/adrien/Documents/Datasets/Faust_5k/shapes");//*/
	
	//test_gpc_interpolation("E:/Users/Adrien/Documents/Datasets/Faust_5k/shapes/tr_reg_000.off", 1751, 0.10, 2, 8);

	/*PrepareDataset("C:/Users/adrien/Documents/Datasets/Faust_5k/shapes",
		"C:/Users/adrien/Documents/Datasets/Faust_5k",
		4.0, 0.10, 2, 8, 2);//*/

	/*compute_shot("C:/Users/adrien/Documents/Datasets/Faust_5k/shapes",
		"C:/Users/adrien/Documents/Datasets/Faust_5k/descs/shot_1_bin_12", 1, 12.0);//*/

	/*visualize_labels("C:/Users/adrien/Documents/Datasets/SIG2017_toriccov_seg_bench/meshes/test/shrec/shrec_6.off",
		"C:/Users/adrien/Documents/Datasets/SIG2017_toriccov_seg_bench/results_sig_17_seg/preds_3D_sync_noBN/shrec_6.txt");*/
	
	//test_gpc_interpolation("C:/Users/adrien/Documents/Datasets/ruqis_regression_2/For_Adrien/SCAPE_8_poses/Shape1.off", 1751, 0.06, 2, 8);
	/*PrepareDataset("C:/Users/adrien/Documents/Datasets/ruqis_regression_2/shapes",
		"C:/Users/adrien/Documents/Datasets/ruqis_regression_2",
	4.0, 0.06, 2, 8, 3);//*/

	//check_numeric_files("C:/Users/adrien/Documents/Datasets/SIG2017_toriccov_seg_bench/patch_ops/train/faust/contributors_weights");
	
	/*convert_to_labelized_pointcloud_normals("C:/Users/adrien/Documents/Datasets/SIG 17/meshes", 
		"C:/Users/adrien/Documents/Datasets/SIG 17/labels",
		"C:/Users/adrien/Documents/Datasets/SIG 17 seg pointcloud normals/sig17_seg_shapes");//*/

	/*convert_to_labelized_pointcloud_normals("C:/Users/adrien/Documents/Datasets/Faust_5k/shapes",
		"C:/Users/adrien/Documents/Datasets/Faust_5k/labels",
		"C:/Users/adrien/Documents/Datasets/Faust_5k pointcloud normals/Faust_5k_shapes");*/

	/*pointnet_dataset_split("C:/Users/adrien/Documents/Datasets/SIG 17/train.txt", 
	"C:/Users/adrien/Documents/Datasets/SIG 17 seg pointcloud normals/train_test_split/shuffled_train_file_list.json",
	"sig17_seg_shapes");*/

	/*pointnet_dataset_split("C:/Users/adrien/Documents/Datasets/Faust_5k/train.txt",
		"C:/Users/adrien/Documents/Datasets/Faust_5k pointcloud normals/train_test_split/shuffled_train_file_list.json",
		"Faust_5k_shapes");//*/

convert_to_labelized_desc("C:/Users/adrien/Documents/Datasets/Faust_5k/descs/shot_1_bin_12",
	5000, 64,
	"C:/Users/adrien/Documents/Datasets/Faust_5k/labels",
	"C:/Users/adrien/Documents/Datasets/Faust_5k pointcloud normals/Faust_5k_shot12");

	system("pause");
}