# MDGCNN
Multi Directional Geodesic Convolutional Neural Networks

### Introduction
This code implements the Multi Directional Geodesic Convolutional Neural Networks (MDGCNN) described in our 
"Multi-directional Geodesic Neural Networks via Equivariant Convolution" article. 
(or MDGCNN) uses layers of directional convolution followed by an angular max pooling operator to transform 
input signals on meshes into point-wise or global predictions in segmentation or classification tasks.
Directional convolution is based on local windows at each vertex. To the windows consist of weighted contributors 
together with parallel transport of angle from their origin vertex.

### Instalation
The code is divided in two parts: A C++ part for data preprocessing and formatting the data 
to be used in the python to train MDGCNN. The C++ is mostly based on header only libraries and does not require to compile them. 
The entry point for the C++ is the main.cpp file in the ShapeAnalysis folder. 
The header dependencies are contained in the PatchOperator and Include folders.
To run the code you will also need to install following libraries:

- Eigen: http://eigen.tuxfamily.org/index.php?title=Main_Page
- libigl: https://github.com/libigl/libigl
- glew: http://glew.sourceforge.net
- GLFW: https://www.glfw.org/
- nanoflann: https://github.com/jlblancoc/nanoflann

and link the following ".lib" files

- opengl32.lib
- glu32.lib
- glfw3.lib
- glew32s.lib

form glew and glfw.

The python part is based on Keras with a tensorflow backend. Visit

https://www.tensorflow.org/install/

and 

https://keras.io/#installation

for more details on the instalation.

### How to use ?

To define directional convolution over a triangle mesh our python code requires 3 tensors:
And index tensor to store the indices of the vertices contributing to each local window,
a barycentric weights tensor storing the weights of the contributors and an angle transport tensor 
storing the parallel transport of the angular coordinate of the central vertex to its window's contributors.

To define pooling we need two tensors: A parent index tensor which, for each vertex of the reduced mesh
stores the index of the closest vertex in the original mesh and an angular shift tensor to transfer 
local angular coordinates from the original mesh to the reduced one.

Given a directory with shapes stored as .off files the function "PrepareDataset" in "ShapeAnalysis/create_dataset.h"
computes the above tensors for every shape. The function "PrepareDataset" can compute tensors to define multiple 
convolution and pooling layers with varying window size and number of vertices.

void PrepareDataset(const std::string& shapes_path,
	const std::string& dataset_path,
	double ratio_, double radius_, int nrings_, int ndirs_, int n = 3)
	
- "shapes_path" must point to the directory containing shapes as .off files
- "dataset_path" points to the target directory where we want to store the tensors
- "ratio_" is the ratio (<=1.0) betweenn the number of vertices in two consecutive layers
- "radius_" is the window radius of the first layer  
- "nrings_" is the number of radial bins of the windows
- "ndirs_" is the number of angular bins of the windows
- "n" indicates the number of convolution layers to compute (n-1) pooling layers.

The target directory "dataset_path" must contain the following subfolders structure:

- bin_contributors
- contributors_weights
- transported_angles
- parent_vertices
- labels

To train MDGCNN for segmentation / or matching tasks, use the function 'heterogeneous_dataset' 
in 'MDGCNN/train_network.py'.

heterogeneous_dataset(task,
					  num_filters,					  
					  train_list,
                      val_list,
                      test_list,
                      train_preds_path,
                      train_patch_op_path,
                      train_desc_paths,
                      val_preds_path,
                      val_patch_op_path,
                      val_desc_paths,
                      test_preds_path,
                      test_patch_op_path,
                      test_desc_paths,
                      radius,
                      nrings,
                      ndirs,
                      ratio,
                      nepochs,
                      num_classes,
					  sync_mode='radial_sync',
                      nresblocks_per_stack=2,
                      batch_norm=False,
                      global_3d=True)

Returns: the training, validation and testing data generators and the trained model.

Parameters:
- task: the type of task, "segmentation", "classification", "regression".
- num_filters: the number of filters to be used on the first layer
- train_list: path to a txt files contaning a list of shapes to be used for training
- val_list: path to a txt files contaning a list of shapes to be used for validation
- test_list: path to a txt files contaning a list of shapes to be used for testing
- train_preds_path: path to the ground truth result for the training shapes.
- train_patch_op_path: path to the conv/pool tensors for the training shapes.
- train_desc_paths: path to the input signals for the training shapes
- val_preds_path: path to the ground truth result for the validation shapes.
- val_patch_op_path: path to the conv/pool tensors for the validation shapes.
- val_desc_paths: path to the input signals for the validation shapes
- test_preds_path: path to the ground truth result for the test shapes.
- test_patch_op_path: path to the conv/pool tensors for the test shapes.
- test_desc_paths: path to the input signals for the test shapes
- radius: list of successive windows radii for each resnet stack (depends on the computed tensors)
- nrings: number of radial bins for each resnet stack
- ndirs: number of angular bins for each resnet stack
- ratio: ratio between the number of vertices before and after pooling layers
- nepochs: the number of epochs to train
- num_classes: number of parts for segmentation, classes for classification 
and dimensionality for regression.
- sync_mode: "radial_sync" to run MDGCNN or "async" to run the GCNN analogue.
- nresblocks_per_stack: number of residial blocks per stack
- batch_norm: batch normalization
- global_3d: data augmentation for 3D coordinates by applying random sacling and rotations.
	  


