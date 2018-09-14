#pragma once
#ifndef CONCAT_DESCS_H
#define CONCAT_DESCS_H

#include "utils.h"

void concat_descs(int nv, int ndescs1, int ndescs2, const std::string& desc1, const std::string& desc2, const std::string& desc3) {
	// list desc1 directory
	std::vector< std::string > names;
	getFilesList_(desc1, ".txt", names);
	Eigen::MatrixXd d1(nv, ndescs1);
	Eigen::MatrixXd d2(nv, ndescs2);
	Eigen::MatrixXd d3(nv, ndescs1 + ndescs2);

	std::vector<int> permutation;
	for (int i = 0; i < names.size(); i++) {
		// permute labels 
		load_matrix_(desc1 + "/" + names[i] + ".txt", d1);
		load_matrix_(desc2 + "/" + names[i] + ".txt", d2);
		concat_matrices(d1, d2, d3);
		save_matrix_(desc3 + "/" + names[i] + ".txt", d3);
	}
}


#endif // !CONCAT_DESCS_H
