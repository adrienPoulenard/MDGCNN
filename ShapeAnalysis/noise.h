#pragma once
#ifndef NOISE_H
#define NOISE_H

#include "utils.h"

template <typename num_t>
void snow_noise(double intensity, std::vector<num_t>& f) {
	for (int i = 0; i < f.size(); i++) {
		f[i] += 2.0*intensity*(random_real() - 0.5);
	}
}

template <typename num_t>
void salt_noise(double t, std::vector<num_t>& f) {
	for (int i = 0; i < f.size(); i++) {
		if (random_real() > 1-t) {
			f[i] = std::max(1.0, f[i]);
		}
	}
}

template <typename num_t>
void perturb_bnd(double t, const Eigen::MatrixXi& F, std::vector<num_t>& f) {
	bool cond;
	for (int i = 0; i < F.rows(); i++) {
		cond = (f[F(i,0)] > 0.5) || (f[F(i, 1)] > 0.5) || (f[F(i, 2)] > 0.5);
		if (cond) {
			if (random_real() > 1 - t) {
				f[F(i, 0)] = std::max(1.0, f[F(i, 0)]);
			}
			if (random_real() > 1 - t) {
				f[F(i, 1)] = std::max(1.0, f[F(i, 1)]);
			}
			if (random_real() > 1 - t) {
				f[F(i, 2)] = std::max(1.0, f[F(i, 2)]);
			}
		}
	}
}

template <typename num_t>
void local_average(const Eigen::MatrixXi& F, std::vector<num_t>& f) {
	std::vector<num_t> f_tmp(f.size());
	std::fill(f_tmp.begin(), f_tmp.end(), 0.0);
	std::vector<int> nb_neigh(f.size());
	std::fill(nb_neigh.begin(), nb_neigh.end(), 0);
	double x;
	for (int i = 0; i < F.rows(); i++) {
		x = (f[F(i, 0)] + f[F(i, 1)] + f[F(i, 2)])/3.0;
		f_tmp[F(i, 0)] += x;
		f_tmp[F(i, 1)] += x;
		f_tmp[F(i, 2)] += x;
		nb_neigh[F(i, 0)]++;
		nb_neigh[F(i, 1)]++;
		nb_neigh[F(i, 2)]++;
	}
	for (int i = 0; i < f.size(); i++) {
		f[i] = f_tmp[i]/nb_neigh[i];
	}
}

template <typename num_t>
void local_average(const Eigen::MatrixXi& F, std::vector<num_t>& f, int nb_iter) {
	for (int i = 0; i < nb_iter; i++) {
		local_average(F, f);
	}
}


#endif // !NOISE_H
