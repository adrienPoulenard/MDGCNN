#ifndef SHOT_DESCRIPTOR_H
#define SHOT_DESCRIPTOR_H

#include "mesh.h"
#include <string>

#include <stdexcept>
#include <cstddef>

#include <cassert>
#include <iostream>
#include <cstddef>
#include <Eigen/Eigen>
#include <Eigen/Eigenvalues>


using namespace std;


class invalid_mesh_descriptor : logic_error {
public:
	explicit invalid_mesh_descriptor() : std::logic_error("Exception invalid_mesh_descriptor caught.") {}
	invalid_mesh_descriptor(const std::string& msg) : std::logic_error("Exception invalid_mesh_descriptor caught: " + msg) {}
};

/**
 * Generic pointwise local shape descriptor for meshes.
 */
template <typename T>
class mesh_descriptor
{
protected:
	T point;

public:
	explicit mesh_descriptor() {}
	mesh_descriptor(const T& p) : point(p) {}
};

/**
 * Generic pointwise indexed vector descriptor.
 */
template <typename T>
struct vec_descriptor : public mesh_descriptor<int>
{
protected:
	std::vector<T> d;

public:
	typedef T value_type;

	explicit vec_descriptor() : mesh_descriptor<int>() {}
	explicit vec_descriptor(size_t n) : mesh_descriptor<int>() { d.resize(n, T(0)); }

	const T& operator()(size_t k) const { return d[k]; }
	T& operator()(size_t k) { return d[k]; }

	const T& at(size_t k) const { return d.at(k); }
	T& at(size_t k) { return d.at(k); }

	size_t size() const { return d.size(); }
	bool empty() const { return d.empty(); }
	void clear() { d.clear(); }
	void resize(size_t s) { d.resize(s); }
	void resize(size_t s, T v) { d.resize(s, v); }

	bool operator==(const vec_descriptor<T>& other) const
	{
		if (this == &other) return true;
		return d == other.d;
	}

	bool operator!=(const vec_descriptor<T>& other) const
	{
		if (this == &other) return false;
		return !(*this == other);
	}

	vec_descriptor<T> operator+(const vec_descriptor<T>& other) const
	{
		const size_t s = size();
		if (s != other.size())
			throw invalid_mesh_descriptor("Cannot sum different length descriptors.");

		vec_descriptor<T> sum;
		for (size_t k = 0; k < s; ++k) sum[k] = this->d[k] + other.d[k];

		return sum;
	}

	vec_descriptor<T> operator-(const vec_descriptor<T>& other) const
	{
		const size_t s = size();
		if (s != other.size())
			throw invalid_mesh_descriptor("Cannot sum different length descriptors.");

		vec_descriptor<T> sum;
		for (size_t k = 0; k < s; ++k) sum[k] = this->d[k] - other.d[k];

		return sum;
	}

	vec_descriptor<T>& operator+=(const vec_descriptor<T>& other)
	{
		const size_t s = size();
		if (s != other.size())
			throw invalid_mesh_descriptor("Cannot sum different length descriptors.");

		for (size_t k = 0; k < s; ++k) this->d[k] += other.d[k];

		return *this;
	}

	vec_descriptor<T>& operator-=(const vec_descriptor<T>& other)
	{
		const size_t s = size();
		if (s != other.size())
			throw invalid_mesh_descriptor("Cannot sum different length descriptors.");

		for (size_t k = 0; k < s; ++k) this->d[k] -= other.d[k];

		return *this;
	}

	template <typename S>
	vec_descriptor<T>& operator/=(const S& val)
	{
		const size_t s = size();
		for (size_t k = 0; k < s; ++k) this->d[k] /= T(val);
		return *this;
	}

	friend std::ostream& operator<<(std::ostream& s, const vec_descriptor& d)
	{
		s << d(0);
		string space = " ";
		for (size_t k = 1; k < d.size(); ++k) s << space << d(k);
		return s;
	}
};

namespace unibo {

#define pi 3.14159265359f
#define DEG_45_TO_RAD 0.78539816339744830961566084581988f
#define DEG_90_TO_RAD 1.5707963267948966192313216916398f
#define DEG_135_TO_RAD 2.3561944901923449288469825374596f
#define DEG_168_TO_RAD 2.7488935718910690836548129603696f

	template <typename Type> inline Type Min(Type a, Type b) { return (a <= b) ? a : b; }
	template <typename Type> inline Type Max(Type a, Type b) { return (a >= b) ? a : b; }

	inline bool areEquals(double val1, double val2, double zeroDoubleEps = 1e-6)
	{
		return (std::abs(val1 - val2) < zeroDoubleEps);
	}


	class shot : public vec_descriptor<float>
	{
		typedef vec_descriptor<float> super;

	public:
		typedef super::value_type value_type;

		explicit shot() : super(), radius(0.0f) {}
		explicit shot(size_t n) : super(n), radius(0.0f) {}

		float radius;
	};

	struct SHOTParams
	{
		float radius;			///< radius of the sphere that defines the local neighborhood
		float localRFradius;	///< radius of the support to be used for the estimation of the local RF
		int bins;				///< quantization bins for the cosine
		bool doubleVolumes;
		bool useInterpolation;
		bool useNormalization;
		int minNeighbors;

		SHOTParams()	// default values
		{
			radius = 15;
			localRFradius = radius;
			bins = 10;
			minNeighbors = 10;

			doubleVolumes = true;
			useInterpolation = true;
			useNormalization = true;
		}
	};

	class SHOTDescriptor
	{
	public:
		explicit SHOTDescriptor(const SHOTParams& params) : m_params(params)
		{
			if (m_params.doubleVolumes) m_k = 32;
			else m_k = 16; //number of onion husks
			m_descLength = m_k * (m_params.bins + 1);
		}

		void describe(mesh_t& data, int feat_index, shot& desc) const {
			desc.radius = m_params.radius;

			desc.resize(m_descLength, 0);
			vec3d<double>  ref_X, ref_Y, ref_Z;

			double sq_radius = m_params.radius*m_params.radius;
			double sqradius4 = sq_radius / 4;
			double radius3_4 = (m_params.radius * 3) / 4;
			double radius1_4 = m_params.radius / 4;
			double radius1_2 = m_params.radius / 2;

			int desc_index, step_index;

			int maxAngularSectors = 12;
			if (m_params.doubleVolumes) maxAngularSectors = 28;

			vector<int> neighs;
			vector<double> dists;

			const  vec3d<double> & centralPoint = data.get_vertex(feat_index);

			if (areEquals(m_params.localRFradius, m_params.radius))
			{
				data.nearest_neighbors_with_dist(feat_index, m_params.radius, neighs, dists);

				try {
					getSHOTLocalRF(data, feat_index, neighs, dists, m_params.localRFradius, ref_X, ref_Y, ref_Z);
				}
				catch (const std::exception& e) {
					//std::cout << "[WARNING] (" << feat_index << ") " << e.what() << std::endl;
					return;
				}
			}
			else
			{
				try {
					getSHOTLocalRF(data, feat_index, m_params.localRFradius, ref_X, ref_Y, ref_Z);
				}
				catch (const std::exception& e) {
					//std::cout << "[WARNING] (" << feat_index << ") " << e.what() << std::endl;
					return;
				}
				data.nearest_neighbors_with_dist(feat_index, m_params.radius, neighs, dists);
			}

			const int n_neighs = neighs.size();
			//std::cout << n_neighs << " neighbors" << std::endl;

			if (n_neighs < m_params.minNeighbors)
			{
				std::cout <<
					"[WARNING] Neighborhood has less than " << m_params.minNeighbors << " vertices. " <<
					"Aborting description of feature point " << feat_index << std::endl;
				return;
			}

			for (int j = 0; j < n_neighs; ++j)
			{
				const  vec3d<double>  q = data.get_vertex(neighs[j]) - centralPoint;

				const double distance = q.x*q.x + q.y*q.y + q.z*q.z;
				const double sqrtSqDistance = std::sqrt(distance);

				//Note: this should not happen since the reference point is assumed not to be in neighs
				if (areEquals(distance, 0.0))
					continue;

				const vec3d<double>& normal = data.get_vertex(neighs[j]).n;

				double cosineDesc = ref_Z.x*normal.x + ref_Z.y*normal.y + ref_Z.z*normal.z;
				if (cosineDesc > 1.0) cosineDesc = 1.0;
				else if (cosineDesc < -1.0) cosineDesc = -1.0;

				double xInFeatRef = dot_product(q, ref_X);
				double yInFeatRef = dot_product(q, ref_Y);
				double zInFeatRef = dot_product(q, ref_Z);

				// To avoid numerical problems afterwards
				if (areEquals(xInFeatRef, 0.0, 1E-30))
					xInFeatRef = 0;
				if (areEquals(yInFeatRef, 0.0, 1E-30))
					yInFeatRef = 0;
				if (areEquals(zInFeatRef, 0.0, 1E-30))
					zInFeatRef = 0;

				unsigned char bit4 = ((yInFeatRef > 0) || ((yInFeatRef == 0.0) && (xInFeatRef < 0))) ? 1 : 0;
				unsigned char bit3 = ((xInFeatRef > 0) || ((xInFeatRef == 0.0) && (yInFeatRef > 0))) ? !bit4 : bit4;

				desc_index = (bit4 << 3) + (bit3 << 2);

				if (m_params.doubleVolumes)
				{
					desc_index = desc_index << 1;

					if ((xInFeatRef * yInFeatRef > 0) || (xInFeatRef == 0.0))
						desc_index += (std::abs(xInFeatRef) >= std::abs(yInFeatRef)) ? 0 : 4;
					else
						desc_index += (std::abs(xInFeatRef) > std::abs(yInFeatRef)) ? 4 : 0;
				}

				desc_index += zInFeatRef > 0 ? 1 : 0;

				// 2 RADII
				//desc_index += (distance > sqradius4) ? 2 : 0;
				desc_index += (sqrtSqDistance > m_params.radius / 2.) ? 2 : 0;

				/* Bits:
						 0: positive (=1) or negative (=0) elevation
						 1: outer (=1) or inner (=0) husk

						 if (!double volumes)
						 2:

				   */

				double binDistance = ((1.0 + cosineDesc) * m_params.bins) / 2;

				step_index = binDistance < 0.0 ? ceil(binDistance - 0.5) : floor(binDistance + 0.5); // round(binDistance)
				const int volume_index = desc_index * (m_params.bins + 1);

				const double weight = 1.0;

				if (m_params.useInterpolation)
				{
					//Interpolation on the cosine (adjacent bins in the histogram)
					binDistance -= step_index;
					double intWeight = (1 - std::abs(binDistance));

					if (binDistance > 0)
						desc(volume_index + ((step_index + 1) % m_params.bins))
						+= binDistance * weight;
					else
						desc(volume_index + ((step_index - 1 + m_params.bins) % m_params.bins))
						+= -binDistance * weight;

					//Interpolation on the distance (adjacent husks)

					/*FIXME: crashes when desc_index is 0 or 1 (idx becomes < 0)*/
					if (sqrtSqDistance > radius1_2)
					{
						double radiusDistance = (sqrtSqDistance - radius3_4) / radius1_2;

						if (sqrtSqDistance > radius3_4)
							intWeight += 1 - radiusDistance;
						else
						{
							intWeight += 1 + radiusDistance;
							const int idx = (desc_index - 2) * (m_params.bins + 1) + step_index;
							try
							{
								desc.at(idx) += weight * (-radiusDistance);
							}
							catch (std::exception& e)
							{
								std::cout << "\nException caught: " << e.what() << ". Accessing el. " << idx << "/" << desc.size() << std::endl;
								throw;
							}
						}
					}
					else
					{
						const double radiusDistance = (sqrtSqDistance - radius1_4) / radius1_2;

						if (sqrtSqDistance < radius1_4)
							intWeight += 1 + radiusDistance;
						else
						{
							intWeight += 1 - radiusDistance;
							desc.at((desc_index + 2) * (m_params.bins + 1) + step_index) += weight * radiusDistance;
						}
					}

					//Interpolation on the inclination (adjacent vertical volumes)

					double inclinationCos = zInFeatRef / sqrtSqDistance;
					if (inclinationCos < -1.0) inclinationCos = -1.0;
					if (inclinationCos > 1.0) inclinationCos = 1.0;

					double inclination = std::acos(inclinationCos);

					assert(inclination >= 0.0 && inclination <= pi);

					if (inclination > DEG_90_TO_RAD || (areEquals(inclination, DEG_90_TO_RAD) && zInFeatRef <= 0))
					{
						double inclinationDistance = (inclination - DEG_135_TO_RAD) / DEG_90_TO_RAD;
						if (inclination > DEG_135_TO_RAD)
							intWeight += 1 - inclinationDistance;
						else
						{
							intWeight += 1 + inclinationDistance;
							assert((desc_index + 1) * (m_params.bins + 1) + step_index >= 0 && (desc_index + 1) * (m_params.bins + 1) + step_index < m_descLength);
							desc((desc_index + 1) * (m_params.bins + 1) + step_index) += weight * (-inclinationDistance);
						}

					}
					else
					{
						double inclinationDistance = (inclination - DEG_45_TO_RAD) / DEG_90_TO_RAD;
						if (inclination < DEG_45_TO_RAD)
							intWeight += 1 + inclinationDistance;
						else
						{
							intWeight += 1 - inclinationDistance;
							assert((desc_index - 1) * (m_params.bins + 1) + step_index >= 0 && (desc_index - 1) * (m_params.bins + 1) + step_index < m_descLength);
							desc((desc_index - 1) * (m_params.bins + 1) + step_index) += weight * inclinationDistance;
						}
					}

					//}

					if (yInFeatRef != 0.0 || xInFeatRef != 0.0)
					{
						//Interpolation on the azimuth (adjacent horizontal volumes)
						double azimuth = std::atan2(yInFeatRef, xInFeatRef);

						// Thanks to the fact that we forced the value of yInFeatRef and xInFeatRef at 0.0 above
						// we don't need this anymore. Actually, it is wrong to do it now, it raises the assertion on the azimuth distance.
						// // atan2 shouldn't return -PI but this implementation does :(
						/*if (areEquals(azimuth, -PST_PI, 1e-30))
							  azimuth = PST_PI;*/

						const int sel = desc_index >> 2;
						const double angularSectorSpan = (m_params.doubleVolumes) ? DEG_45_TO_RAD : DEG_90_TO_RAD;
						const double angularSectorStart = (m_params.doubleVolumes) ? -DEG_168_TO_RAD : -DEG_135_TO_RAD;

						double azimuthDistance = (azimuth - (angularSectorStart + angularSectorSpan * sel)) / angularSectorSpan;

						assert((azimuthDistance < 0.5 || areEquals(azimuthDistance, 0.5)) && (azimuthDistance > -0.5 || areEquals(azimuthDistance, -0.5)));

						azimuthDistance = Max(-0.5, Min(azimuthDistance, 0.5));

						if (azimuthDistance > 0)
						{
							intWeight += 1 - azimuthDistance;
							int interp_index = (desc_index + 4) % maxAngularSectors;
							assert(interp_index * (m_params.bins + 1) + step_index >= 0 && interp_index * (m_params.bins + 1) + step_index < m_descLength);
							desc(interp_index * (m_params.bins + 1) + step_index) += weight * azimuthDistance;
						}
						else
						{
							int interp_index = (desc_index - 4 + maxAngularSectors) % maxAngularSectors;
							assert(interp_index * (m_params.bins + 1) + step_index >= 0 && interp_index * (m_params.bins + 1) + step_index < m_descLength);
							intWeight += 1 + azimuthDistance;
							desc(interp_index * (m_params.bins + 1) + step_index) += weight * (-azimuthDistance);
						}

					}

					assert(volume_index + step_index >= 0 && volume_index + step_index < m_descLength);
					desc(volume_index + step_index) += weight * intWeight;
				}
				else
					desc(volume_index + step_index) += weight;

				assert(volume_index >= 0 && volume_index < (m_params.bins + 1)*m_k);

			} // next neighbor

			if (m_params.useNormalization)
			{
				double accNorm = 0;
				for (int j = 0; j < m_descLength; j++)
					accNorm += desc(j) * desc(j);

				accNorm = std::sqrt(accNorm);

				for (int j = 0; j < m_descLength; j++)
					desc(j) /= accNorm;
			}
		}

		int getDescriptorLength() const { return m_descLength; }

	private:
		SHOTParams m_params;
		int m_k; //number of onion husks
		int m_descLength;

		void getSHOTLocalRF(
			mesh_t& data,
			int p,
			double radius,
			vec3d<double>& X,
			vec3d<double>& Y,
			vec3d<double>& Z) const {

			vector<int> neighs; vector<double> dists;
			data.nearest_neighbors_with_dist(p, radius, neighs, dists);
			getSHOTLocalRF(data, p, neighs, dists, radius, X, Y, Z);
		}

		void getSHOTLocalRF(
			mesh_t& data,
			int p,
			const std::vector<int>& pts,
			const std::vector<double>& dists,
			double radius,
			vec3d<double>& X,
			vec3d<double>& Y,
			vec3d<double>& Z) const {
			const int np = pts.size();
			const  vec3d<double> & pt = data.get_vertex(p);

			// Weighted covariance matrix

			double sumw = 0.;

			Eigen::Matrix3d M;
			M.fill(0.);

			if (np < 5)
				throw std::logic_error("Not enough points for computing SHOT descriptor");

			for (int i = 0; i < np; ++i)
			{
				const double w = radius - dists[i];
				const  vec3d<double>  q = data.get_vertex(pts[i]) - pt;

				M(0, 0) += w * q.x * q.x;
				//M[0] += w * q.x * q.x;

				M(1, 1) += w * q.y * q.y;
				//M[4] += w * q.y * q.y;

				M(2, 2) += w * q.z * q.z;
				//M[8] += w * q.z * q.z;

				double tmp = w * q.x * q.y;
				M(0, 1) += tmp; M(1, 0) += tmp;
				//M[1] += tmp; M[3] += tmp;

				tmp = w * q.x * q.z;
				M(0, 2) += tmp; M(2, 0) += tmp;
				//M[2] += tmp; M[6] += tmp;

				tmp = w * q.y * q.z;
				M(1, 2) += tmp; M(2, 1) += tmp;
				//M[5] += tmp; M[7] += tmp;

				sumw += w;
			}
			M(0, 0) /= sumw; M(0, 1) /= sumw; M(0, 2) /= sumw;
			M(1, 0) /= sumw; M(1, 1) /= sumw; M(1, 2) /= sumw;
			M(2, 0) /= sumw; M(2, 1) /= sumw; M(2, 2) /= sumw;
			//for (int i=0; i<9; ++i) M[i] /= sumw;

			// Eigenvalue decomposition

			Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(M, Eigen::ComputeEigenvectors);
			Eigen::VectorXd eval = solver.eigenvalues();  // sorted in increasing order
			Eigen::Matrix3d evec = solver.eigenvectors(); // eigenvectors are normalized and stored as columns

														  //std::cout << "The eigenvalues of A are:" << std::endl << solver.eigenvalues() << std::endl;

														  //double evec[9], eval[3];
														  //if ( dgeev_driver(3, M, evec, eval) != 3 )
														  //   std::cout << "[WARNING] (" << p << ") Less than 3 eigenvalues" << std::endl;

														  //std::cout << eval[0] << " " << eval[1] << " " << eval[2] << std::endl;

														  // Sign disambiguation

														  /*
														  int x, y, z; // decreasing eigenvalue order
														  if (eval[0] > eval[1])
														  {
														  if (eval[0] > eval[2])
														  {
														  x = 0;
														  if (eval[1] > eval[2]) { y = 1; z = 2; }
														  else { y = 2; z = 1; }
														  }
														  else { x = 2; y = 0; z = 1; }
														  }
														  else
														  {
														  if (eval[1] > eval[2])
														  {
														  x = 1;
														  if (eval[0] > eval[2]) { y = 0; z = 2; }
														  else { y = 2; z = 0; }
														  }
														  else { x = 2; y = 1; z = 0; }
														  }
														  */

			int x = 2, y = 1, z = 0;

			if (eval[x] < eval[y] || eval[y] < eval[z])
				throw std::logic_error(
					"[ERROR] eigenvalues are not decreasing: ");// + ntos(eval[0]) + " " + ntos(eval[1]) + " " + ntos(eval[2]));

																// disambiguate x and z

			int posx = 0, posz = 0;

			X.x = evec(0, x); X.y = evec(1, x); X.z = evec(2, x);
			Z.x = evec(0, z); Z.y = evec(1, z); Z.z = evec(2, z);
			//X.x = evec[3*x + 0]; X.y = evec[3*x + 1]; X.z = evec[3*x + 2];
			//Z.x = evec[3*z + 0]; Z.y = evec[3*z + 1]; Z.z = evec[3*z + 2];

			//std::cout << "norms: " << X.x*X.x + X.y*X.y + X.z*X.z << " " << Z.x*Z.x + Z.y*Z.y + Z.z*Z.z << ", ";
			//std::cout << "inner: " << X.x*Z.x + X.y*Z.y + X.z*Z.z << std::endl;

			//normalize(X);
			//normalize(Z);

			for (int i = 0; i < np; ++i)
			{
				const  vec3d<double>  q = data.get_vertex(pts[i]) - pt;
				if (dot_product(q, X) >= 0) ++posx;
				if (dot_product(q, Z) >= 0) ++posz;
			}

			if (posx < np - posx) X = -X;
			if (posz < np - posz) Z = -Z;

			// get y

			Y = cross_product(Z, X);
		}
	};

} // namespace unibo

#endif // SHOT_DESCRIPTOR_H
