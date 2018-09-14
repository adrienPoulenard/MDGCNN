#pragma once
#ifndef CONNECTIVITY_H
#define CONNECTIVITY_H
#include <geodesic_mesh.h>
#include <iostream>
using namespace std;


// Faces->adjacent_faces  not implemented !!!!
// complete build_adjacencies definition in geodesic::Mesh definition
#define PI 3.14159265359



inline bool contains(geodesic::edge_pointer e, geodesic::vertex_pointer v) {
	return((e->adjacent_vertices()[0]->id() == v->id()) ||
		(e->adjacent_vertices()[1]->id() == v->id()));
}
inline int containsVertex(geodesic::base_pointer obj, int i) {
	for (unsigned int j = 0; j < obj->adjacent_vertices().size(); j++) {
		if (obj->adjacent_vertices()[j]->id() == i) {
			return j;
		}
	}
	return -1;
}
inline int containsEdge(geodesic::face_pointer f, int i) {
	for (unsigned j = 0; j < 3; j++) {
		if (f->adjacent_edges()[j]->id() == i) {
			return j;
		}
	}
	return -1;
}
inline bool isAdjacent(geodesic::face_pointer f1, geodesic::face_pointer f2) {
	if (f1->id() != f2->id()) {
		for (int i = 0; i < 3; i++) {
			if (containsEdge(f2, f1->adjacent_edges()[i]->id()) > -1) {
				return true;
			}
		}
	}
	return false;
}

inline int IdAt(geodesic::base_pointer obj, geodesic::face_pointer f) {
	for (unsigned int i = 0; i < obj->adjacent_faces().size(); i++) {
		if (obj->adjacent_faces()[i]->id() == f->id()) {
			return i;
		}
	}
	return -1;
}


inline int IdAt(geodesic::base_pointer obj, geodesic::edge_pointer e) {
	for (unsigned int i = 0; i < obj->adjacent_edges().size(); i++) {
		if (obj->adjacent_edges()[i]->id() == e->id()) {
			return i;
		}
	}
	return -1;
}

inline int IdAt(geodesic::base_pointer obj, geodesic::vertex_pointer v) {
	for (int i = 0; i < obj->adjacent_vertices().size(); i++) {
		if (obj->adjacent_vertices()[i]->id() == v->id()) {
			return i;
		}
	}
	return -1;
}

/*
inline void set1RingOrder(geodesic::vertex_pointer basepoint,
	std::vector<int>& faces_idx,
	std::vector<int>& edges_idx,
	std::vector<int>& vertices_idx,
	std::vector<int>& v_in_faces_idx,
	std::vector<double>& incident_angles,
	std::vector<double>& angles, 
	double& total_angle) {

	int bp_id = basepoint->id();
	int nb_neigh = basepoint->adjacent_edges().size();
	int nb_adj_faces = basepoint->adjacent_faces().size();
	//cout << "nb_neigh " << nb_neigh << endl;
	//std::vector<int> faces_idx;
	faces_idx.resize(nb_adj_faces);
	v_in_faces_idx.resize(nb_adj_faces);
	incident_angles.resize(nb_adj_faces);
	angles.resize(nb_neigh);
	edges_idx.resize(nb_neigh);
	vertices_idx.resize(nb_neigh);

	geodesic::vertex_pointer v;
	geodesic::face_pointer f_tmp;
	geodesic::face_pointer f_next;
	geodesic::edge_pointer e;
	geodesic::edge_pointer e_tmp;
	geodesic::edge_pointer e_next;

	e_tmp = basepoint->adjacent_edges()[0];
	f_tmp = e_tmp->adjacent_faces()[0];

	for (unsigned i = 0; i < nb_neigh; i++) {
		if (basepoint->adjacent_edges()[0]->adjacent_faces().size() == 1) {
			e_tmp = basepoint->adjacent_edges()[i];
			f_tmp = e_tmp->adjacent_faces()[0];
		}
	}

	faces_idx[0] = f_tmp->id();
	edges_idx[0] = e_tmp->id();
	if (e_tmp->adjacent_vertices()[0]->id() != bp_id) {
		v = e_tmp->adjacent_vertices()[0];
	}
	else {
		v = e_tmp->adjacent_vertices()[1];
	}
	vertices_idx[0] = v->id();
	v_in_faces_idx[0] = IdAt(f_tmp, basepoint);
	incident_angles[0] = f_tmp->corner_angles()[v_in_faces_idx[0]];

	int k = 0;
	for (unsigned i = 1; i < nb_adj_faces; i++) {
		if (f_tmp->adjacent_edges().size() != 3) {
			cout << "aaaaaaaaaaa " << f_tmp->id() << " " << f_tmp->adjacent_edges().size() << endl;
		}
		for (unsigned j = 0; j < 3; j++) {
			e = f_tmp->adjacent_edges()[j];
			if ((e->id() != e_tmp->id()) & 
				((e->adjacent_vertices()[0]->id() == bp_id) ||
				 (e->adjacent_vertices()[1]->id() == bp_id))) {
				if (e->adjacent_faces()[0]->id() != f_tmp->id()) {
					f_next = e->adjacent_faces()[0];
				}
				else {
					f_next = e->adjacent_faces()[1];
				}
				
				faces_idx[i] = f_next->id();
				edges_idx[i] = e->id();
				
				if (e->adjacent_vertices()[0]->id() != bp_id) {
					v = e->adjacent_vertices()[0];
				}
				else {
					v = e->adjacent_vertices()[1];
				}
				vertices_idx[i] = v->id();
				
				v_in_faces_idx[i] = IdAt(f_next, basepoint);
				incident_angles[i] = f_next->corner_angles()[v_in_faces_idx[i]];
				e_next = e;
				if(bp_id == 0){
					k++;
				}
			}
			//break;
		} // for (unsigned j = 0; j < 3; j++) {
		f_tmp = f_next;
		e_tmp = e_next;
	}

	angles[0] = 0.0;
	total_angle = incident_angles[nb_adj_faces - 1];
	for (unsigned i = 0; i < nb_adj_faces-1; i++) {
		angles[i + 1] = angles[i] + incident_angles[i];
		total_angle += incident_angles[i];
	}
	double ratio = (2.0*PI / total_angle);
	if (nb_neigh > nb_adj_faces) {
		angles[nb_adj_faces] = angles[nb_adj_faces - 1] + incident_angles[nb_adj_faces - 1];
	}
	for (unsigned i = 0; i < nb_adj_faces; i++) {
		incident_angles[i] *= ratio;
		angles[i] *= ratio;
	}


}
*/

inline bool set1RingOrder(geodesic::vertex_pointer basepoint,
	std::vector<int>& faces_idx,
	std::vector<int>& edges_idx,
	std::vector<int>& vertices_idx,
	std::vector<int>& v_in_faces_idx,
	std::vector<double>& incident_angles,
	std::vector<double>& angles,
	double& total_angle,
	bool direct = true) {

	int bp_id = basepoint->id();
	int deg = basepoint->adjacent_vertices().size();
	int nb_adj_faces = basepoint->adjacent_faces().size();

	if (deg != nb_adj_faces) {
		return false;
	}
	faces_idx.resize(nb_adj_faces);
	v_in_faces_idx.resize(nb_adj_faces);
	incident_angles.resize(nb_adj_faces);
	angles.resize(deg);
	edges_idx.resize(deg);
	vertices_idx.resize(deg);

	geodesic::vertex_pointer v;
	geodesic::face_pointer f_tmp;
	geodesic::face_pointer f_next;
	geodesic::edge_pointer e;
	geodesic::edge_pointer e_tmp;
	geodesic::edge_pointer e_next;

	e_tmp = basepoint->adjacent_edges()[0];
	if (direct) {
		f_tmp = e_tmp->adjacent_faces()[0];
	}
	else {
		f_tmp = e_tmp->adjacent_faces()[1];
	}

	for (unsigned i = 0; i < deg; i++) {
		faces_idx[i] = f_tmp->id();
		edges_idx[i] = e_tmp->id();
		if (e_tmp->adjacent_vertices()[0]->id() != bp_id) {
			v = e_tmp->adjacent_vertices()[0];
		}
		else {
			v = e_tmp->adjacent_vertices()[1];
		}
		vertices_idx[i] = v->id();

		v_in_faces_idx[i] = IdAt(f_tmp, basepoint);
		incident_angles[i] = f_tmp->corner_angles()[v_in_faces_idx[i]];
		// next
		for (unsigned j = 0; j < 3; j++) {
			e = f_tmp->adjacent_edges()[j];
			if ((containsVertex(e, bp_id) != -1) & (e->id() != e_tmp->id())) {
				e_next = e;
			}
		}
		e_tmp = e_next;
		if (e_tmp->adjacent_faces()[0]->id() != f_tmp->id()) {
			f_next = e_tmp->adjacent_faces()[0];
		}
		else {
			f_next = e_tmp->adjacent_faces()[1];
		}
		f_tmp = f_next;
	}
	// angles
	total_angle = 0.0;
	for (unsigned i = 0; i < nb_adj_faces; i++) {
		total_angle += incident_angles[i];
	}
	double ratio = (2.0*PI) / total_angle;
	for (unsigned i = 0; i < nb_adj_faces; i++) {
		incident_angles[i] *= ratio;
	}
	angles[0] = 0.0;
	for (unsigned i = 1; i < deg; i++) {
		angles[i] = angles[i-1] + incident_angles[i-1];
	}

	return true;
}

#endif // !CONNECTIVITY_H
