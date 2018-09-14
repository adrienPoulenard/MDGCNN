# -*- coding: utf-8 -*-
# Script to write registrations as obj files
# Copyright (c) [2015] [Gerard Pons-Moll]

from argparse import ArgumentParser
import os
from os import mkdir
# from os.path import join, exists
import h5py
import sys

sids_list = ['50002', '50004', '50007', '50009', '50020',
            '50021', '50022', '50025', '50026', '50027']
sids_gender_list = ['m', 'f', 'm', 'm', 'f',
                    'f', 'f', 'f', 'm', 'm']
sid_seq_list = []

# 50002 (male)
sid_seq_list.append(['chicken_wings',
'hips',
'jiggle_on_toes',
'jumping_jacks',
'knees',
'light_hopping_loose',
'light_hopping_stiff',
'one_leg_jump',
'one_leg_loose',
'punching',
'running_on_spot',
'shake_arms',
'shake_hips',
'shake_shoulders'])
# 50004 (female)
sid_seq_list.append(['chicken_wings',
'hips',
'jiggle_on_toes',
'jumping_jacks',
'knees',
'light_hopping_loose',
'light_hopping_stiff',
'one_leg_jump',
'one_leg_loose',
'punching',
'running_on_spot_bugfix',
'shake_arms',
'shake_hips',
'shake_shoulders'])

# 50007 (male)
sid_seq_list.append(['chicken_wings',
'jiggle_on_toes',
'jumping_jacks',
'knees',
'light_hopping_loose',
'light_hopping_stiff',
'one_leg_jump',
'one_leg_loose',
'punching',
'running_on_spot',
'shake_arms',
'shake_hips',
'shake_shoulders'])

# 50009 (male)
sid_seq_list.append(['chicken_wings',
'hips',
'jiggle_on_toes',
'jumping_jacks',
'light_hopping_loose',
'light_hopping_stiff',
'one_leg_jump',
'one_leg_loose',
'punching',
'running_on_spot',
'shake_hips'])

# 50020 (female)
sid_seq_list.append(['chicken_wings',
'hips',
'jiggle_on_toes',
'knees',
'light_hopping_loose',
'light_hopping_stiff',
'one_leg_jump',
'one_leg_loose',
'personal_move',
'punching',
'running_on_spot',
'shake_arms',
'shake_hips',
'shake_shoulders'])

# 50021 (female)
sid_seq_list.append(['chicken_wings',
'hips',
'knees',
'light_hopping_stiff',
'one_leg_jump',
'one_leg_loose',
'punching',
'running_on_spot',
'shake_arms',
'shake_hips',
'shake_shoulders'])

# 50022 (female)
sid_seq_list.append(['hips',
'jiggle_on_toes',
'jumping_jacks',
'knees',
'light_hopping_loose',
'light_hopping_stiff',
'one_leg_jump',
'one_leg_loose',
'punching',
'running_on_spot',
'shake_arms',
'shake_hips',
'shake_shoulders'])

# 50025 (female)
sid_seq_list.append(['chicken_wings',
'hips',
'jiggle_on_toes',
'knees',
'light_hopping_loose',
'light_hopping_stiff',
'one_leg_jump',
'one_leg_loose',
'punching',
'running_on_spot',
'shake_arms',
'shake_hips',
'shake_shoulders'])

# 50026 (male)
sid_seq_list.append(['chicken_wings',
'hips',
'jiggle_on_toes',
'jumping_jacks',
'knees',
'light_hopping_loose',
'light_hopping_stiff',
'one_leg_jump',
'one_leg_loose',
'punching',
'running_on_spot',
'shake_arms',
'shake_hips',
'shake_shoulders'])

# 50027 (male)
sid_seq_list.append(['hips',
'jiggle_on_toes',
'jumping_jacks',
'knees',
'light_hopping_loose',
'light_hopping_stiff',
'one_leg_jump',
'one_leg_loose',
'punching',
'running_on_spot',
'shake_arms',
'shake_hips',
'shake_shoulders'])

seq_set = set([])
for i in range(len(sid_seq_list)):
    seq_set = seq_set.union(set(sid_seq_list[i]))

common_seq = seq_set
for i in range(len(sid_seq_list)):
    common_seq = common_seq.intersection(set(sid_seq_list[i]))


print(seq_set)

print(common_seq)


def write_mesh_as_obj(fname, verts, faces):
    with open(fname, 'w') as fp:
        for v in verts:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in faces + 1:  # Faces are 1-based, not 0-based in obj files
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))


def write_mesh_as_off(fname, verts, faces):
    with open(fname, 'w') as fp:
        fp.write('OFF\n')
        fp.write('%d %d %d\n' % (len(verts), len(faces), 0))
        for v in verts:
            fp.write('%f %f %f\n' % (v[0], v[1], v[2]))
        for f in faces:
            fp.write('%d %d %d %d\n' % (3, f[0], f[1], f[2]))


def save_dfaust_object(registrations, seq, sid, tdir, ext='off', frame=None, save_name=None):
    # Subject ids
    sids = ['50002', '50004', '50007', '50009', '50020',
            '50021', '50022', '50025', '50026', '50027']
    # Sequences available for each subject id are listed in scripts/subjects_and_sequences.txt

    parser = ArgumentParser(description='Save sequence registrations as obj')
    parser.add_argument('--path', type=str, default=registrations,
                        help='dataset path in hdf5 format')
    parser.add_argument('--seq', type=str, default=seq,
                        help='sequence name')
    parser.add_argument('--sid', type=str, default=sid, choices=sids,
                        help='subject id')
    parser.add_argument('--tdir', type=str, default=tdir,
                        help='target directory')
    args = parser.parse_args()

    sidseq = args.sid + '_' + args.seq
    with h5py.File(args.path, 'r') as f:
        if sidseq not in f:
            print('Sequence %s from subject %s not in %s' % (args.seq, args.sid, args.path))
            f.close()
            sys.exit(1)
        verts = f[sidseq].value.transpose([2, 0, 1])
        faces = f['faces'].value

    # tdir = os.path.join(args.tdir, sidseq)
    if not os.path.exists(tdir):
        mkdir(tdir)

    if frame is None:
        # Write to an obj or off file
        for iv, v in enumerate(verts):

            if ext is 'obj':
                fname = os.path.join(tdir, '%05d.obj' % iv)
                print('Saving mesh %s' % fname)
                write_mesh_as_obj(fname, v, faces)
            else:
                fname = os.path.join(tdir, '%05d.off' % iv)
                print('Saving mesh %s' % fname)
                write_mesh_as_off(fname, v, faces)
    else:
        if save_name is None:
            if ext is 'obj':
                fname = os.path.join(tdir, '%05d.obj' % frame)
                print('Saving mesh %s' % fname)
                write_mesh_as_obj(fname, verts[frame], faces)
            else:
                fname = os.path.join(tdir, '%05d.off' % frame)
                print('Saving mesh %s' % fname)
                write_mesh_as_off(fname, verts[frame], faces)
        else:
            if ext is 'obj':
                fname = os.path.join(tdir, save_name + '.obj')
                print('Saving mesh %s' % fname)
                write_mesh_as_obj(fname, verts[frame], faces)
            else:
                fname = os.path.join(tdir, save_name + '.off')
                print('Saving mesh %s' % fname)
                write_mesh_as_off(fname, verts[frame], faces)



drive = 'E'
dfaust_path = drive + ':/Users/Adrien/Documents/shapes/DFaust'


# dfaust_to_obj(os.path.join(dfaust_path, 'registrations_m.hdf5'),
#              'light_hopping_stiff', '50009', os.path.join(dfaust_path, 'obj'))


def dfaust_off_fixed_dataset(dfaust_path, tdir):
    for i in range(len(sids_list)):
        if sids_gender_list[i] is 'm':
            registrations = os.path.join(dfaust_path, 'registrations_m.hdf5')
        else:
            registrations = os.path.join(dfaust_path, 'registrations_f.hdf5')
        save_dfaust_object(registrations, 'light_hopping_stiff', sids_list[i], tdir,
                           ext='off', frame=0, save_name=sids_list[i])


dfaust_off_fixed_dataset('E:/Users/Adrien/Documents/shapes/DFaust',
                         'E:/Users/Adrien/Documents/shapes/DFaust/off_sids')

"""
if __name__ == '__main__':

    # Subject ids
    sids = ['50002', '50004', '50007', '50009', '50020',
            '50021', '50022', '50025', '50026', '50027']
    # Sequences available for each subject id are listed in scripts/subjects_and_sequences.txt

    parser = ArgumentParser(description='Save sequence registrations as obj')
    parser.add_argument('--path', type=str, default='../registrations_f.hdf5',
                        help='dataset path in hdf5 format')
    parser.add_argument('--seq', type=str, default='jiggle_on_toes',
                        help='sequence name')
    parser.add_argument('--sid', type=str, default='50004', choices=sids,
                        help='subject id')
    parser.add_argument('--tdir', type=str, default='./',
                        help='target directory')
    args = parser.parse_args()

    sidseq = args.sid + '_' + args.seq
    with h5py.File(args.path, 'r') as f:
        if sidseq not in f:
            print('Sequence %s from subject %s not in %s' % (args.seq, args.sid, args.path))
            f.close()
            sys.exit(1)
        verts = f[sidseq].value.transpose([2, 0, 1])
        faces = f['faces'].value

    tdir = join(args.tdir, sidseq)
    if not exists(tdir):
        mkdir(tdir)

    # Write to an obj file
    for iv, v in enumerate(verts):
        fname = join(tdir, '%05d.obj' % iv)
        print('Saving mesh %s' % fname)
        write_mesh_as_obj(fname, v, faces)
"""

