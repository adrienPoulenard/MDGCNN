import keras
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, Activation, GlobalAveragePooling1D
from keras.models import Model
from keras import backend as K
from custom_layers import RepeatAxis, Max, AsyncConv, SyncConv, FrameTransporter, ExponentialMap
from custom_layers import Input3d, SyncConvBis, AsyncConvBis, SyncGeodesicConv
from custom_layers import ExpMapBis, FrameTransporterBis, GeodesicField
from conv import GeodesicConv
from pooling import ShrinkPatchOpDisc, MaxPooling, AngularMaxPooling, AngularAveragePooling
from custom_layers import FrameBundlePullBack
from custom_layers import ShuffleVertices, DeShuffleOutput, ReshapeBis
from pooling import PoolSurfaceFixed, PoolFrameBundleFixed, Pooling, TransposedPooling, LocalAverage, Shape
import modules
from modules import gcnn_resnet_layer
from patch_operators import ExpMapFixed, GeodesicFieldFixed
import numpy as np
from sampling import ImageSampling
from pooling import PoolingOperatorFixed
from patch_operators import PatchOperatorFixed
from custom_layers import ConstantTensor
from load_data import int_to_string
from keras.layers import Concatenate


class GcnnPatchOperatorInput(object):
    """
    """

    def __init__(self, n_batch, n_v, n_rings, n_dirs, ratios, name=''):

        self.contributors = []
        self.weights = []
        self.transport = []

        self.parents = []
        self.angular_shifts = []

        self.inputs_names = []
        self.inputs = []

        self.nbatch = n_batch
        self.nv = n_v
        self.ndirs = n_dirs
        self.ratios = ratios
        self.nrings = n_rings
        self.npools = len(ratios)-1

        for i in range(self.npools):
            # pool_op = PoolingOperatorFixed(parents=parents[i], angular_shifts=angular_shifts[i], batch_size=n_batch)
            if n_v is None:
                new_nv = None
            else:
                new_nv = min(n_v[i], n_v[i+1])
            self.inputs_names.append(name + 'parents_' + int_to_string(i))
            x = Input(shape=(new_nv,),
                      batch_shape=(n_batch,) + (new_nv,),
                      dtype='int32',
                      name=name + 'parents_' + int_to_string(i))
            self.inputs.append(x)
            self.parents.append(x)
            self.inputs_names.append(name + 'angular_shifts_' + int_to_string(i))
            x = Input(shape=(new_nv,),
                      batch_shape=(n_batch,) + (new_nv,),
                      dtype='float32',
                      name=name + 'angular_shifts_' + int_to_string(i))
            self.angular_shifts.append(x)
            self.inputs.append(x)

        for stack in range(self.npools+1):
            # patch_op = PatchOperatorFixed(contributors=contributors[stack],
            #                              weights=weights[stack],
            #                             angles=angles[stack])
            if n_v is None:
                patch_op_shape = (None, n_rings[stack], n_dirs[stack], 3)
            else:
                patch_op_shape = (n_v[stack], n_rings[stack], n_dirs[stack], 3)
            self.inputs_names.append(name + 'contributors_' + int_to_string(stack))
            x = Input(shape=patch_op_shape, batch_shape=(n_batch,) + patch_op_shape, dtype='int32',
                      name=name + 'contributors_' + int_to_string(stack))
            self.contributors.append(x)
            self.inputs.append(x)
            x = Input(shape=patch_op_shape, batch_shape=(n_batch,) + patch_op_shape, dtype='float32',
                                      name=name + 'weights_' + int_to_string(stack))
            self.inputs_names.append(name + 'weights_' + int_to_string(stack))
            self.weights.append(x)
            self.inputs.append(x)

            x = Input(shape=patch_op_shape, batch_shape=(n_batch,) + patch_op_shape, dtype='float32',
                      name=name + 'transport_' + int_to_string(stack))
            self.inputs_names.append(name + 'transport_' + int_to_string(stack))
            self.transport.append(x)
            self.inputs.append(x)

        self.inputs_dict = dict(zip(self.inputs_names, self.inputs))

    def get_inputs_dict(self):
        return self.inputs_dict

    def get_inputs(self):
        return self.inputs

    def get_nbatch(self):
        return self.nbatch

    def get_nv(self):
        return self.nv

    def get_nrings(self):
        return self.nrings

    def get_ndirs(self):
        return self.ndirs

    def get_input(self, name):
        return self.inputs_dict[name]

    def get_contributors(self):
        return self.contributors

    def get_weights(self):
        return self.weights

    def get_transport(self):
        return self.transport

    def get_parents(self):
        return self.parents

    def get_angular_shifts(self):
        return self.angular_shifts


class GcnnResnet(object):
    """
    """

    def __init__(self, n_batch, ratio, n_v, n_rings, n_dirs,
                 inputs,
                 bin_contributors,
                 weights,
                 transport,
                 parents,
                 angular_shifts,
                 batch_norm=False,
                 nstacks=1,
                 nresblocks_per_stack=2,
                 nfilters=16,
                 sync_mode='radial_sync',
                 additional_inputs=None,
                 name=''):

        """ResNet Version 1 Model builder [a]
        Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
        Last ReLU is after the shortcut connection.
        At the beginning of each stage, the feature map size is halved (downsampled)
        by a convolutional layer with strides=2, while the number of filters is
        doubled. Within each stage, the layers have the same number filters and the
        same number of filters.
        Features maps sizes:
        stage 0: 32x32, 16
        stage 1: 16x16, 32
        stage 2:  8x8,  64
        The Number of parameters is approx the same as Table 6 of [a]:
        ResNet20 0.27M
        ResNet32 0.46M
        ResNet44 0.66M
        ResNet56 0.85M
        ResNet110 1.7M
        # Arguments
            input_shape (tensor): shape of input image tensor
            depth (int): number of core convolutional layers
            num_classes (int): number of classes (CIFAR10 has 10)
        # Returns
            model (Model): Keras model instance
        """
        if name is not '':
            name = name + '_'

        bn = batch_norm
        pool_ = True

        take_max = False
        if sync_mode is 'async':
            take_max = True

        if n_v is None:
            n_v = [None for _ in range(len(n_dirs))]

        # if (depth - 2) % 6 != 0:
        #    raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
        # Start model definition.
        num_filters = nfilters
        self.nfilters = []
        # num_res_blocks = int((depth - 2) / 6)
        self.num_res_blocks = nresblocks_per_stack
        self.additional_inputs = additional_inputs
        self.nstacks = nstacks
        self.inputs_names = [name + 'input_signal']

        x = inputs

        C = bin_contributors
        W = weights
        TA = transport
        P = parents
        AS = angular_shifts
        NV = []
        for i in range(nstacks-1):
            self.inputs_names.append(name + 'parents_' + int_to_string(i))
            self.inputs_names.append(name + 'angular_shifts_' + int_to_string(i))
        for i in range(nstacks):
            self.inputs_names.append(name + 'contributors_' + int_to_string(i))
            self.inputs_names.append(name + 'weights_' + int_to_string(i))
            self.inputs_names.append(name + 'transport_' + int_to_string(i))
            NV.append(Shape(axis=1)(C[i]))



        """
        inputs = Input(shape=(n_v[0], input_dim), batch_shape=(n_batch,) + (n_v[0], input_dim),
                       name=name + 'input_signal')
        x = inputs

        # patch operator


        
        C = []
        W = []
        TA = []

        P = []
        AS = []
        
        
        for i in range(nstacks - 1):
            # pool_op = PoolingOperatorFixed(parents=parents[i], angular_shifts=angular_shifts[i], batch_size=n_batch)

            self.inputs_names.append(name + 'parents_' + int_to_string(i))
            P.append(Input(shape=(n_v[i + 1],),
                           batch_shape=(n_batch,) + (n_v[i + 1],),
                           dtype='int32',
                           name=name + 'parents_' + int_to_string(i)))
            self.inputs_names.append(name + 'angular_shifts_' + int_to_string(i))
            AS.append(Input(shape=(n_v[i + 1],),
                            batch_shape=(n_batch,) + (n_v[i + 1],),
                            dtype='float32',
                            name=name + 'angular_shifts_' + int_to_string(i)))

        for stack in range(nstacks):
            # patch_op = PatchOperatorFixed(contributors=contributors[stack],
            #                              weights=weights[stack],
            #                             angles=angles[stack])
            patch_op_shape = (n_v[stack], n_rings[stack], n_dirs[stack], 3)
            self.inputs_names.append(name + 'contributors_' + int_to_string(stack))
            C.append(Input(shape=patch_op_shape, batch_shape=(n_batch,) + patch_op_shape, dtype='int32',
                           name=name + 'contributors_' + int_to_string(stack)))
            self.inputs_names.append(name + 'weights_' + int_to_string(stack))
            W.append(Input(shape=patch_op_shape, batch_shape=(n_batch,) + patch_op_shape, dtype='float32',
                           name=name + 'weights_' + int_to_string(stack)))
            self.inputs_names.append(name + 'transport_' + int_to_string(stack))
            TA.append(Input(shape=patch_op_shape, batch_shape=(n_batch,) + patch_op_shape, dtype='float32',
                            name=name + 'transport_' + int_to_string(stack)))

        """
        stack_ = 0

        if num_filters is None:
            num_filters = K.int_shape(x)[-1]

        if K.int_shape(x)[-1] is not num_filters:
            x = gcnn_resnet_layer(inputs=x,
                                  contributors=C[stack_],
                                  weights=W[stack_],
                                  angles=TA[stack_],
                                  n_v=n_v[0],
                                  n_rings=n_rings[0],
                                  n_dirs=n_dirs[0],
                                  num_filters=num_filters,
                                  sync_mode=sync_mode,
                                  batch_normalization=bn,
                                  take_max=take_max)

        self.stacks = []

        # Instantiate the stack of residual units
        for stack in range(nstacks):
            for res_block in range(self.num_res_blocks):
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    # pooling
                    # num_filters = 2*num_filters
                    if pool_:
                        # num_filters = int(np.sqrt((1. * n_v[stack - 1]) / (1. * n_v[stack])) * num_filters) + 1
                        # num_filters = int(np.sqrt(ratio[stack-1] / ratio[stack] + 0.0001)*num_filters)
                        # num_filters = int(np.sqrt(ratio[stack-1] / ratio[stack] + 0.0001)*K.int_shape(x)[-1])
                        stack_ = stack
                        if ratio[stack-1] > ratio[stack]:
                            x = Pooling()([x, P[stack-1], AS[stack-1]])
                        else:
                            x = TransposedPooling(new_nv=n_v[stack],
                                                  new_ndirs=n_dirs[stack])([x, P[stack - 1],
                                                                            AS[stack - 1],
                                                                           NV[stack]])
                            """
                            if n_v[stack] is None:
                                
                                key = 'stack_' + int_to_string(stack)
                                if key in self.additional_inputs:
                                    new_nv = K.shape(self.additional_inputs[key])[1]
                                else:
                                    raise ValueError('number of vertices could not be inferred')
                                
                                
                            else:
                                new_nv = n_v[stack]
                                x = TransposedPooling(new_nv=new_nv,
                                                      new_ndirs=n_dirs[stack])([x, P[stack - 1], AS[stack - 1]])
                            """
                    else:
                        num_filters *= 2
                        stack_ = 0

                    if self.additional_inputs is not None:
                        key = 'stack_' + int_to_string(stack)
                        if key in self.additional_inputs:
                            x = Concatenate(axis=-1)([x, self.additional_inputs[key]])

                    num_filters = int(np.sqrt(ratio[stack - 1] / ratio[stack] + 0.0001) * K.int_shape(x)[-1])

                y = gcnn_resnet_layer(inputs=x,
                                      contributors=C[stack_],
                                      weights=W[stack_],
                                      angles=TA[stack_],
                                      n_v=n_v[stack_],
                                      n_rings=n_rings[stack_],
                                      n_dirs=n_dirs[stack_],
                                      num_filters=num_filters,
                                      sync_mode=sync_mode,
                                      batch_normalization=bn,
                                      take_max=take_max)
                y = gcnn_resnet_layer(inputs=y,
                                      contributors=C[stack_],
                                      weights=W[stack_],
                                      angles=TA[stack_],
                                      n_v=n_v[stack_],
                                      n_rings=n_rings[stack_],
                                      n_dirs=n_dirs[stack_],
                                      num_filters=num_filters,
                                      sync_mode=sync_mode,
                                      batch_normalization=bn,
                                      take_max=take_max,
                                      activation=None)

                if stack > 0 and res_block == 0:  # first layer but not first stack
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x = Dense(units=num_filters, use_bias=False, activation=None)(x)

                    # x = Dropout(0.25)(x)

                x = keras.layers.add([x, y])

                if res_block == self.num_res_blocks-1:
                    # save stack
                    self.stacks.append(x)

                x = Activation('relu')(x)


            # if stack > 0:
            #    num_filters = int(np.sqrt((1. * n_v[stack - 1]) / (1. * n_v[stack])) * num_filters)

        """
        # Add classifier on top.
        # v1 does not use BN after last shortcut connection-ReLU
        x = AngularMaxPooling(r=1, take_max=True)(x)
        # x = AngularAveragePooling(r=1, take_average=True)(x)
        x = GlobalAveragePooling1D()(x)
        y = Dense(num_classes,
                  kernel_initializer='he_normal',
                  name='final_vote')(x)
        outputs = Activation('softmax')(y)
        """

        # Instantiate model.
        self.output_dim = num_filters
        self.output = x
        self.input = inputs
        self.inputs_list = [self.input]

        for i in range(len(C)):
            self.inputs_list.append(C[i])
            self.inputs_list.append(W[i])
            self.inputs_list.append(TA[i])
        for i in range(len(P)):
            self.inputs_list.append(P[i])
            self.inputs_list.append(AS[i])
        self.inputs_dict = dict(zip(self.inputs_names, self.inputs_list))


        # self.model = Model(inputs=self.inputs_list, outputs=self.output)

    def get_stack(self, i):
        return self.stacks[i]

    def get_stack_shape(self, i):
        return K.shape(self.stacks[i])

    def get_additional_inputs(self):
        return self.additional_inputs

    def get_inputs_list(self):
        return self.inputs_list

    def get_inputs_dict(self):
        return self.inputs_dict

    def get_input(self, name):
        return self.inputs_dict[name]

    def get_output(self):
        return self.output

    def get_output_shape(self):
        return K.shape(self.output)

    def get_additional_input(self, name):
        return self.additional_inputs[name]

    def get_output_dim(self):
        return self.output_dim

    # def get_model(self):
    #    return self.model


def merge_layers_dicts(dict1, dict2):
    if dict1 is None and dict2 is None:
        return {}
    elif dict1 is None:
        return dict2
    elif dict2 is None:
        return dict1

    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())
    keys = list(keys1.union(keys2))
    inter = keys1.intersection(keys2)
    lst = []
    for i in range(len(keys)):
        if keys[i] in inter:
            lst.append(K.concatenate([dict1[keys[i]], dict2[keys[i]]], axis=-1))
        elif keys[i] in keys1:
            lst.append(dict1[keys[i]])
        elif keys[i] in keys2:
            lst.append(dict2[keys[i]])

    return dict(zip(keys, lst))


class GcnnUResnet(object):
    """
    """

    def __init__(self, n_batch, ratio, n_v, n_rings, n_dirs,
                 inputs,
                 bin_contributors,
                 weights,
                 transport,
                 parents,
                 angular_shifts,
                 batch_norm=False,
                 nstacks=1,
                 nresblocks_per_stack=2,
                 nfilters=16,
                 sync_mode='radial_sync',
                 encode_add_inputs=None,
                 decode_add_inputs=None,
                 name=''):

        if name is not '':
            name = name + '_'

        self.encode_add_inputs = encode_add_inputs
        encoder = GcnnResnet(n_batch=n_batch, ratio=ratio, n_v=n_v, n_rings=n_rings, n_dirs=n_dirs,
                             inputs=inputs,
                             bin_contributors=bin_contributors,
                             weights=weights,
                             transport=transport,
                             parents=parents,
                             angular_shifts=angular_shifts,
                             batch_norm=batch_norm,
                             nstacks=nstacks,
                             nresblocks_per_stack=nresblocks_per_stack,
                             nfilters=nfilters,
                             sync_mode=sync_mode,
                             additional_inputs=encode_add_inputs,
                             name=name + 'encoder')

        decode_ratio = []
        decode_nv = []
        decode_nrings = []
        decode_ndirs = []

        # decode patch op
        decode_contributors = []
        decode_weights = []
        decode_transport = []
        decode_parents = []
        decode_angular_shifts = []

        for i in range(nstacks):
            decode_ratio.append(ratio[nstacks - 1 - i])
            if n_v is None:
                decode_nv.append(None)
            else:
                decode_nv.append(n_v[nstacks - 1 - i])
            decode_nrings.append(n_rings[nstacks - 1 - i])
            decode_ndirs.append(n_dirs[nstacks - 1 - i])

            decode_contributors.append(bin_contributors[nstacks-1-i])
            decode_weights.append(weights[nstacks-1-i])
            decode_transport.append(transport[nstacks-1-i])

        for i in range(nstacks-1):
            decode_parents.append(parents[nstacks-2-i])
            decode_angular_shifts.append(angular_shifts[nstacks-2-i])


        # add shortcuts:

        shortcuts_names = []
        shortcuts_list = []

        for i in range(nstacks):
            shortcuts_names.append('stack_' + int_to_string(i))
            shortcuts_list.append(encoder.get_stack(nstacks - 1 - i))
        shortcuts = dict(zip(shortcuts_names, shortcuts_list))
        self.decode_add_inputs = merge_layers_dicts(decode_add_inputs, shortcuts)
        print('zzzzzzzzzzzzzzzzzzzzz')
        print(self.decode_add_inputs)
        # self.decode_add_inputs = None

        decode_nfilters = encoder.get_output_dim()

        decoder = GcnnResnet(n_batch=n_batch, ratio=decode_ratio, n_v=decode_nv, n_rings=decode_nrings,
                             n_dirs=decode_ndirs,
                             inputs=encoder.get_output(),
                             bin_contributors=decode_contributors,
                             weights=decode_weights,
                             transport=decode_transport,
                             parents=decode_parents,
                             angular_shifts=decode_angular_shifts,
                             nstacks=nstacks,
                             nresblocks_per_stack=nresblocks_per_stack,
                             nfilters=decode_nfilters,
                             sync_mode=sync_mode,
                             batch_norm=batch_norm,
                             additional_inputs=self.decode_add_inputs,
                             name=name + 'decoder')

        self.encoder = encoder
        self.decoder = decoder

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder







