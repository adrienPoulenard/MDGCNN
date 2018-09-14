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
from pooling import PoolSurfaceFixed, PoolFrameBundleFixed, Pooling, LocalAverage
import modules
from modules import gcnn_resnet_layer
from patch_operators import ExpMapFixed, GeodesicFieldFixed
import numpy as np
from sampling import ImageSampling
from pooling import PoolingOperatorFixed
from patch_operators import PatchOperatorFixed
from custom_layers import ConstantTensor
from load_data import int_to_string

from gcnn_resnet import GcnnResnet, GcnnUResnet, GcnnPatchOperatorInput
from conv import Local3dGeodesicConv
from keras.layers import BatchNormalization





def SGCNN_3D(nb_classes, n_batch, n_v, n_rings, n_dirs, shuffle_vertices=True):

    # create input layer
    inp3d_shape = (n_v, n_rings, n_dirs, 3)
    input3d = Input(shape=inp3d_shape, batch_shape=(n_batch,) + inp3d_shape)

    inpExp_shape = (n_v, n_rings, n_dirs)
    inputExp = Input(shape=inpExp_shape, batch_shape=(n_batch,) + inpExp_shape, dtype='int32')

    inpFrame_shape = (n_v, n_rings, n_dirs)
    inputFrame = Input(shape=inpFrame_shape, batch_shape=(n_batch,) + inpFrame_shape, dtype='int32')

    x = Input3d(n_batch, n_v, n_rings, n_dirs, 16)(input3d)

    """
    x, Exp, Frame, s, rs = ShuffleVertices(n_batch, n_v, n_rings, n_dirs,
                                           shuffle=shuffle_vertices)([x, inputExp, inputFrame])

    
    X = ShuffleVertices(n_batch, n_v, n_rings, n_dirs, shuffle=shuffle_vertices)
    
    x = X[0]
    Exp = X[1]
    Frame = X[2]
    s = X[3]
    rs = X[4]
    """

    Frame = FrameTransporterBis(n_batch, n_v, n_rings, n_dirs)([inputExp, inputFrame])

    x = SyncConvBis(nfilters=32, nv=n_v, ndirs=n_dirs, nrings=n_rings)([x, Frame])
    nunits = 4
    x00 = Dense(units=nunits, activation='relu')(x)
    x01 = Dense(units=nunits, activation='relu')(x)
    x02 = Dense(units=nunits, activation='relu')(x)
    x03 = Dense(units=nunits, activation='relu')(x)
    x04 = Dense(units=nunits, activation='relu')(x)
    x05 = Dense(units=nunits, activation='relu')(x)
    x06 = Dense(units=nunits, activation='relu')(x)
    x07 = Dense(units=nunits, activation='relu')(x)

    nfilters = 8
    x10 = SyncConvBis(nfilters=nfilters, nv=n_v, ndirs=n_dirs, nrings=n_rings)([x00, Frame])
    x11 = SyncConvBis(nfilters=nfilters, nv=n_v, ndirs=n_dirs, nrings=n_rings)([x01, Frame])
    x12 = SyncConvBis(nfilters=nfilters, nv=n_v, ndirs=n_dirs, nrings=n_rings)([x02, Frame])
    x13 = SyncConvBis(nfilters=nfilters, nv=n_v, ndirs=n_dirs, nrings=n_rings)([x03, Frame])
    x14 = SyncConvBis(nfilters=nfilters, nv=n_v, ndirs=n_dirs, nrings=n_rings)([x04, Frame])
    x15 = SyncConvBis(nfilters=nfilters, nv=n_v, ndirs=n_dirs, nrings=n_rings)([x05, Frame])
    x16 = SyncConvBis(nfilters=nfilters, nv=n_v, ndirs=n_dirs, nrings=n_rings)([x06, Frame])
    x17 = SyncConvBis(nfilters=nfilters, nv=n_v, ndirs=n_dirs, nrings=n_rings)([x07, Frame])

    x = Concatenate(axis=-1)([x10, x11, x12, x13, x14, x15, x16, x17])

    # angular max pooling
    x = Max(axis=2, keepdims=False)(x)
    # max over vertexes
    # x = Max(axis=1, keepdims=False)(x)
    # final vote
    # x = Dropout(0.25)(x)
    x = Dense(units=128, activation='relu')(x)
    # x = Dropout(0.5)(x)
    # softmax
    res = Dense(units=nb_classes, activation='softmax')(x)
    # res = DeShuffleOutput(n_batch, n_v, n_rings, n_dirs, shuffle=shuffle_vertices)([x, rs])

    return input3d, inputExp, inputFrame, res


def SGCNN_signals(nb_classes, n_descs, n_batch, n_v, n_rings, n_dirs, shuffle_vertices=True):
    # create input layer
    descs_shape = (n_v, n_descs)
    descs = Input(shape=descs_shape, batch_shape=(n_batch,) + descs_shape)

    inpExp_shape = (n_v, n_rings, n_dirs)
    inputExp = Input(shape=inpExp_shape, batch_shape=(n_batch,) + inpExp_shape, dtype='int32')

    inpFrame_shape = (n_v, n_rings, n_dirs)
    inputFrame = Input(shape=inpFrame_shape, batch_shape=(n_batch,) + inpFrame_shape, dtype='int32')

    # x = input3d

    """
    x, Exp, Frame, s, rs = ShuffleVertices(n_batch, n_v, n_rings, n_dirs,
                                           shuffle=shuffle_vertices)([x, inputExp, inputFrame])


    X = ShuffleVertices(n_batch, n_v, n_rings, n_dirs, shuffle=shuffle_vertices)

    x = X[0]
    Exp = X[1]
    Frame = X[2]
    s = X[3]
    rs = X[4]
    """

    Frame = FrameTransporterBis(n_batch, n_v, n_rings, n_dirs)([inputExp, inputFrame])
    Exp = ExpMapBis(n_batch, n_v, n_rings, n_dirs)(inputExp)

    x = AsyncConvBis(nfilters=32, nv=n_v, ndirs=n_dirs, nrings=n_rings, take_max=False)([descs, Exp])
    nunits = 4
    x00 = Dense(units=nunits, activation='relu')(x)
    x01 = Dense(units=nunits, activation='relu')(x)
    x02 = Dense(units=nunits, activation='relu')(x)
    x03 = Dense(units=nunits, activation='relu')(x)
    x04 = Dense(units=nunits, activation='relu')(x)
    x05 = Dense(units=nunits, activation='relu')(x)
    x06 = Dense(units=nunits, activation='relu')(x)
    x07 = Dense(units=nunits, activation='relu')(x)

    nfilters = 8
    x10 = SyncConvBis(nfilters=nfilters, nv=n_v, ndirs=n_dirs, nrings=n_rings)([x00, Frame])
    x11 = SyncConvBis(nfilters=nfilters, nv=n_v, ndirs=n_dirs, nrings=n_rings)([x01, Frame])
    x12 = SyncConvBis(nfilters=nfilters, nv=n_v, ndirs=n_dirs, nrings=n_rings)([x02, Frame])
    x13 = SyncConvBis(nfilters=nfilters, nv=n_v, ndirs=n_dirs, nrings=n_rings)([x03, Frame])
    x14 = SyncConvBis(nfilters=nfilters, nv=n_v, ndirs=n_dirs, nrings=n_rings)([x04, Frame])
    x15 = SyncConvBis(nfilters=nfilters, nv=n_v, ndirs=n_dirs, nrings=n_rings)([x05, Frame])
    x16 = SyncConvBis(nfilters=nfilters, nv=n_v, ndirs=n_dirs, nrings=n_rings)([x06, Frame])
    x17 = SyncConvBis(nfilters=nfilters, nv=n_v, ndirs=n_dirs, nrings=n_rings)([x07, Frame])

    x = Concatenate(axis=-1)([x10, x11, x12, x13, x14, x15, x16, x17])

    # angular max pooling
    x = Max(axis=2, keepdims=False)(x)
    # max over vertexes
    # x = Max(axis=1, keepdims=False)(x)
    # final vote
    x = Dropout(0.25)(x)
    x = Dense(units=128, activation='relu')(x)
    x = Dropout(0.5)(x)
    # softmax
    x = Dense(units=nb_classes, activation='softmax')(x)
    print('x shape')
    print(x.get_shape())
    # res = ReshapeBis((n_batch, n_v*nb_classes))(x)
    # res = Flatten()(x)
    res = x
    # res = DeShuffleOutput(n_batch, n_v, n_rings, n_dirs, shuffle=shuffle_vertices)([x, rs])

    return descs, inputExp, inputFrame, res


def segmentation(nb_classes, n_descs, n_batch, n_v, n_rings, n_dirs, shuffle_vertices=True):
    # create input layer
    descs_shape = (n_v, n_descs)
    descs = Input(shape=descs_shape, batch_shape=(n_batch,) + descs_shape)

    inpExp_shape = (n_v, n_rings, n_dirs)
    inputExp = Input(shape=inpExp_shape, batch_shape=(n_batch,) + inpExp_shape, dtype='int32')

    inpFrame_shape = (n_v, n_rings, n_dirs)
    inputFrame = Input(shape=inpFrame_shape, batch_shape=(n_batch,) + inpFrame_shape, dtype='int32')

    inp3d_shape = (n_v, 3)
    input3d = Input(shape=inp3d_shape, batch_shape=(n_batch,)+inp3d_shape)

    inp3dframes_shape = (n_v, 3, 3)
    input3dframes = Input(shape=inp3dframes_shape, batch_shape=(n_batch,)+inp3dframes_shape)



    x = descs

    Frame = FrameTransporterBis(n_batch, n_v, n_rings, n_dirs)([inputExp, inputFrame])
    Exp = ExpMapBis(n_batch, n_v, n_rings, n_dirs)(inputExp)
    Field = GeodesicField(n_batch, n_v, n_rings, n_dirs)([inputExp, inputFrame])

    Exp1, Frame1 = ShrinkPatchOpDisc(nrings=1)([Exp, Frame])

    #x3d = Input3d(batch_size=n_batch, nv=n_v, nrings=n_rings, ndirs=n_dirs, nfilters=16, take_max=False)(
    #    [input3d, input3dframes, Exp])

    syncConv = modules.ConvOperator(conv_op=SyncConvBis, patch_op=Frame,
                                    nv=n_v, nrings=n_rings, ndirs=n_dirs, nfilters=32, take_max=False)

    asyncConv = modules.ConvOperator(conv_op=AsyncConvBis, patch_op=Exp,
                                     nv=n_v, nrings=n_rings, ndirs=n_dirs, take_max=True)


    sync = False

    if sync:
        #x = Input3d(batch_size=n_batch, nv=n_v, nrings=n_rings, ndirs=n_dirs, nfilters=16, take_max=False)(
        #    [input3d, input3dframes, Exp])
        x = descs
        x = AsyncConvBis(nv=n_v, nrings=n_rings, ndirs=n_dirs, nfilters=16, take_max=False)([x, Exp])
        x = AngularMaxPooling(r=1, take_max=False)(x)
        # x = SyncConvBis(nv=n_v, nrings=n_rings, ndirs=n_dirs, nfilters=32, take_max=False)([x, Frame])
        x = SyncGeodesicConv(nv=n_v, nrings=n_rings, ndirs=n_dirs, nfilters=32, take_max=False)([x, Field])
        x = AngularMaxPooling(r=1, take_max=False)(x)
        #x = MaxPooling(take_max=False)([x, Frame1])
        # x = SyncConvBis(nv=n_v, nrings=n_rings, ndirs=n_dirs, nfilters=64, take_max=True)([x, Frame])
        x = SyncGeodesicConv(nv=n_v, nrings=n_rings, ndirs=n_dirs, nfilters=64, take_max=True)([x, Field])
        x = MaxPooling(take_max=True)([x, Exp1])
        #x = SyncConvBis(nv=n_v, nrings=n_rings, ndirs=n_dirs, nfilters=128, take_max=True)([x, Frame])

    else:
        #x = Input3d(batch_size=n_batch, nv=n_v, nrings=n_rings, ndirs=n_dirs, nfilters=16, take_max=True)(
        #    [input3d, input3dframes, Exp])
        x = descs
        x = AsyncConvBis(nv=n_v, nrings=n_rings, ndirs=n_dirs, nfilters=16, take_max=True)([x, Exp])
        x = AsyncConvBis(nv=n_v, nrings=n_rings, ndirs=n_dirs, nfilters=32, take_max=True)([x, Exp])
        #x = MaxPooling(take_max=True)([x, Exp1])
        x = AsyncConvBis(nv=n_v, nrings=n_rings, ndirs=n_dirs, nfilters=64, take_max=True)([x, Exp])
        # x = AsyncConvBis(nv=n_v, nrings=n_rings, ndirs=n_dirs, nfilters=128, take_max=True)([x, Exp])
        x = MaxPooling(take_max=True)([x, Exp1])




    # angular max pooling
    # x = Max(axis=2, keepdims=False)(x)
    # x = Dropout(0.25)(x)
    x = Dense(units=128, activation='relu')(x)

    # x = Dropout(0.1)(x)
    # softmax
    x = Dense(units=nb_classes, activation='softmax')(x)
    print('x shape')
    print(x.get_shape())
    # res = ReshapeBis((n_batch, n_v*nb_classes))(x)
    # res = Flatten()(x)
    res = x
    # res = DeShuffleOutput(n_batch, n_v, n_rings, n_dirs, shuffle=shuffle_vertices)([x, rs])

    return descs, input3d, input3dframes, inputExp, inputFrame, res


def test_reconstruct(nb_classes, n_descs, n_batch, n_v, n_rings, n_dirs, shuffle_vertices=True):
    # create input layer
    descs_shape = (n_v, n_descs)
    descs = Input(shape=descs_shape, batch_shape=(n_batch,) + descs_shape)

    inpExp_shape = (n_v, n_rings, n_dirs)
    inputExp = Input(shape=inpExp_shape, batch_shape=(n_batch,) + inpExp_shape, dtype='int32')

    inpFrame_shape = (n_v, n_rings, n_dirs)
    inputFrame = Input(shape=inpFrame_shape, batch_shape=(n_batch,) + inpFrame_shape, dtype='int32')

    inp3d_shape = (n_v, 3)
    input3d = Input(shape=inp3d_shape, batch_shape=(n_batch,) + inp3d_shape)

    inp3dframes_shape = (n_v, 3, 3)
    input3dframes = Input(shape=inp3dframes_shape, batch_shape=(n_batch,) + inp3dframes_shape)

    Frame = FrameTransporterBis(n_batch, n_v, n_rings, n_dirs)([inputExp, inputFrame])
    Exp = ExpMapBis(n_batch, n_v, n_rings, n_dirs)(inputExp)
    Field = GeodesicField(n_batch, n_v, n_rings, n_dirs)([inputExp, inputFrame])

    Exp1, Frame1 = ShrinkPatchOpDisc(nrings=1)([Exp, Frame])

    syncConv = modules.ConvOperator(conv_op=SyncConvBis, patch_op=Frame,
                                    nv=n_v, nrings=n_rings, ndirs=n_dirs, nfilters=32, take_max=False)

    asyncConv = modules.ConvOperator(conv_op=AsyncConvBis, patch_op=Exp,
                                     nv=n_v, nrings=n_rings, ndirs=n_dirs, take_max=True)

    sync = False

    #x3d = Input3d(batch_size=n_batch, nv=n_v, nrings=n_rings, ndirs=n_dirs, nfilters=64, take_max=not sync)(
        #[input3d, input3dframes, Exp])



    nfilters = 32
    nnew3d = 8

    """
    if sync:
        #init
        Transfert = Frame
        syncConv.set_nfilters(nfilters)
        # link kernel
        link = SyncConvBis(nfilters=nfilters, nv=n_v, ndirs=n_dirs, nrings=n_rings, take_max=False)
        x3d = Input3d(batch_size=n_batch, nv=n_v, nrings=n_rings, ndirs=n_dirs, nfilters=80, take_max=not sync)(
        [input3d, input3dframes, Exp])
        x0 = Dense(units=nfilters, activation='relu')(x3d)

        # begin
        new3d = Dense(units=nnew3d, activation='relu')(x3d)
        y = Concatenate(axis=-1)([new3d, x0])
        # x = link([y, Transfert])  # fixed kernel
        syncConv.set_nfilters(16)
        x1 = syncConv(y)

        new3d = Dense(units=nnew3d, activation='relu')(x3d)
        y = Concatenate(axis=-1)([new3d, x1])
        # x = link([y, Transfert])  # fixed kernel
        syncConv.set_nfilters(32)
        x2 = syncConv(y)

        new3d = Dense(units=nnew3d, activation='relu')(x3d)
        y = Concatenate(axis=-1)([new3d, x2])
        # x = link([y, Transfert])  # fixed kernel
        syncConv.set_nfilters(64)
        x = syncConv(y)
        x = Max(axis=2, keepdims=False)(x)
    else:
        #init
        Transfert = Exp
        asyncConv.set_nfilters(nfilters)
        # link kernel
        link = AsyncConvBis(nfilters=nfilters, nv=n_v, ndirs=n_dirs, nrings=n_rings, take_max=True)
        x3d = Input3d(batch_size=n_batch, nv=n_v, nrings=n_rings, ndirs=n_dirs, nfilters=80, take_max=not sync)(
        [input3d, input3dframes, Exp])
        x0 = Dense(units=nfilters, activation='relu')(x3d)

        # begin
        new3d = Dense(units=nnew3d, activation='relu')(x3d)
        y = Concatenate(axis=-1)([new3d, x0])
        # x = link([y, Transfert])  # fixed kernel
        asyncConv.set_nfilters(16)
        x1 = asyncConv(y)

        new3d = Dense(units=nnew3d, activation='relu')(x3d)
        y = Concatenate(axis=-1)([new3d, x1])
        # x = link([y, Transfert])  # fixed kernel
        asyncConv.set_nfilters(32)
        x2 = asyncConv(y)

        new3d = Dense(units=nnew3d, activation='relu')(x3d)
        y = Concatenate(axis=-1)([new3d, x2])
        # x = link([y, Transfert])  # fixed kernel
        asyncConv.set_nfilters(64)
        x = asyncConv(y)
    """



    if sync:
        x = Input3d(batch_size=n_batch, nv=n_v, nrings=n_rings, ndirs=n_dirs, nfilters=16, take_max=False)(
           [input3d, input3dframes, Exp])
        """
        x = descs
        x = AsyncConvBis(nv=n_v, nrings=n_rings, ndirs=n_dirs, nfilters=16, take_max=False)([x, Exp])
        x = AngularMaxPooling(r=1, take_max=False)(x)
        x = SyncConvBis(nv=n_v, nrings=n_rings, ndirs=n_dirs, nfilters=32, take_max=False)([x, Frame])
        x = AngularMaxPooling(r=1, take_max=False)(x)
        x = MaxPooling(take_max=False)([x, Frame1])
        x = SyncConvBis(nv=n_v, nrings=n_rings, ndirs=n_dirs, nfilters=64, take_max=True)([x, Frame])
        x = MaxPooling(take_max=True)([x, Exp1])
        #x = SyncConvBis(nv=n_v, nrings=n_rings, ndirs=n_dirs, nfilters=128, take_max=True)([x, Frame])
        """


        #x = descs
        # x = AsyncConvBis(nv=n_v, nrings=n_rings, ndirs=n_dirs, nfilters=16, take_max=False)([x, Exp])
        #x = AngularMaxPooling(r=1, take_max=False)(x)

        x = SyncGeodesicConv(nv=n_v, nrings=n_rings, ndirs=n_dirs, nfilters=32, take_max=False)([x, Field])
        #x = AngularMaxPooling(r=1, take_max=False)(x)
        x = MaxPooling(take_max=False)([x, Frame1])
        x = SyncGeodesicConv(nv=n_v, nrings=n_rings, ndirs=n_dirs, nfilters=64, take_max=True)([x, Field])
        x = MaxPooling(take_max=True)([x, Exp1])

    else:
        #x = Input3d(batch_size=n_batch, nv=n_v, nrings=n_rings, ndirs=n_dirs, nfilters=16, take_max=True)(
        #    [input3d, input3dframes, Exp])
        x = descs
        #x = AsyncConvBis(nv=n_v, nrings=n_rings, ndirs=n_dirs, nfilters=16, take_max=True)([x, Exp])
        #x = MaxPooling(take_max=True)([x, Exp1])
        #x = AsyncConvBis(nv=n_v, nrings=n_rings, ndirs=n_dirs, nfilters=32, take_max=True)([x, Exp])

        #x = AsyncConvBis(nv=n_v, nrings=n_rings, ndirs=n_dirs, nfilters=64, take_max=True)([x, Exp])
        #x = AsyncConvBis(nv=n_v, nrings=n_rings, ndirs=n_dirs, nfilters=64, take_max=True)([x, Exp])
        #x = MaxPooling(take_max=True)([x, Exp1])
    # angular max pooling
    # x = Max(axis=2, keepdims=False)(x)
    # x = Dropout(0.25)(x)
    # x = Concatenate(axis=-1)([x0, x1, x2, x3])
    #x = Concatenate(axis=-1)([x, descs])
    x = Dense(units=2048, activation='relu')(x)
    x = Dense(units=32, activation='linear')(x)
    # x = Dropout(0.5)(x)
    # softmax
    x = Dense(units=nb_classes, activation='linear')(x)
    print('x shape')
    print(x.get_shape())
    # res = ReshapeBis((n_batch, n_v*nb_classes))(x)
    # res = Flatten()(x)
    res = x
    # res = DeShuffleOutput(n_batch, n_v, n_rings, n_dirs, shuffle=shuffle_vertices)([x, rs])

    return descs, input3d, input3dframes, inputExp, inputFrame, res


def shape_classification(nb_classes, n_descs, n_batch, n_v, n_rings, n_dirs, shuffle_vertices=True):
    # create input layer
    descs_shape = (n_v, n_descs)
    descs = Input(shape=descs_shape, batch_shape=(n_batch,) + descs_shape)

    inpExp_shape = (n_v, n_rings, n_dirs)
    inputExp = Input(shape=inpExp_shape, batch_shape=(n_batch,) + inpExp_shape, dtype='int32')

    inpFrame_shape = (n_v, n_rings, n_dirs)
    inputFrame = Input(shape=inpFrame_shape, batch_shape=(n_batch,) + inpFrame_shape, dtype='int32')

    inp3d_shape = (n_v, 3)
    input3d = Input(shape=inp3d_shape, batch_shape=(n_batch,)+inp3d_shape)

    inp3dframes_shape = (n_v, 3, 3)
    input3dframes = Input(shape=inp3dframes_shape, batch_shape=(n_batch,)+inp3dframes_shape)



    x = descs

    Frame = FrameTransporterBis(n_batch, n_v, n_rings, n_dirs)([inputExp, inputFrame])
    Exp = ExpMapBis(n_batch, n_v, n_rings, n_dirs)(inputExp)
    Field = GeodesicField(n_batch, n_v, n_rings, n_dirs)([inputExp, inputFrame])

    Exp1, Frame1 = ShrinkPatchOpDisc(nrings=1)([Exp, Frame])

    #x3d = Input3d(batch_size=n_batch, nv=n_v, nrings=n_rings, ndirs=n_dirs, nfilters=16, take_max=False)(
    #    [input3d, input3dframes, Exp])

    syncConv = modules.ConvOperator(conv_op=SyncConvBis, patch_op=Frame,
                                    nv=n_v, nrings=n_rings, ndirs=n_dirs, nfilters=32, take_max=False)

    asyncConv = modules.ConvOperator(conv_op=AsyncConvBis, patch_op=Exp,
                                     nv=n_v, nrings=n_rings, ndirs=n_dirs, take_max=True)


    sync = False
    n_filters = 16
    if sync:
        #x = Input3d(batch_size=n_batch, nv=n_v, nrings=n_rings, ndirs=n_dirs, nfilters=16, take_max=False)(
        #    [input3d, input3dframes, Exp])
        x = descs
        x = AsyncConvBis(nv=n_v, nrings=n_rings, ndirs=n_dirs, nfilters=n_filters, take_max=False)([x, Exp])
        x = AngularMaxPooling(r=1, take_max=False)(x)

        x = SyncGeodesicConv(nv=n_v, nrings=n_rings, ndirs=n_dirs, nfilters=n_filters, take_max=False)([x, Field])
        x = AngularMaxPooling(r=1, take_max=False)(x)

        # x = SyncGeodesicConv(nv=n_v, nrings=n_rings, ndirs=n_dirs, nfilters=n_filters, take_max=False)([x, Field])
        # x = AngularMaxPooling(r=1, take_max=False)(x)

        #x = SyncGeodesicConv(nv=n_v, nrings=n_rings, ndirs=n_dirs, nfilters=n_filters, take_max=False)([x, Field])
        #x = AngularMaxPooling(r=1, take_max=False)(x)

        x = SyncGeodesicConv(nv=n_v, nrings=n_rings, ndirs=n_dirs, nfilters=n_filters, take_max=True)([x, Field])


        #x = SyncConvBis(nv=n_v, nrings=n_rings, ndirs=n_dirs, nfilters=128, take_max=True)([x, Frame])

    else:
        #x = Input3d(batch_size=n_batch, nv=n_v, nrings=n_rings, ndirs=n_dirs, nfilters=16, take_max=True)(
        #    [input3d, input3dframes, Exp])
        x = descs
        x = AsyncConvBis(nv=n_v, nrings=n_rings, ndirs=n_dirs, nfilters=n_filters, take_max=True)([x, Exp])
        x = AsyncConvBis(nv=n_v, nrings=n_rings, ndirs=n_dirs, nfilters=n_filters, take_max=True)([x, Exp])
        # x = MaxPooling(take_max=True)([x, Exp1])
        # x = AsyncConvBis(nv=n_v, nrings=n_rings, ndirs=n_dirs, nfilters=n_filters, take_max=True)([x, Exp])
        # x = AsyncConvBis(nv=n_v, nrings=n_rings, ndirs=n_dirs, nfilters=n_filters, take_max=True)([x, Exp])
        # x = AsyncConvBis(nv=n_v, nrings=n_rings, ndirs=n_dirs, nfilters=n_filters, take_max=True)([x, Exp])
        # x = AsyncConvBis(nv=n_v, nrings=n_rings, ndirs=n_dirs, nfilters=128, take_max=True)([x, Exp])
        # x = MaxPooling(take_max=True)([x, Exp1])




    # angular max pooling
    x = Max(axis=1, keepdims=False)(x)
    # x = Dropout(0.25)(x)
    x = Dense(units=128, activation='relu')(x)

    # x = Dropout(0.1)(x)
    # softmax
    x = Dense(units=nb_classes, activation='softmax')(x)
    print('x shape')
    print(K.shape(x))
    # res = ReshapeBis((n_batch, n_v*nb_classes))(x)
    # res = Flatten()(x)
    res = x
    # res = DeShuffleOutput(n_batch, n_v, n_rings, n_dirs, shuffle=shuffle_vertices)([x, rs])

    return descs, input3d, input3dframes, inputExp, inputFrame, res


def gcnn_resnet_v1(n_batch, ratio, n_v, n_rings, n_dirs,
                   fixed_patch_op=False,
                   contributors=None,
                   weights=None,
                   angles=None,
                   parents=None,
                   angular_shifts=None,
                   batch_norm=False,
                   uv=None,
                   input_dim=3,
                   nstacks=1,
                   nresblocks_per_stack=2,
                   nfilters=16,
                   sync_mode='radial_sync',
                   num_classes=10):
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
    # num_res_blocks = int((depth - 2) / 6)
    num_res_blocks = nresblocks_per_stack

    if uv is None:
        inputs = Input(shape=(n_v[0], input_dim), batch_shape=(n_batch,) + (n_v[0], input_dim),
                       name='input_signal')
        x = inputs
    else:
        inputs = Input(shape=(32, 32, 3), batch_shape=(n_batch,) + (32, 32, 3),
                       name='input_signal')
        x = ImageSampling(nbatch=n_batch, uv=uv)(inputs)
    # patch operator

    C = []
    W = []
    TA = []

    P = []
    AS = []

    if fixed_patch_op:
        for i in range(nstacks-1):
            # pool_op = PoolingOperatorFixed(parents=parents[i], angular_shifts=angular_shifts[i], batch_size=n_batch)
            P.append(ConstantTensor(const=parents[i], batch_size=n_batch, dtype='int')([]))
            AS.append(ConstantTensor(const=angular_shifts[i], batch_size=n_batch, dtype='float')([]))

        for stack in range(nstacks):
            # patch_op = PatchOperatorFixed(contributors=contributors[stack],
            #                              weights=weights[stack],
            #                             angles=angles[stack])
            # patch_op_shape = (n_v[i], n_rings[i], n_dirs[i], 3)
            C.append(ConstantTensor(const=contributors[stack], batch_size=n_batch, dtype='int')([]))
            W.append(ConstantTensor(const=weights[stack], batch_size=n_batch, dtype='float')([]))
            TA.append(ConstantTensor(const=angles[stack], batch_size=n_batch, dtype='float')([]))
    else:
        for i in range(nstacks - 1):

            # pool_op = PoolingOperatorFixed(parents=parents[i], angular_shifts=angular_shifts[i], batch_size=n_batch)

            P.append(Input(shape=(n_v[i + 1],),
                           batch_shape=(n_batch,) + (n_v[i + 1],),
                           dtype='int32',
                           name='parents_' + int_to_string(i)))

            AS.append(Input(shape=(n_v[i + 1],),
                            batch_shape=(n_batch,) + (n_v[i + 1],),
                            dtype='float32',
                            name='angular_shifts_' + int_to_string(i)))

        for stack in range(nstacks):
            # patch_op = PatchOperatorFixed(contributors=contributors[stack],
            #                              weights=weights[stack],
            #                             angles=angles[stack])
            patch_op_shape = (n_v[stack], n_rings[stack], n_dirs[stack], 3)
            C.append(Input(shape=patch_op_shape, batch_shape=(n_batch,) + patch_op_shape, dtype='int32',
                           name='contributors_' + int_to_string(stack)))
            W.append(Input(shape=patch_op_shape, batch_shape=(n_batch,) + patch_op_shape, dtype='float32',
                           name='weights_' + int_to_string(stack)))
            TA.append(Input(shape=patch_op_shape, batch_shape=(n_batch,) + patch_op_shape, dtype='float32',
                            name='transport_' + int_to_string(stack)))


    stack_ = 0
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

    # Instantiate the stack of residual units
    for stack in range(nstacks):
        for res_block in range(num_res_blocks):
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # pooling

                # num_filters = 2*num_filters
                if pool_:
                    # num_filters = int(np.sqrt((1. * n_v[stack - 1]) / (1. * n_v[stack])) * num_filters) + 1
                    num_filters = int(np.sqrt(ratio[stack-1] / ratio[stack] + 0.0001)*num_filters)
                    stack_ = stack
                    x = Pooling()([x, P[stack-1], AS[stack-1]])
                else:
                    num_filters *= 2
                    stack_ = 0
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

            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        # if stack > 0:
        #    num_filters = int(np.sqrt((1. * n_v[stack - 1]) / (1. * n_v[stack])) * num_filters)

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AngularMaxPooling(r=1, take_max=True)(x)
    # x = AngularAveragePooling(r=1, take_average=True)(x)
    # x = GlobalAveragePooling1D()(x)
    y = Dense(num_classes,
              kernel_initializer='he_normal',
              name='final_vote')(x)
    outputs = Activation('softmax')(y)

    # Instantiate model.
    if fixed_patch_op:
        model = Model(inputs=inputs, outputs=outputs)
    else:
        I = [inputs]
        for i in range(len(C)):
            I.append(C[i])
            I.append(W[i])
            I.append(TA[i])
        for i in range(len(P)):
            I.append(P[i])
            I.append(AS[i])
        model = Model(inputs=I, outputs=outputs)

    return model


def add_regressor(x, num_channels):
    if K.ndim(x) is 4:
        x = AngularMaxPooling(r=1, take_max=True)(x)
    x = GlobalAveragePooling1D()(x)
    y = Dense(num_channels,
              name='regressor')(x)
    return y


def add_pointwise_prediction(x, num_channels):
    if K.ndim(x) is 4:
        x = AngularMaxPooling(r=1, take_max=True)(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    y = Dense(num_channels, name='pointwise_pred')(x)
    return y


def add_classifier(x, num_classes):
    if K.ndim(x) is 4:
        x = AngularMaxPooling(r=1, take_max=True)(x)

    x = GlobalAveragePooling1D()(x)
    y = Dense(num_classes,
              kernel_initializer='he_normal',
              name='classifier')(x)
    return Activation('softmax')(y)


def add_segmenter(x, num_classes):
    if K.ndim(x) is 4:
        x = AngularMaxPooling(r=1, take_max=True)(x)

    # x = Dense(128, activation='relu')(x)
    # x = Dropout(rate=0.25)(x)
    # x = Dense(32, activation='relu')(x)
    y = Dense(num_classes,
              kernel_initializer='he_normal',
              name='classifier')(x)
    return Activation('softmax')(y)


class GcnnResNetClassifier(object):

    def __init__(self, n_batch, ratios, n_v, n_rings, n_dirs,
                       num_classes,
                       batch_norm=False,
                       input_dim=3,
                       nstacks=1,
                       nresblocks_per_stack=2,
                       nfilters=16,
                       sync_mode='radial_sync'):

        patch_op_inputs = GcnnPatchOperatorInput(n_batch, n_v, n_rings, n_dirs, ratios, name='')

        inputs = Input(shape=(n_v[0], input_dim), batch_shape=(n_batch,) + (n_v[0], input_dim), name='input_signal')

        resnet = GcnnResnet(n_batch, ratios, n_v, n_rings, n_dirs,
                            inputs=inputs,
                            bin_contributors=patch_op_inputs.get_contributors(),
                            weights=patch_op_inputs.get_weights(),
                            transport=patch_op_inputs.get_transport(),
                            parents=patch_op_inputs.get_parents(),
                            angular_shifts=patch_op_inputs.get_angular_shifts(),
                            batch_norm=batch_norm,
                            nstacks=nstacks,
                            nresblocks_per_stack=nresblocks_per_stack,
                            nfilters=16,
                            sync_mode=sync_mode)

        self.output = add_classifier(resnet.get_output(), num_classes)
        I = [inputs] + patch_op_inputs.get_inputs()
        self.input = inputs
        self.model = Model(inputs=I, outputs=self.output)

    def get_model(self):
        return self.model

    def get_input(self):
        return self.input

    def get_output(self):
        return self.output


class GcnnResNetRegressor(object):

    def __init__(self, n_batch, ratios, n_v, n_rings, n_dirs,
                       num_channels,
                       batch_norm=False,
                       input_dim=3,
                       nstacks=1,
                       nresblocks_per_stack=2,
                       nfilters=16,
                       sync_mode='radial_sync'):

        patch_op_inputs = GcnnPatchOperatorInput(n_batch, n_v, n_rings, n_dirs, ratios, name='')

        if n_v is None:
            nv = None
        else:
            nv = n_v[0]

        inputs = Input(shape=(nv, input_dim), batch_shape=(n_batch,) + (nv, input_dim), name='input_signal')

        resnet = GcnnResnet(n_batch, ratios, n_v, n_rings, n_dirs,
                            inputs=inputs,
                            bin_contributors=patch_op_inputs.get_contributors(),
                            weights=patch_op_inputs.get_weights(),
                            transport=patch_op_inputs.get_transport(),
                            parents=patch_op_inputs.get_parents(),
                            angular_shifts=patch_op_inputs.get_angular_shifts(),
                            batch_norm=batch_norm,
                            nstacks=nstacks,
                            nresblocks_per_stack=nresblocks_per_stack,
                            nfilters=nfilters,
                            sync_mode=sync_mode)

        self.output = add_regressor(resnet.get_output(), num_channels)
        I = [inputs] + patch_op_inputs.get_inputs()
        self.input = inputs
        self.model = Model(inputs=I, outputs=self.output)

    def get_model(self):
        return self.model

    def get_input(self):
        return self.input

    def get_output(self):
        return self.output


class GcnnResNetSegmenter(object):

    def __init__(self, n_batch, ratios, n_v, n_rings, n_dirs,
                       num_classes,
                       batch_norm=False,
                       input_dim=3,
                       nstacks=1,
                       nresblocks_per_stack=2,
                       nfilters=16,
                       sync_mode='radial_sync'):

        patch_op_inputs = GcnnPatchOperatorInput(n_batch, n_v, n_rings, n_dirs, ratios, name='')

        if n_v is None:
            nv = None
        else:
            nv = n_v[0]

        inputs = Input(shape=(nv, input_dim), batch_shape=(n_batch,) + (nv, input_dim), name='input_signal')

        resnet = GcnnResnet(n_batch, ratios, n_v, n_rings, n_dirs,
                            inputs=inputs,
                            bin_contributors=patch_op_inputs.get_contributors(),
                            weights=patch_op_inputs.get_weights(),
                            transport=patch_op_inputs.get_transport(),
                            parents=patch_op_inputs.get_parents(),
                            angular_shifts=patch_op_inputs.get_angular_shifts(),
                            batch_norm=batch_norm,
                            nstacks=nstacks,
                            nresblocks_per_stack=nresblocks_per_stack,
                            nfilters=nfilters,
                            sync_mode=sync_mode)

        self.output = add_segmenter(resnet.get_output(), num_classes)
        I = [inputs] + patch_op_inputs.get_inputs()
        self.input = inputs
        self.model = Model(inputs=I, outputs=self.output)

    def get_model(self):
        return self.model

    def get_input(self):
        return self.input

    def get_output(self):
        return self.output


class GcnnUresNetSegmenter(object):

    def __init__(self, n_batch, ratios, n_v, n_rings, n_dirs,
                       num_classes,
                       batch_norm=False,
                       input_dim=3,
                       nstacks=1,
                       nresblocks_per_stack=2,
                       nfilters=16,
                       sync_mode='radial_sync',
                       global_3d=False):

        patch_op_inputs = GcnnPatchOperatorInput(n_batch, n_v, n_rings, n_dirs, ratios, name='')

        if n_v is None:
            nv = None
        else:
            nv = n_v[0]

        inputs = Input(shape=(nv, input_dim), batch_shape=(n_batch,) + (nv, input_dim), name='input_signal')

        if global_3d:
            x = Local3dGeodesicConv(nfilters=nfilters, nv=nv, ndirs=n_dirs[0], nrings=n_rings[0],
                 use_global_context=True,
                 ntraindirs=None,
                 take_max=False,
                 activation='linear',
                 use_bias=False)([inputs,
                                 patch_op_inputs.get_contributors()[0],
                                 patch_op_inputs.get_weights()[0]])
        else:
            x = inputs

        unet = GcnnUResnet(n_batch, ratios, n_v, n_rings, n_dirs,
                           inputs=x,
                           bin_contributors=patch_op_inputs.get_contributors(),
                           weights=patch_op_inputs.get_weights(),
                           transport=patch_op_inputs.get_transport(),
                           parents=patch_op_inputs.get_parents(),
                           angular_shifts=patch_op_inputs.get_angular_shifts(),
                           batch_norm=batch_norm,
                           nstacks=nstacks,
                           nresblocks_per_stack=nresblocks_per_stack,
                           nfilters=nfilters,
                           sync_mode=sync_mode)

        self.encoder = unet.get_encoder()
        self.decoder = unet.get_decoder()

        self.output = add_segmenter(self.decoder.get_output(), num_classes)
        I = self.encoder.get_inputs_list()
        I[0] = inputs
        self.input = inputs
        self.model = Model(inputs=I, outputs=self.output)

    def get_model(self):
        return self.model

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_input(self):
        return self.input

    def get_output(self):
        return self.output


class GcnnUresNet(object):

    def __init__(self, n_batch, ratios, n_v, n_rings, n_dirs,
                       task,
                       output_channels,
                       batch_norm=False,
                       input_dim=3,
                       nstacks=1,
                       nresblocks_per_stack=2,
                       nfilters=16,
                       sync_mode='radial_sync',
                       global_3d=False):

        patch_op_inputs = GcnnPatchOperatorInput(n_batch, n_v, n_rings, n_dirs, ratios, name='')

        if n_v is None:
            nv = None
        else:
            nv = n_v[0]

        inputs = Input(shape=(nv, input_dim), batch_shape=(n_batch,) + (nv, input_dim), name='input_signal')

        if global_3d:
            x = Local3dGeodesicConv(nfilters=nfilters, nv=nv, ndirs=n_dirs[0], nrings=n_rings[0],
                 use_global_context=True,
                 ntraindirs=None,
                 take_max=False,
                 activation='linear',
                 use_bias=False)([inputs,
                                 patch_op_inputs.get_contributors()[0],
                                 patch_op_inputs.get_weights()[0]])
        else:
            x = inputs

        unet = GcnnUResnet(n_batch, ratios, n_v, n_rings, n_dirs,
                           inputs=x,
                           bin_contributors=patch_op_inputs.get_contributors(),
                           weights=patch_op_inputs.get_weights(),
                           transport=patch_op_inputs.get_transport(),
                           parents=patch_op_inputs.get_parents(),
                           angular_shifts=patch_op_inputs.get_angular_shifts(),
                           batch_norm=batch_norm,
                           nstacks=nstacks,
                           nresblocks_per_stack=nresblocks_per_stack,
                           nfilters=nfilters,
                           sync_mode=sync_mode)

        self.encoder = unet.get_encoder()
        self.decoder = unet.get_decoder()

        if task == 'segmentation':
            self.output = add_segmenter(self.decoder.get_output(), output_channels)
        elif task == 'pointwise_pred':
            self.output = add_pointwise_prediction(self.decoder.get_output(), output_channels)
        elif task == 'classification':
            self.output = add_classifier(self.decoder.get_output(), output_channels)
        elif task == 'regression':
            self.output = add_regressor(self.decoder.get_output(), output_channels)
        else:
            raise ValueError('Unknown task')

        I = self.encoder.get_inputs_list()
        I[0] = inputs
        self.input = inputs
        self.model = Model(inputs=I, outputs=self.output)

    def get_model(self):
        return self.model

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_input(self):
        return self.input

    def get_output(self):
        return self.output


class MlpSegmenter(object):

    def __init__(self, n_batch, n_v,
                       num_classes,
                       input_dim=3,
                       nstacks=1,
                       nresblocks_per_stack=2,
                       nfilters=16):

        num_filters = nfilters

        self.input = Input(shape=(n_v[0], input_dim), batch_shape=(n_batch,) + (n_v[0], input_dim), name='input_signal')
        x = self.input
        """
        x = Dense(units=num_filters, activation=None)(self.input)
        BatchNormalization()(x)
        x = Activation('relu')(x)
        """

        for i in range(nstacks):
            for j in range(nresblocks_per_stack):
                if i > 0 and j == 0:
                    num_filters *= 2

                y = Dense(units=num_filters, activation=None)(x)
                y = BatchNormalization()(y)
                y = Activation('linear')(y)

                y = Dense(units=num_filters, activation=None)(y)
                y = BatchNormalization()(y)
                x = Activation('linear')(y)

                """
                if i > 0 and j == 0:  # first layer but not first stack
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x = Dense(units=num_filters, use_bias=False, activation=None)(x)
                """

                # x = keras.layers.add([x, y])

        # x = Dense(units=20, activation=None)(x)
        # x = BatchNormalization()(x)
        # x = Activation('relu')(x)

        self.output = add_segmenter(x, num_classes)

        self.model = Model(inputs=self.input, outputs=self.output)

    def get_model(self):
        return self.model

    def get_input(self):
        return self.input

    def get_output(self):
        return self.output



