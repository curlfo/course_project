import numpy as np
import random
from scipy.ndimage.interpolation import map_coordinates, affine_transform
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def makeTransform(matrices):
    """
    Creates one transformation matrix from matricies list by multiplying.
    :param matrices: list of matrices
    :return: product of matrices
    """
    ret = matrices[0].copy()
    for xx in matrices[1:]:
        ret = ret.dot(xx)
    return ret


def affine_transformation_3d(image3d, pshiftXYZ, protCntXYZ, protAngleXYZ, pscaleXYZ, pcropSizeXYZ,
                             pshear=(0., 0., 0., 0., 0., 0.),
                             isRandomizeRot=False,
                             isDebug=False,
                             pmode='constant',
                             pval=0,
                             porder=0):
    """
    based on https://github.com/gakarak/BTBDB_ImageAnalysisSubPortal

    scipy-based 3d image transformation: for data augumentation

    :param image3d: input 3d-image with 1 or more channels (with shape like [sizX, sizY, sizZ]
            or [sizX, sizY, sizZ, num_channels])
    :param pshiftXYZ: shift of coordinates: (dx, dy, dz)
    :param protCntXYZ: rotation center: (x0, y0, z0)
    :param protAngleXYZ: rotation angle (anti-clock-wise), like: (angle_x, angle_y, angle_z)
    :param pscaleXYZ: cale transformation: (sx, sy, sz)
    :param pcropSizeXYZ: output size of cropped image, like: [outSizeX, outSizeY, outSizeZ].If None,
            then output image shape is equal to input image shape
    :param pshear: shear-transform 3D coefficients. Two possible formats:
            - 6-dimensional vector, like: (Syx, Szx, Sxy, Szy, Sxz, Syz)
            - 3x3 matrix, like:
                [  1, Syx, Szx]
                [Sxy,   1, Szy]
                [Sxz, Syz,   1]
    :param isRandomizeRot: if True - random shuffle order of X/Y/Z rotation
    :param isDebug: if True - show the debug visualization
    :param pmode: parameter is equal 'mode' :parameter in scipy.ndimage.interpolation.affine_transform
    :param pval: parameter is equal 'val' :parameter in scipy.ndimage.interpolation.affine_transform
    :param porder: parameter is equal 'order' :parameter in scipy.ndimage.interpolation.affine_transform
    :return: transformed 3d image
    """
    nshp=3
    # (1) precalc parameters
    angRadXYZ = (np.pi / 180.) * np.array(protAngleXYZ)
    cosaXYZ = np.cos(angRadXYZ)
    sinaXYZ = np.sin(angRadXYZ)
    # (2) prepare separate affine transformation matrices
    # (2.0) shift matrices
    matShiftXYZ = np.array([
        [1., 0., 0., +pshiftXYZ[0]],
        [0., 1., 0., +pshiftXYZ[1]],
        [0., 0., 1., +pshiftXYZ[2]],
        [0., 0., 0.,            1.]
    ])
    # (2.1) shift-matrices for rotation: backward and forward
    matShiftB = np.array([
        [1., 0., 0., -protCntXYZ[0]],
        [0., 1., 0., -protCntXYZ[1]],
        [0., 0., 1., -protCntXYZ[2]],
        [0., 0., 0.,          1.]
    ])
    matShiftF = np.array([
        [1., 0., 0., +protCntXYZ[0]],
        [0., 1., 0., +protCntXYZ[1]],
        [0., 0., 1., +protCntXYZ[2]],
        [0., 0., 0.,          1.]
    ])
    # (2.2) partial and full-rotation matrix
    lstMatRotXYZ = []
    for ii in range(len(angRadXYZ)):
        cosa = cosaXYZ[ii]
        sina = sinaXYZ[ii]
        if ii==0:
            # Rx
            tmat = np.array([
                [1.,    0.,    0., 0.],
                [0., +cosa, -sina, 0.],
                [0., +sina, +cosa, 0.],
                [0.,    0.,    0., 1.]
            ])
        elif ii==1:
            # Ry
            tmat = np.array([
                [+cosa,  0., +sina,  0.],
                [   0.,  1.,    0.,  0.],
                [-sina,  0., +cosa,  0.],
                [   0.,  0.,    0.,  1.]
            ])
        else:
            # Rz
            tmat = np.array([
                [+cosa, -sina,  0.,  0.],
                [+sina, +cosa,  0.,  0.],
                [   0.,    0.,  1.,  0.],
                [   0.,    0.,  0.,  1.]
            ])
        lstMatRotXYZ.append(tmat)
    if isRandomizeRot:
        random.shuffle(lstMatRotXYZ)
    matRotXYZ = lstMatRotXYZ[0].copy()
    for mm in lstMatRotXYZ[1:]:
        matRotXYZ = matRotXYZ.dot(mm)
    # (2.3) scale matrix
    sx,sy,sz = pscaleXYZ
    matScale = np.array([
        [sx, 0., 0., 0.],
        [0., sy, 0., 0.],
        [0., 0., sz, 0.],
        [0., 0., 0., 1.],
    ])
    # (2.4) shear matrix
    if pshear is not None:
        if len(pshear) == 6:
            matShear = np.array([
                [1., pshear[0], pshear[1], 0.],
                [pshear[2], 1., pshear[3], 0.],
                [pshear[4], pshear[5], 1., 0.],
                [0., 0., 0., 1.]
            ])
        else:
            matShear = np.eye(4, 4)
            pshear = np.array(pshear)
            if pshear.shape == (3, 3):
                matShear[:3, :3] = pshear
            elif pshear.shape == (4, 4):
                matShear = pshear
            else:
                raise Exception('Invalid shear-matrix format: [{0}]'.format(pshear))
    else:
        matShear = np.eye(4, 4)
    # (3) build total-matrix
    if pcropSizeXYZ is None:
        # matTotal = matShiftF.dot(matRotXYZ.dot(matScale.dot(matShiftB)))
        matTotal = makeTransform([matShiftF, matRotXYZ, matShear, matScale, matShiftB])
        pcropSizeXYZ = image3d.shape[:nshp]
    else:
        matShiftCropXYZ = np.array([
            [1., 0., 0., pcropSizeXYZ[0] / 2.],
            [0., 1., 0., pcropSizeXYZ[1] / 2.],
            [0., 0., 1., pcropSizeXYZ[2] / 2.],
            [0., 0., 0.,                   1.]
        ])
        # matTotal = matShiftCropXYZ.dot(matRotXYZ.dot(matScale.dot(matShiftB)))
        matTotal = makeTransform([matShiftCropXYZ, matRotXYZ, matShear, matScale, matShiftB])
    # (3.1) shift after rot-scale transformation
    matTotal = matShiftXYZ.dot(matTotal)
    # (3.2) invert matrix for back-projected mapping
    matTotalInv = np.linalg.inv(matTotal)
    # (4) warp image with total affine-transform
    idxXYZ = np.indices(pcropSizeXYZ).reshape(nshp, -1)
    idxXYZH = np.insert(idxXYZ, nshp, values=[1], axis=0)
    idxXYZT = matTotalInv.dot(idxXYZH)[:nshp, :]
    # (5) processing 3D layer-by-layer
    if image3d.ndim>nshp:
        tret = []
        for ii in range(image3d.shape[-1]):
            tret.append(map_coordinates(image3d[:, :, :, ii], idxXYZT, order=porder, cval=pval, mode=pmode).reshape(pcropSizeXYZ))
        tret = np.dstack(tret)
    else:
        #tret = map_coordinates(image3d, idxXYZT, order=porder, cval=pval, mode=pmode).reshape(pcropSizeXYZ)
        tret = affine_transform(image3d, matTotal, order=porder, cval=pval, mode=pmode).reshape(pcropSizeXYZ)
    # (6) Debug
    if isDebug:
        protCntXYZ = np.array(protCntXYZ)
        protCntXYZPrj = matTotal.dot(list(protCntXYZ) + [1])[:nshp]
        print (':: Total matrix:\n{0}'.format(matTotal))
        print ('---')
        print (':: Total matrix inverted:\n{0}'.format(matTotalInv))
        s0, s1, s2 = image3d.shape
        s0n, s1n, s2n = tret.shape
        #
        plt.subplot(3, 3, 1 + 0*nshp)
        plt.imshow(image3d[s0 // 2, :, :])
#         plt.gcf().gca().add_artist(plt.Circle(protCntXYZ[[1, 2]], 5, edgecolor='r', fill=False))
        plt.subplot(3, 3, 2 + 0*nshp)
        plt.imshow(image3d[:, s1 // 2, :])
#         plt.gcf().gca().add_artist(plt.Circle(protCntXYZ[[0, 2]], 5, edgecolor='r', fill=False))
        plt.subplot(3, 3, 3 + 0*nshp)
        plt.imshow(image3d[:, :, s2 // 2])
#         plt.gcf().gca().add_artist(plt.Circle(protCntXYZ[[0, 1]], 5, edgecolor='r', fill=False))
        #
        plt.subplot(3, 3, 1 + 1*nshp)
        plt.imshow(tret[s0n // 2, :, :])
#         plt.gcf().gca().add_artist(plt.Circle(protCntXYZPrj[[1, 2]], 5, edgecolor='r', fill=False))
        plt.subplot(3, 3, 2 + 1*nshp)
        plt.imshow(tret[:, s1n // 2, :])
#         plt.gcf().gca().add_artist(plt.Circle(protCntXYZPrj[[0, 2]], 5, edgecolor='r', fill=False))
        plt.subplot(3, 3, 3 + 1*nshp)
        plt.imshow(tret[:, :, s2n // 2])
#         plt.gcf().gca().add_artist(plt.Circle(protCntXYZPrj[[0, 1]], 5, edgecolor='r', fill=False))
        #
        plt.subplot(3, 3, 1 + 2 * nshp)
        plt.imshow(np.sum(tret, axis=0))
#         plt.gcf().gca().add_artist(plt.Circle(protCntXYZPrj[[1, 2]], 5, edgecolor='r', fill=False))
        plt.subplot(3, 3, 2 + 2 * nshp)
        plt.imshow(np.sum(tret, axis=1))
#         plt.gcf().gca().add_artist(plt.Circle(protCntXYZPrj[[0, 2]], 5, edgecolor='r', fill=False))
        plt.subplot(3, 3, 3 + 2 * nshp)
        plt.imshow(np.sum(tret, axis=2))
#         plt.gcf().gca().add_artist(plt.Circle(protCntXYZPrj[[0, 1]], 5, edgecolor='r', fill=False))
        plt.show()
    return tret


def augment_image_3d(image, isDebug=False):
    """
        Augments image using randomly generated transformation.

        :param image: input 3d-image with 1 or more channels (with shape like [sizX, sizY, sizZ]
            or [sizX, sizY, sizZ, num_channels])
        :param isDebug: flag for showing debug information

        :return: augmented image, same shape as input
    """

    protAngleXYZ, protCntXYZ, pscaleXYZ, pshear, pshiftXYZ = get_transform(isDebug)

    transformed_img = affine_transformation_3d(image, pshiftXYZ, protCntXYZ, protAngleXYZ, pscaleXYZ, None, pshear,
                                               isRandomizeRot=True, isDebug=isDebug, pval=np.min(image), porder=3)
    return transformed_img


def augment_image_masked_3d(image, mask, isDebug=False):
    """
        Augments image and its mask using the same randomly generated transformation.

        :param image: input 3d-image with 1 or more channels (with shape like [sizX, sizY, sizZ]
            or [sizX, sizY, sizZ, num_channels])
        :param mask: input 3d-image with 1 or more channels (with shape like [sizX, sizY, sizZ]
            or [sizX, sizY, sizZ, num_channels])
        :param isDebug: flag for showing debug information

        :return:
            augmented image, same shape as input
            augmented mask, same shape as input

    """
    protAngleXYZ, protCntXYZ, pscaleXYZ, pshear, pshiftXYZ = get_transform(isDebug)

    transformed_img = affine_transformation_3d(image, pshiftXYZ, protCntXYZ, protAngleXYZ, pscaleXYZ, None, pshear,
                                               isRandomizeRot=False, isDebug=isDebug, pval=np.min(image), porder=3)
    transformed_mask = affine_transformation_3d(mask, pshiftXYZ, protCntXYZ, protAngleXYZ, pscaleXYZ, None, pshear,
                                                isRandomizeRot=False, isDebug=isDebug, pval=np.min(image), porder=3)
    return transformed_img, transformed_mask


def get_transform(debug):
    """
        Generated transformation for augmentation.

        :param debug: flag for showing debug information

        :return:
            transformation paramers, as follows:

            rotation angles (X, Y, Z),
            rotation center (X, Y, Z),
            scales vector (X, Y, Z),
            shear vector (6-coords format),
            shift vector (X, Y, Z)

    """
    SIZE_XY = 256
    SIZE_Z = 128
    MAX_SHIFT = 0.05
    MAX_ROT_SHIFT = 0.05
    MAX_ROT_ANGLE = 5
    MAX_SCALE = 0.05
    MAX_SHEAR = 0.01
    augm_hot_vec = np.random.randint(2, size=5)
    # augm_hot_vec = [1, 0, 0, 0, 0]
    if augm_hot_vec[0]:
        pshiftXYZ = np.concatenate(
            (np.random.uniform(0, SIZE_XY * MAX_SHIFT, 2), np.random.uniform(0, SIZE_Z * MAX_SHIFT, 1))).astype(int)
    else:
        pshiftXYZ = [0, 0, 0]
    if debug:
        print('shift:', pshiftXYZ)
    if augm_hot_vec[1]:
        protCntXYZ = np.concatenate((
            [SIZE_XY / 2, SIZE_XY / 2] - np.random.uniform(-SIZE_XY * MAX_ROT_SHIFT, SIZE_XY * MAX_ROT_SHIFT, 2),
            [SIZE_Z / 2] - np.random.uniform(-SIZE_Z * MAX_ROT_SHIFT, SIZE_Z * MAX_ROT_SHIFT, 1))).astype(int)
    else:
        protCntXYZ = [SIZE_XY / 2, SIZE_XY / 2, SIZE_Z / 2]
    if debug:
        print('rot center:', protCntXYZ)
    if augm_hot_vec[2]:
        protAngleXYZ = np.random.uniform(-MAX_ROT_ANGLE, MAX_ROT_ANGLE, 3)
    else:
        protAngleXYZ = [0, 0, 0]
    if debug:
        print('rot angle:', protAngleXYZ)
    if augm_hot_vec[3]:
        pscaleXYZ = np.random.uniform(1 - MAX_SCALE, 1 + MAX_SCALE, 3)
    else:
        pscaleXYZ = [1, 1, 1]
    if debug:
        print('scale:', pscaleXYZ)
    if augm_hot_vec[4]:
        pshear = np.random.uniform(-MAX_SHEAR, MAX_SHEAR, 6)
    else:
        pshear = [0] * 6
    if debug:
        print('shear:', pshear)
    return protAngleXYZ, protCntXYZ, pscaleXYZ, pshear, pshiftXYZ
