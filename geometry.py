import numpy as np
import math


def rotation_matrix_zyz(ang):
    phi = ang[0]
    theta = ang[1]
    psi_t = ang[2]

    # first rotate about z axis for angle psi_t
    a1 = rotation_matrix_axis(2, psi_t)
    a2 = rotation_matrix_axis(1, theta)
    a3 = rotation_matrix_axis(2, phi)

    # for matrix left multiplication
    rm = a3.dot(a2).dot(a1)

    # note: transform because tformarray use right matrix multiplication
    rm = rm.transpose()

    return rm


def rotation_matrix_axis(dim, theta):
    """
    following are left handed system (clockwise rotation)
    IMPORTANT: different to MATLAB version, this dim starts from 0, instead of 1
    """
    # x-axis
    if dim == 0:
        rm = np.array(
            [[1.0, 0.0, 0.0], [0.0, math.cos(theta), -math.sin(theta)], [0.0, math.sin(theta), math.cos(theta)]])
    # y-axis
    elif dim == 1:
        rm = np.array(
            [[math.cos(theta), 0.0, math.sin(theta)], [0.0, 1.0, 0.0], [-math.sin(theta), 0.0, math.cos(theta)]])
    # z-axis
    elif dim == 2:
        rm = np.array(
            [[math.cos(theta), -math.sin(theta), 0.0], [math.sin(theta), math.cos(theta), 0.0], [0.0, 0.0, 1.0]])
    else:
        raise

    return rm


def random_rotation_matrix():
    """generate a random rotation matrix using SVD on a random matrix"""
    m = np.random.random((3, 3))
    u, s, v = np.linalg.svd(m)

    return u


def random_rotation_angle_zyz():
    rm = random_rotation_matrix()
    return rotation_matrix_zyz_normalized_angle(rm)


def rotation_matrix_zyz_normalized_angle(rm):
    assert (all(np.isreal(rm.flatten())));
    assert (rm.shape == (3, 3));

    cos_theta = rm[2, 2]
    if np.abs(cos_theta) > 1.0:
        # warning(sprintf('cos_theta %g', cos_theta));
        cos_theta = np.sign(cos_theta)

    theta = np.arctan2(np.sqrt(1.0 - (cos_theta * cos_theta)), cos_theta)

    # use a small epslon to increase numerical stability when abs(cos_theta) is very close to 1!!!!
    if np.abs(cos_theta) < (1.0 - 1e-10):
        phi = np.arctan2(rm[2, 1], rm[2, 0])
        psi_t = np.arctan2(rm[1, 2], -rm[0, 2])
    else:
        theta = 0.0
        phi = 0.0
        psi_t = np.arctan2(rm[0, 1], rm[1, 1])

    ang = np.array([phi, theta, psi_t], dtype=np.float64)

    return ang


def reverse_transform(rm, loc_r):
    rev_rm = rm.T
    rev_loc_r = (- np.dot(loc_r, rev_rm))
    return rev_rm, rev_loc_r


def reverse_transform_ang_loc(ang, loc_r):
    rm = rotation_matrix_zyz(ang)
    (rev_rm, rev_loc_r) = reverse_transform(rm, loc_r)
    return rotation_matrix_zyz_normalized_angle(rev_rm), rev_loc_r

import scipy.ndimage.interpolation as SNI

def fft_mid_co(siz):
    assert all((np.mod(siz, 1) == 0))
    assert all((np.array(siz) > 0))
    mid_co = np.zeros(len(siz))
    for i in range(len(mid_co)):
        m = siz[i]
        mid_co[i] = np.floor((m / 2))
    return mid_co

def rotate(v, angle=None, rm=None, c1=None, c2=None, loc_r=None, siz2=None, default_val=float('NaN')):
    if angle is not None:
        assert (rm is None)
        angle = np.array(angle, dtype=np.float64).flatten()
        rm = rotation_matrix_zyz(angle)
    if rm is None:
        rm = np.eye(v.ndim)
    siz1 = np.array(v.shape, dtype=np.float64)
    if c1 is None:
        c1 = ((siz1 - 1) / 2.0)
    else:
        c1 = c1.flatten()
    assert (c1.shape == (3,))
    if siz2 is None:
        siz2 = siz1
    siz2 = np.array(siz2, dtype=np.float64)
    if c2 is None:
        c2 = ((siz2 - 1) / 2.0)
    else:
        c2 = c2.flatten()
    assert (c2.shape == (3,))
    if loc_r is not None:
        loc_r = np.array(loc_r, dtype=np.float64).flatten()
        assert (loc_r.shape == (3,))
        c2 += loc_r
    c = ((- rm.dot(c2)) + c1)
    vr = SNI.affine_transform(input=v, matrix=rm, offset=c, output_shape=siz2.astype(np.int32), cval=default_val)
    return vr


def rotate3d_zyz(data, angle=None, rm=None, center=None, order=2, cval=0.0):
    """Rotate a 3D data using ZYZ convention (phi: z1, the: x, psi: z2)."""
    from scipy import mgrid
    # Figure out the rotation center
    if center is None:
        cx = data.shape[0] / 2
        cy = data.shape[1] / 2
        cz = data.shape[2] / 2
    else:
        assert len(center) == 3
        (cx, cy, cz) = center

    if rm is None:
        Inv_R = rotation_matrix_zyz(angle)
    else:
        Inv_R = rm

    grid = mgrid[-cx:data.shape[0] - cx, -cy:data.shape[1] - cy, -cz:data.shape[2] - cz]
    temp = grid.reshape((3, np.int32(grid.size / 3)))
    temp = np.dot(Inv_R, temp)
    grid = np.reshape(temp, grid.shape)
    grid[0] += cx
    grid[1] += cy
    grid[2] += cz

    # Interpolation
    from scipy.ndimage import map_coordinates
    d = map_coordinates(data, grid, order=order, cval=cval)

    return d


def translate3d_zyz(data, dx=0, dy=0, dz=0, order=2, cval=0.0):
    """Translate the data.
    @param
        data: data to be shifted.
        dx: translation along x-axis.
        dy: translation along y-axis.
        dz: translation along z-axis.
    
    @return: the data after translation.
    """
    from scipy import mgrid
    if dx == 0 and dy == 0 and dz == 0:
        return data

    # from scipy.ndimage.interpolation import shift
    # res = shift(data, [dx, dy, dz])
    # return res
    grid = mgrid[0.:data.shape[0], 0.:data.shape[1], 0.:data.shape[2]]
    grid[0] -= dx
    grid[1] -= dy
    grid[2] -= dz
    from scipy.ndimage import map_coordinates
    d = map_coordinates(data, grid, order=order, cval=cval)

    return d


def rotate_interpolate_pad_mean(v, angle=None, rm=None, loc_r=None):
    cval = v.mean()

    vr = rotate3d_zyz(v, angle=angle, cval=cval)

    vr = translate3d_zyz(vr, loc_r[0], loc_r[1], loc_r[2], cval=cval)

    return vr


def rotate_pad_mean(v, angle=None, rm=None, c1=None, c2=None, loc_r=None, siz2=None):
    vr = rotate(v, angle=angle, rm=rm, c1=c1, c2=c2, loc_r=loc_r, siz2=siz2, default_val=float('NaN'))
    if False:
        vr[np.logical_not(np.isfinite(vr))] = vr[np.isfinite(vr)].mean()
    else:
        vr[np.logical_not(np.isfinite(vr))] = v.mean()
    return vr


def rotate_pad_zero(v, angle=None, rm=None, c1=None, c2=None, loc_r=None, siz2=None):
    vr = rotate(v, angle=angle, rm=rm, c1=c1, c2=c2, loc_r=loc_r, siz2=siz2, default_val=float('NaN'))

    vr[np.logical_not(np.isfinite(vr))] = 0.0

    return vr


def rotate_mask(v, angle=None, rm=None):
    c1 = fft_mid_co(v.shape)
    c2 = np.copy(c1)
    vr = rotate(v, angle=angle, rm=rm, c1=c1, c2=c2, default_val=float('NaN'))
    vr[np.logical_not(np.isfinite(vr))] = 0.0
    return vr
