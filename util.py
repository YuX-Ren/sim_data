"""
Utility function for generate different geometirc models
"""

import numpy as np
import geometry
from scipy.spatial import Delaunay

from scipy.ndimage import gaussian_filter

def grid_displacement_to_center(size, mid_co=None):
    """grid functions"""
    size = np.array(size, dtype=np.float64)
    assert size.ndim == 1

    if mid_co is None:
        # IMPORTANT: following python convension, in index starts from 0 to size-1!!!
        # So (siz-1)/2 is real symmetry center of the volume
        mid_co = (np.array(size) - 1) / 2

    if size.size == 3:
        # construct a gauss function whose center is at center of volume
        grid = np.mgrid[0:size[0], 0:size[1], 0:size[2]]

        for dim in range(3):
            grid[dim, :, :, :] -= mid_co[dim]

    elif size.size == 2:
        # construct a gauss function whose center is at center of volume
        grid = np.mgrid[0:size[0], 0:size[1]]

        for dim in range(2):
            grid[dim, :, :] -= mid_co[dim]

    else:
        assert False

    return grid


def grid_distance_sq_to_center(grid):
    dist_sq = np.zeros(grid.shape[1:])
    if grid.ndim == 4:
        for dim in range(3):
            dist_sq += np.squeeze(grid[dim, :, :, :]) ** 2
    elif grid.ndim == 3:
        for dim in range(2):
            dist_sq += np.squeeze(grid[dim, :, :]) ** 2
    else:
        assert False

    return dist_sq


def grid_distance_to_center(grid):
    dist_sq = grid_distance_sq_to_center(grid)
    return np.sqrt(dist_sq)


def generate_hollow_sphere(diameter, density=1.0):
    radius = diameter // 2
    size = diameter + 2  # Add extra space for boundary effects
    center = size // 2

    # Create a 3D grid
    x, y, z = np.indices((size, size, size))
    distance = np.sqrt((x - center)**2 + (y - center)**2 + (z - center)**2)
    
    # Define the sphere
    outer_sphere = (distance <= radius).astype(float)
    inner_radius = radius - 4
    inner_sphere = (distance <= inner_radius).astype(float)
    
    # Create hollow sphere by subtracting the inner sphere from the outer sphere
    hollow_sphere = outer_sphere - inner_sphere
    # Apply a constant density
    density_map = hollow_sphere * density
    
    return density_map

def smooth_density_map(density_map, sigma=2.0):
    # Apply Gaussian filter to smooth the density map
    smoothed_map = gaussian_filter(density_map, sigma=sigma)
    return smoothed_map

def sphere(res):
    phi, theta = np.mgrid[0:np.pi:complex(res), 0:2*np.pi:complex(res)]
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return x, y, z

def generate_prism(dim_siz):
    dim_size = 64
    X, Y, Z = np.meshgrid(np.linspace(-(dim_size-1)/2, (dim_size-1)/2, dim_size), np.linspace(-(dim_size-1)/2, (dim_size-1)/2, dim_size), np.linspace(-(dim_size-1)/2, (dim_size-1)/2, dim_size))
    QP = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    P0 = np.random.rand() * 10 + 20
    P1 = np.random.rand() * 2.5 + 2.5
    tipX, tipY, tipZ = sphere(100)
    tipX = tipX.ravel() * P1
    tipY = tipY.ravel() * P1
    tipZ = tipZ.ravel() * P1
    T = np.vstack([np.column_stack([tipX - P0/2, tipY, tipZ]),
                    np.column_stack([tipX + P0/2, tipY, tipZ]),
                    np.column_stack([tipX, tipY + P0/2*np.sqrt(3), tipZ])])

    T[:, 1] -= P0/2*np.sqrt(3)/3
    S = Delaunay(T)
    indexIntersect = S.find_simplex(QP)
    mask = indexIntersect!=-1
    mask = mask.reshape((dim_size, dim_size, dim_size)).astype(float)
    density = gaussian_filter(mask, sigma=1)
    return density

def gauss_function(size, sigma):
    grid = grid_displacement_to_center(size)
    dist_sq = grid_distance_sq_to_center(grid)

    del grid

    # gauss function
    g = (1 / ((2 * np.pi) ** (3.0 / 2.0) * (sigma ** 3))) * np.exp(- (dist_sq) / (2.0 * (sigma ** 2)))

    return g


def generate_toy_model(dim_siz=64, model_id=0):
    """translated from SphericalHarmonicsUtil.generate_toy_vol()"""
    siz = np.array([dim_siz, dim_siz, dim_siz])

    mid = siz / 2.0

    xg = np.mgrid[0:siz[0], 0:siz[1], 0:siz[2]]

    if model_id == 0:
        # four gauss functions
        short_dia = 0.4
        mid_dia = 0.8
        long_dia = 1.2

        e0 = generate_toy_model__gaussian(dim_siz=dim_siz, xg=xg, xm=(mid + np.array([siz[0] / 4.0, 0.0, 0.0])),
                                          dias=[long_dia, short_dia, short_dia])
        e1 = generate_toy_model__gaussian(dim_siz=dim_siz, xg=xg, xm=(mid + np.array([0.0, siz[1] / 4.0, 0.0])),
                                          dias=[short_dia, long_dia, short_dia])
        e2 = generate_toy_model__gaussian(dim_siz=dim_siz, xg=xg, xm=(mid + np.array([0.0, 0.0, siz[2] / 4.0])),
                                          dias=[short_dia, short_dia, long_dia])

        e3 = geometry.rotate_pad_zero(np.array(e0, order='F'), angle=np.array([np.pi / 4.0, 0.0, 0.0]),
                                loc_r=np.array([0.0, 0.0, 0.0]))

        e = e0 + e1 + e2 + e3

    return e


def generate_toy_model__gaussian(dim_siz, xg, xm, dias):
    x = np.zeros(xg.shape)
    for dim_i in range(3):
        x[dim_i] = xg[dim_i] - xm[dim_i]
    xs = np.array([x[0] / (dim_siz * dias[0]), x[1] / (dim_siz * dias[1]), x[2] / (dim_siz * dias[2])])
    e = np.exp(- np.sum(xs * x, axis=0))

    return e


def sphere_mask(shape, center=None, radius=None, smooth_sigma=None):
    shape = np.array(shape)

    v = np.zeros(shape)

    if center is None:
        # IMPORTANT: following python convension, in index starts from 0 to
        # size-1 !!! So (siz-1)/2 is real symmetry center of the volume
        center = (shape - 1) / 2.0

    center = np.array(center)

    if radius is None:
        radius = np.min(shape / 2.0)

    grid = grid_displacement_to_center(shape, mid_co=center)
    dist = grid_distance_to_center(grid)

    v[dist <= radius] = 1.0

    if smooth_sigma is not None:

        assert smooth_sigma > 0
        v_s = np.exp(-((dist - radius) / smooth_sigma) ** 2)
        # use a cutoff of -3 looks nicer, although the tom toolbox uses -2
        v_s[v_s < np.exp(-3)] = 0.0
        v[dist >= radius] = v_s[dist >= radius]

    return v


def cub_img(v, view_dir=2):
    if view_dir == 0:
        vt = np.transpose(v, [1, 2, 0])
    elif view_dir == 1:
        vt = np.transpose(v, [2, 0, 1])
    elif view_dir == 2:
        vt = v

    row_num = vt.shape[0] + 1
    col_num = vt.shape[1] + 1
    slide_num = vt.shape[2]
    disp_len = int(np.ceil(np.sqrt(slide_num)))

    slide_count = 0
    im = np.zeros((row_num * disp_len, col_num * disp_len)) + float('nan')
    for i in range(disp_len):
        for j in range(disp_len):
            im[(i * row_num): ((i + 1) * row_num - 1), (j * col_num): ((j + 1) * col_num - 1)] = vt[:, :, slide_count]
            slide_count += 1

            if slide_count >= slide_num:
                break

        if slide_count >= slide_num:
            break

    im_v = im[np.isfinite(im)]

    if im_v.max() > im_v.min():
        im = (im - im_v.min()) / (im_v.max() - im_v.min())

    return {'im': im, 'vt': vt}
