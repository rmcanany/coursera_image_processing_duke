from __future__ import division, print_function

import numpy as np
from matplotlib import pyplot as plt
import time
import sys

from skimage import img_as_float
from skimage import io
from skimage.color import rgb2gray, gray2rgb
from skimage.draw import circle, circle_perimeter, line
from skimage.transform import rescale

from scipy import ndimage

# from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import extract_patches

def read_image(filename, float, scale=1):
    I = io.imread(filename)
    if not scale == 1:
        I = rescale(I, (scale, scale))
    if float:
        I = img_as_float(I)
    return(I)

def save_image(filename, I):
    io.imsave(filename, I)
    
def get_timestamp():
    return time.strftime("%Y%m%d_%H%M%S")

# Plotting functions

def plot_1x1(imgs_list, titles_list, save_file=None, fs=7):
    cols = len(imgs_list)
    
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(fs,fs))
    
    i = 0
    
    # print(imgs_list[i], titles_list[i])

    axes.imshow(imgs_list[i], cmap="gray", interpolation="nearest")
    axes.set_title(titles_list[i], size=20)
    axes.set_xticks([])
    axes.set_yticks([])
    plt.tight_layout();

    if not (save_file == None):
        filename = time.strftime("%Y%m%d_%H%M") + "_" + save_file + ".png"
        fig.savefig(filename, bbox_inches='tight')
    

def plot_1xc(imgs_list, titles_list, save_file=None, fs=7):
    cols = len(imgs_list)

    if cols == 1: 
        plot_1x1(imgs_list, titles_list, save_file=None, fs=fs)
    else:
        i = 0

        fig, axes = plt.subplots(nrows=1, ncols=cols, figsize=(fs,fs))
        for c in range(cols):
            axes[c].imshow(imgs_list[i], cmap="gray", interpolation="nearest")
            axes[c].set_title(titles_list[i], size=20)
            axes[c].set_xticks([])
            axes[c].set_yticks([])
            i = i + 1
        plt.tight_layout();

        if not (save_file == None):
            filename = time.strftime("%Y%m%d_%H%M") + "_" + save_file + ".png"
            fig.savefig(filename, bbox_inches='tight')
        
def plot_rxc(imgs_list, titles_list, ncol=3, save_file=None, fs=7):
    cols = ncol
    rows = np.ceil(len(imgs_list)/cols).astype(int)
    if rows==1:
        plot_1xc(imgs_list, titles_list, save_file, fs=fs)
    else:
        i = 0
    
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(fs,fs*rows/cols))
        for r in range(rows):
            for c in range(cols):
                if i < len(imgs_list):
                    axes[r,c].imshow(imgs_list[i], cmap="gray", interpolation="nearest")
                    axes[r,c].set_title(titles_list[i], size=20)
                    axes[r,c].set_xticks([])
                    axes[r,c].set_yticks([])
                i = i + 1
        plt.tight_layout();
    
        if not (save_file == None):
            filename = time.strftime("%Y%m%d_%H%M") + "_" + save_file + ".png"
            fig.savefig(filename, bbox_inches='tight')
            
def plot_quivers(vecs_list, titles_list, scales_list, save_file=None, fs=7):
    
    cols = len(vecs_list)

    fig, axes = plt.subplots(nrows=1, ncols=cols, figsize=(fs,fs))
    
    for c in range(cols):
        x = np.linspace(0, vecs_list[c][0].shape[1], vecs_list[c][0].shape[1]).astype(np.int)
        y = np.linspace(0, vecs_list[c][0].shape[0], vecs_list[c][0].shape[0]).astype(np.int)
        x, y = np.meshgrid(x[:-1], y[:-1])

        vy = vecs_list[c][0][y,x] 
        vx = vecs_list[c][1][y,x]
        
        axes[c].quiver(x, y, vx, -vy, pivot='tip', scale=scales_list[c], headwidth=5, headlength=10)
        axes[c].set(aspect=1)
        axes[c].invert_yaxis()
        
        axes[c].set_title(titles_list[c], size=20)
        axes[c].set_xticks([])
        axes[c].set_yticks([])
    plt.tight_layout();
    
    if not (save_file == None):
        filename = time.strftime("%Y%m%d_%H%M") + "_" + save_file + ".png"
        fig.savefig(filename, bbox_inches='tight')

def paint_line(img, rr, cc, color, channel, max_val):
    im = np.copy(img)
    im[rr,cc,:] = 0
    if not color == 'k':
        if color == 'w':
            im[rr,cc,:] = max_val
        else:
            im[rr,cc,channel] = max_val
    return im

def add_rectangle(img, y0, x0, y1, x1, color="r", width=1):
    """Colors: 'r', 'g', 'b', 'w', 'k'"""
    im = np.copy(img)
    if im.ndim == 2:
        im = gray2rgb(im)
    max_val = 1
    if np.max(img) > 1:
        max_val = 255
    
    channel = 3 # Bogus value when color = 'w' or 'k'
    if color=='r': 
        channel = 0
    if color=='g': 
        channel = 1
    if color=='b': 
        channel = 2
    
    for i in range(width):
        yy0 = y0+i; xx0 = x0+i; yy1 = y1-i; xx1 = x1-i
        rr, cc = line(yy0, xx0, yy1, xx0) # left
        im = paint_line(im, rr, cc, color, channel, max_val)
        rr, cc = line(yy1, xx0, yy1, xx1) # bottom
        im = paint_line(im, rr, cc, color, channel, max_val)
        rr, cc = line(yy1, xx1, yy0, xx1) # right
        im = paint_line(im, rr, cc, color, channel, max_val)
        rr, cc = line(yy0, xx1, yy0, xx0) # top
        im = paint_line(im, rr, cc, color, channel, max_val)

    return im

def create_mask(img, num_circles, lo_thickness, hi_thickness, patch_size):
    im = rgb2gray(img)
    m = np.ones_like(im)

    np.random.seed(31415926)
    for i in range(num_circles):
        im_tmp = np.ones_like(m)
        yy = np.random.randint(0, m.shape[0])
        xx = np.random.randint(0, m.shape[1])
        r = np.random.randint(20, m.shape[0]/2)
        t = np.random.randint(lo_thickness, hi_thickness)
        rro, cco = circle(yy, xx, r, shape=m.shape)
        rri, cci = circle(yy, xx, r-t, shape=m.shape)
        im_tmp[rro,cco] = 0
        im_tmp[rri,cci] = 1
        m[im_tmp==0] = 0

    # Fix mask border.
    d = patch_size + 1
    m[:d,:] = 1
    m[-d:, :] = 1
    m[:, :d] = 1
    m[:, -d:] = 1
    
    return m

def initialize_confidence(mask):
    return mask

def pad_shape(shape, n_pixels):
    b = -(np.copy(shape)-1)
    out = np.copy(b)

    for i in range(n_pixels):
        # shift the mask up, down, left, right and add to b.
        out[:-1 , :  ] = out[:-1 , :  ] + b[1:  , :  ] # up
        out[1:  , :  ] = out[1:  , :  ] + b[:-1 , :  ] # down
        out[:   , :-1] = out[:   , :-1] + b[:   , 1: ] # left
        out[:   , 1: ] = out[:   , 1: ] + b[:   , :-1] # right
        out[:-1 , :-1] = out[:-1 , :-1] + b[1:  , 1: ] # up left
        out[:-1 , 1: ] = out[:-1 , 1: ] + b[1:  , :-1] # up right
        out[1:  , :-1] = out[1:  , :-1] + b[:-1 , 1: ] # down left
        out[1:  , 1: ] = out[1:  , 1: ] + b[:-1 , :-1] # down right
        b = np.copy(out)
    
    # threshold b and subtract out the shape
    b[shape==0] = 0
    b = np.clip(b, 0, 1)

    return b
    
def get_boundary(mask):
    b = pad_shape(mask, 1)
    return b

def get_grad(img, mask, boundary):
    im = rgb2gray(img)

    # grad = np.gradient(im)
    grad = [np.zeros_like(im), np.zeros_like(im)]
    
    grad[0][mask==0] = 0
    grad[1][mask==0] = 0
    
    for y, x in zip(*np.where(boundary==1)):
        # x
        valid_left = (not x == 0) and (mask[y,x-1] == 1)
        valid_right = (not x == img.shape[1]-1) and (mask[y,x+1] == 1)
        
        if valid_left and valid_right:
            grad[1][y,x] = (im[y,x+1] - im[y,x-1])/2.
        if valid_left and not valid_right:
            grad[1][y,x] = im[y,x] - im[y,x-1]
        if not valid_left and valid_right:
            grad[1][y,x] = im[y,x+1] - im[y,x]
        if not valid_left and not valid_right:
            grad[1][y,x] = 0

        # y
        valid_up = (not y == 0) and (mask[y-1,x] == 1)
        valid_down = (not y == img.shape[0]-1) and (mask[y+1,x] == 1)
        
        if valid_up and valid_down:
            grad[0][y,x] = (im[y+1,x] - im[y-1,x])/2.
        if valid_up and not valid_down:
            grad[0][y,x] = im[y,x] - im[y-1,x]
        if not valid_up and valid_down:
            grad[0][y,x] = im[y+1,x] - im[y,x]
        if not valid_up and not valid_down:
            grad[0][y,x] = 0
    
    return grad

def get_isophote(grad):
    return -1*grad[1], grad[0]

def get_mask_normal(mask, boundary):
    m = -(mask-1) # Flip black and white for correct vector direction.
    normal = np.gradient(m)

    # set to 0 outside src boundary
    normal[0][np.where(boundary==0)] = 0
    normal[1][np.where(boundary==0)] = 0

    # normalize with abs value
    normal_abs = np.sqrt(normal[0]**2 + normal[1]**2)
    normal_abs[normal_abs==0] = 1
    normal = normal/normal_abs
    return normal

def update_inner(img, mask, boundary_old, boundary_new, inner_old):
    b = boundary_new - boundary_old
    b[b<0] = 0 # Delete boundary_old remainder
    
    grad_new = get_grad(img, mask, b)
    
    isophote = get_isophote(grad_new)

    normal = get_mask_normal(mask, b)

    inner_new = isophote[0]*normal[0] + isophote[1]*normal[1]
    
    inner_new[boundary_new == boundary_old] = inner_old[boundary_new == boundary_old]

    inner_new = np.abs(inner_new)
    
    if np.sum(inner_new) < 1e-10:
        # print("np.sum(inner)", np.sum(inner))
        inner_new[boundary_new==1] = .001
        
    return inner_new


def get_inner(img, mask, boundary):
    grad = get_grad(img, mask, boundary)
    isophote = get_isophote(grad)
    normal = get_mask_normal(mask, boundary)
    
    inner = isophote[0]*normal[0] + isophote[1]*normal[1]
    
    inner = np.abs(inner)
    
    if np.sum(inner) < 1e-10:
        # print("np.sum(inner)", np.sum(inner))
        inner[boundary==1] = .001
        
    return inner

def get_patch_to_inpaint_indices(confidence, inner, patch_size):
    
    from scipy import ndimage
    
    ymax = confidence.shape[0]
    xmax = confidence.shape[1]

    win = np.ones((patch_size, patch_size))/patch_size**2
    average_confidence = ndimage.convolve(confidence, win, mode='reflect', cval=0.0)
    
    Pp = average_confidence * inner
    patch_center = np.where(Pp == np.max(Pp)) 
    # All max values are returned in case of ties.  Take the first one.
    patch_center_y = patch_center[0][0]
    patch_center_x = patch_center[1][0]
    
    offset = int((patch_size - 1)/2)
    
    y0 = patch_center_y - offset
    x0 = patch_center_x - offset
    y1 = patch_center_y + offset
    x1 = patch_center_x + offset
    
    # if y0 < 0 or x0 < 0 or y1 > ymax or x1 > xmax:
        # print()
        # print("Patch out of bounds.  Shape, (y0, x0, y1, x1) =", confidence.shape, y0, x0, y1, x1)
    
    return y0, x0, y1, x1

def get_coords(matrix, y0, x0, y1, x1, patch_size):
    N = matrix.shape[0] - 1
    M = matrix.shape[1] - 1
      
    if y0 >= 0:
        y0_patch = 0
        y0_im = y0
    if y0 < 0:
        y0_patch = -y0
        y0_im = 0
    if y1 <= N:
        y1_patch = patch_size - 1
        y1_im = y1
    if y1 > N:
        y1_patch = (patch_size - 1) - (y1 - N)
        y1_im = N

    if x0 >= 0:
        x0_patch = 0
        x0_im = x0
    if x0 < 0:
        x0_patch = -x0
        x0_im = 0
    if x1 <= M:
        x1_patch = patch_size - 1
        x1_im = x1
    if x1 > M:
        x1_patch = (patch_size - 1) - (x1 - M)
        x1_im = M
        
#     print(y0_patch, y1_patch, x0_patch, x1_patch, y0_im, y1_im, x0_im, x1_im)
    return y0_patch, y1_patch, x0_patch, x1_patch, y0_im, y1_im, x0_im, x1_im
    
def get_patch_contents(matrix, y0, x0, y1, x1, patch_size, default_fill= 0):
    
    y0_patch, y1_patch, x0_patch, x1_patch, y0_im, y1_im, x0_im, x1_im = get_coords(matrix, y0, x0, y1, x1, patch_size)

    if matrix.ndim == 2:
        patch = default_fill + np.zeros((patch_size, patch_size))
        patch[y0_patch:y1_patch+1, x0_patch:x1_patch+1] = matrix[y0_im:y1_im+1, x0_im:x1_im+1]

    if matrix.ndim == 3:
        patch = default_fill + np.zeros((patch_size, patch_size, matrix.shape[2]))

        patch[y0_patch:y1_patch+1, x0_patch:x1_patch+1, :] = matrix[y0_im:y1_im+1, x0_im:x1_im+1, :]

    return patch

def put_patch_contents(matrix, patch_contents, y0, x0, y1, x1, patch_size):
    
    m = np.copy(matrix)

    y0_patch, y1_patch, x0_patch, x1_patch, y0_im, y1_im, x0_im, x1_im = get_coords(matrix, y0, x0, y1, x1, patch_size)
    
    if m.ndim == 2:
        m[y0_im:y1_im+1, x0_im:x1_im+1] = patch_contents[y0_patch:y1_patch+1, x0_patch:x1_patch+1]
    
    if m.ndim == 3:
        m[y0_im:y1_im+1, x0_im:x1_im+1, :] = patch_contents[y0_patch:y1_patch+1, x0_patch:x1_patch+1, :]
    
    return m 

def get_patch_contents_centered(matrix, y0, x0, y1, x1, patch_size, default_fill):
    offset = int(patch_size/2.0)
    y0c = y0 - offset
    x0c = x0 - offset
    y1c = y0 + offset
    x1c = x0 + offset
    
    return get_patch_contents(matrix, y0c, x0c, y1c, x1c, patch_size, default_fill)

def get_patch_matrix(source, invalid_pixels, patch_size, max_patches=None):
    MAX_MEM = 500e6 # Maximum memory allowed.  These things can get big fast.
    
    I_s = np.copy(source)
    I_ip = np.copy(invalid_pixels)
    
    h, w, d = I_s.shape
        
    # Downsize if the patch matrix will be too big.
    factor = 1
    mem = 8 * h * w * d * patch_size * patch_size
    if mem > MAX_MEM:
        factor = int(np.sqrt(mem/MAX_MEM)) + 1
        mem = 8 * h/factor * w/factor * d * patch_size * patch_size
    if mem > MAX_MEM:
        print("Patch matrix too large.  Size:", mem)
        sys.exit()
        
    # Mark invalid pixels.
    I_s[I_ip < 1, :] = 1e5
    
    ## Generate the patch_matrix.
    # Returns a matrix of dims [h-ps, w-ps, 1, ps, ps, nc]
    # Where 'ps' is patch_size and 'nc' is number of colors (3 assumed.)
    patch_matrix = extract_patches(I_s, 
                                   patch_shape=(patch_size, patch_size, 3), 
                                   extraction_step=(factor,factor,1))
    pmh, pmw = patch_matrix.shape[:2]
    patch_matrix = patch_matrix.reshape((pmh, pmw, patch_size, patch_size, 3))

    # Get the differences between adjacent patches in 2D.
    adh = patch_size * patch_size * 3 * np.ones((pmh, pmw))
    adw = np.copy(adh)

    adh[1:, :] = np.sum((patch_matrix[:-1] - patch_matrix[1:])**2, axis=(2,3,4))
    adw[:, 1:] = np.sum((patch_matrix[:, :-1] - patch_matrix[:, 1:])**2, axis=(2,3,4))

    ad = np.copy(adh)

    replace = adw < adh

    ad[replace] = adw[replace]

    # Reshape to new dims [(h-ps) * (w-ps), ps, ps, nc]
    patch_matrix = patch_matrix.reshape(-1, patch_size, patch_size, 3)
    ad = ad.reshape(pmh * pmw)

    # Remove invalid patches.  Assume image values in range 0.0 to 1.0 for now.
    patch_sum = np.sum(patch_matrix, axis=(1,2,3))
    valid = patch_sum <= patch_size * patch_size * I_s.shape[2]

    patch_matrix = patch_matrix[valid]
    ad = ad[valid]
    
    # If a maximum number of patches is specified, reduce the size if necessary.
    # For speed, only checking for differences of adjacent patches for now.
    if max_patches is not None:
        if patch_matrix.shape[0] > max_patches:
            # adjacent patch difference, 'apd'.  Keep patches with large differences.
            # apd = np.sum((patch_matrix[:-1] - patch_matrix[1:])**2, axis=(1,2,3))
            
            cutoff = np.sort(ad)[::-1][max_patches]

            # valid = np.zeros((1 + ad.shape[0]), dtype=np.bool)
            # valid[0] = True
            # valid[1:] = ad >= cutoff
            valid = ad >= cutoff

            patch_matrix = patch_matrix[valid]
            patch_matrix = patch_matrix[:max_patches]
    
    return patch_matrix

def get_image_patch_matrix(img_color, confidence, patch_size):
    
    print("Function 'get_image_patch_matrix' no longer used.")
    sys.exit()

    im = np.copy(img_color)
    
    im[confidence < 1, :] = 1e5 # To make sure min ssd does not use infilled patches.
    patch_matrix = extract_patches_2d(im, (patch_size, patch_size))
        
    return patch_matrix


def get_best_fill_patch(img_color, mask, confidence, y0, x0, y1, x1, patch_size, block_size=None, patch_matrix=None):
    
    if block_size is not None:
        img_block = get_patch_contents_centered(img_color, y0, x0, y1, x1, block_size, default_fill=1e5)
        confidence_block = get_patch_contents_centered(confidence, y0, x0, y1, x1, block_size, default_fill=0)

        # patch_matrix = get_image_patch_matrix(img_block, confidence_block, patch_size)
        patch_matrix = get_patch_matrix(img_block, confidence_block, patch_size)

    pm = np.copy(patch_matrix)
    
    patch_mask = get_patch_contents(mask, y0, x0, y1, x1, patch_size)

    patch_image = get_patch_contents(img_color, y0, x0, y1, x1, patch_size)
    
    patch_image[patch_mask==0,:] = 0
    
    pm[:, patch_mask==0, :] = 0
    pm[patch_matrix > 1e4] = 1e5

    ssd = np.sum((pm - patch_image)**2, axis=(1,2,3))
    
    b = np.where(ssd == np.min(ssd))
    fill_patch = patch_matrix[b[0], :, :, :]
    fill_patch = fill_patch[0]
    
    # print(b[0])
    
    return fill_patch

def update_image(img, mask, fill_patch, y0t, x0t, y1t, x1t, patch_size):
  
    source_patch = np.copy(fill_patch)

    dest_patch = get_patch_contents(img, y0t, x0t, y1t, x1t, patch_size)

    mask_patch = get_patch_contents(mask, y0t, x0t, y1t, x1t, patch_size)
    
    dest_patch[mask_patch == 0, :] = source_patch[mask_patch == 0, :]
    
    im = put_patch_contents(img, dest_patch, y0t, x0t, y1t, x1t, patch_size)
    
    return im


def update_parameters(confidence, mask, boundary, y0t, x0t, y1t, x1t, img, patch_size, inner):
    # These operations must be done in Confidence, Mask, Boundary, Inner Product order.
    
    # Confidence
    c = np.copy(confidence)
    
    c_patch = get_patch_contents(c, y0t, x0t, y1t, x1t, patch_size)
    Cp = np.average(c_patch)

    m_patch = get_patch_contents(mask, y0t, x0t, y1t, x1t, patch_size)
    c_patch[m_patch == 0] = Cp
    c = put_patch_contents(c, c_patch, y0t, x0t, y1t, x1t, patch_size)

    # Mask
    m_patch_new = np.ones((patch_size, patch_size))
    m = put_patch_contents(mask, m_patch_new, y0t, x0t, y1t, x1t, patch_size)
    
    # Boundary
    b = get_boundary(m)

    # Inner Product
    inner_new = update_inner(img, m, boundary, b, inner)

    return c, m, b, inner_new


def progress_tracker(start_val, end_val, last_val, current_val, report_every):
    
    upcount = True
    if end_val < start_val:
        upcount = False
        
    milestones = np.arange(0, 1, report_every)
    
    if upcount:
        last_pct = (last_val - start_val)/(end_val - start_val)
        current_pct = (current_val - start_val)/(end_val - start_val)
    else:
        last_pct = (start_val - last_val)/(start_val - end_val)
        current_pct = (start_val - current_val)/(start_val - end_val)
    
    last_bin = np.sum(last_pct > milestones)
    current_bin = np.sum(current_pct > milestones)
    
    if current_bin > last_bin:
        return True, 100 * round(milestones[current_bin - 1], 2)
    else:
        return False, 0

def infill(img, mask, patch_size, 
           block_size=None, patch_source=None, 
           patch_mask=None, max_patches=None, 
           iters=None, verbose=1, update_every=0.2):
    M = np.copy(mask)
    I_c = np.copy(img)
    I_c[M==0,:] = 1 # Corrupted image.
    I_r = np.copy(I_c) # Repaired image.
    
    B = get_boundary(M)
    C = initialize_confidence(M)
    inner = get_inner(I_c, M, B)
    
    if (block_size is None) and (patch_source is None):
        print("(block_size is None) and (patch_source is None)")
        sys.exit()
    if block_size is None:
        PM = get_patch_matrix(patch_source, patch_mask, 
                              patch_size, max_patches)
    if patch_source is None:
        PM = None
    
    c = np.sum(M==0) # Count of remaining pixels to fill.
    c_start = c
    c_old = c
    
    if iters==None:
        end_val = 0
    else:
        end_val = c - iters
    
    while c > end_val:
        
        if verbose > 0:
            flag, pct = progress_tracker(c_start, end_val, 
                                         c_old, c, update_every)
            if flag:
                print(time.strftime("%Y-%m-%d %H:%M:%S"), pct, 
                                    "% complete.  c=", str(c))
        
        # 't' means 'to', i.e., the destination patch.
        y0t, x0t, y1t, x1t = get_patch_to_inpaint_indices(C, inner, patch_size)
        
        # 'f' means 'from', i.e., the source patch.
        fill_patch = get_best_fill_patch(I_r, M, C, 
                                         y0t, x0t, y1t, x1t, 
                                         patch_size, block_size, PM)
    
        I_r = update_image(I_r, M, fill_patch, 
                           y0t, x0t, y1t, x1t, 
                           patch_size)
    
        C, M, B, inner = update_parameters(C, M, B, 
                                           y0t, x0t, y1t, x1t, 
                                           I_r, patch_size, inner)
        
        c_old = c
        c = np.sum(M==0)
        if c == c_old:
            print()
            print("No pixels were filled.  Quitting.",  
                  "(c_old, c)", (c_old, c))
            break
        
    return I_c, I_r

def infill_file(img_file, save_file, mask_coords, patch_size, source_coords, max_patches=None):

    I = read_image(img_file, float=True)

    y0s = source_coords[0]; y1s = source_coords[1]; x0s = source_coords[2]; x1s = source_coords[3]; 
    I_s = np.copy(I[y0s:y1s, x0s:x1s, :])
    I_sm = np.ones_like(I_s[:, :, 0])

    y0m = mask_coords[0]; y1m = mask_coords[1]; x0m = mask_coords[2]; x1m = mask_coords[3]; 
    d = 2 * patch_size
    I_o = np.copy(I[y0m-d:y1m+d, x0m-d:x1m+d, :])

    M = np.ones_like(I_o[:, :, 0])
    M[d:-d, d:-d] = 0

    I_c, I_r = infill(I_o, M, patch_size, 
                         patch_source = I_s, 
                         patch_mask = I_sm, 
                         max_patches = max_patches)

    I[y0m:y1m, x0m:x1m] = I_r[d:-d, d:-d]

    filename = save_file
    filename = filename + get_timestamp() + ".jpg"

    save_image(filename, I)
    
    return filename

def repaint(img_dest, img_source, img_source_mask, patch_size, 
            max_patches, iters=None, verbose=0, update_every=0.1):
    
    I_d = np.copy(img_dest)
    
    offset = int(patch_size/2.0)
    
    # Define starting points for repainting.
    h = I_d.shape[0]
    w = I_d.shape[1]
    npts = h
    hi = np.random.randint(0,h,npts)
    wi = np.random.randint(0,w,npts)

    complete = np.zeros((h, w))
    complete[hi, wi] = 1
    
    boundary = pad_shape(complete, 1)
    
    # Get patches.
    patch_matrix = get_patch_matrix(img_source, img_source_mask, patch_size, max_patches)
    if verbose > 1:
        print("patch_matrix.shape", patch_matrix.shape)
    
    c = np.sum(complete == 0)
    c_start = c
    c_old = c
    
    if iters==None:
        end_val = 0
    else:
        end_val = c - iters
    
    while c > end_val:
        if verbose > 0:
            flag, pct = progress_tracker(c_start, end_val, 
                                         c_old, c, update_every)
            if flag:
                print(time.strftime("%Y-%m-%d %H:%M:%S"), pct, "% complete.  c=", str(c))
       
        valid = np.where(boundary==1)
        n = len(valid[0])
        r = np.random.randint(n)
        y = valid[0][r]
        x = valid[1][r]
        
        y0 = y - offset
        x0 = x - offset
        y1 = y0 + patch_size - 1
        x1 = x0 + patch_size - 1
        
        patch_to_fill = get_patch_contents(I_d, y0, x0, y1, x1, patch_size)
        
        ssd = np.sum((patch_matrix - patch_to_fill)**2, axis=(1,2,3))
        valid = np.where(ssd == np.min(ssd))
        
        fill_patch = patch_matrix[valid[0][0]]
        
        I_d = update_image(I_d, complete, fill_patch, y0, x0, y1, x1, patch_size)
        
        complete_patch = np.ones((patch_size, patch_size))
        complete = put_patch_contents(complete, complete_patch, y0, x0, y1, x1, patch_size)
        
        boundary = pad_shape(complete, 1)
        
        c_old = c
        c = np.sum(complete == 0)
    
    return I_d

def block_repaint(img_dest, img_source, scale, patch_size, max_patches, block_size, row_num):

    scale = float(scale)
    patch_size = int(float(patch_size))
    max_patches = int(float(max_patches))
    block_size = int(float(block_size))
    row_num = int(float(row_num))
    
    print("Input file:", img_dest)
    print("Patch source file:", img_source)
    print("Scale:", scale)
    print("Patch size:", patch_size)
    print("Max patches:", max_patches)
    print("Block size:", block_size)
    print("Row number:", row_num)

    I_d = read_image(img_dest, float=True, scale=scale)
    
    I_s = read_image(img_source, float=True, scale=scale)
    
    I_sm = np.ones((I_s.shape[0], I_s.shape[1]))
    I_sm[rgb2gray(I_s) == 1] = 0 # implied mask

    I_r = np.zeros_like(I_d)
    
    outfile = ""
    outfile = outfile + "p" + str(patch_size) + "_mp" + str(max_patches)
    outfile = outfile + "_r" + str(row_num)
    outfile = outfile + "_" + get_timestamp() + "_repaint.jpg"

    for r in np.arange(0, I_d.shape[0], block_size):
        for c in np.arange(0, I_d.shape[1], block_size):
            if (row_num == -1) or (r == (row_num - 1) * block_size):
                re = r + block_size - 1
                ce = c + block_size - 1

                pad_block = get_patch_contents(I_d, r-patch_size, c-patch_size, 
                                               re+patch_size, ce+patch_size, 
                                               block_size + 2*patch_size)

                block = repaint(pad_block, I_s, I_sm, patch_size, max_patches, verbose=1)

                block = block[patch_size:-patch_size, patch_size:-patch_size, :]
                # print(r, re, c, ce, block.shape)
                I_r = put_patch_contents(I_r, block, r, c, re, ce, block_size)

                save_image(outfile, I_r)


