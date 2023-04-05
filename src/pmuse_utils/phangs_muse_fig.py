# -*- coding: utf-8 -*-

# -----------------------------------------------
# Script to run figures from the DAP DR MUSE PHANGS maps
#
# Eric Emsellem - Sep 2020
#
# versions:
#           v1.0.1 - Some tuning - Feb 2021
#           v1.0.0 - Full rearrangement - 21/12/2020
#           v0.0.1 - First go at it - 18/09/2020
# -----------------------------------------------

# ================== INITIALISATION OF MUSE quantities =====

# ------------- General import ---------------------------
# Importing general modules
import os
import copy

# astropy / numpy / mpl
from astropy.io import fits as pyfits
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_rgb import make_rgb_axes, RGBAxes
from matplotlib import cm
import multicolorfits as mcf

from .phangs_muse_init import Phangs_Muse_Info
from .phangs_muse_config import (default_rgb_bands, default_colormap, maps_rules_dict,
                                 maps_keyw_dict, default_jwst_bands)

# ------- Useful Functions ---------------
def get_name_err_map(name):
    return f"{name}_ERR"

def mask_0_and_invalid(data, badvalue=0):
    """Small useful function to filter out nan and 0
    """
    return np.ma.masked_array(data, np.isnan(data) | (data == badvalue))

def set_bad_value(data, bad=np.nan, badval=np.nan):
    """Small useful function to filter out nan and 0
    """
    data[data == bad] = badval
    return data

def get_limits(data, pixsize=0.2, center_pix=None):
    lastp = (data.shape[1] - 1, data.shape[0] - 1)
    if center_pix is None:
        center_pix = np.array([lastp[0] / 2., lastp[1] / 2.])
    # Returning a default limit assuming spaxels of pixsize
    xlim = (np.array([0, lastp[0]]) - center_pix[0]) * pixsize
    ylim = (np.array([0, lastp[1]]) - center_pix[1]) * pixsize
    limits = [xlim[0], xlim[1], ylim[0], ylim[1]]
    return limits

def get_plot_norm(data, vmin=None, vmax=None, zscale=False, scale='linear', 
                  percentiles=[5,95], centred_cuts=False, verbose=True):
    """Building the normalisation for a map. Still need some tweaking for
    e.g. noise clipping. This return a 'norm' that can be used for plotting.

    Note: this is edited from an mpdaf routine
    """
    from astropy import visualization as viz
    from astropy.visualization.mpl_normalize import ImageNormalize

    # Mask 0's and Nan
    mdata = mask_0_and_invalid(data)

    # Choose vmin and vmax automatically?
    if zscale:
        interval = viz.ZScaleInterval()
        if mdata.dtype == np.float64:
            try:
                vmin, vmax = interval.get_limits(mdata[~mdata.mask])
            except Exception:
                # catch failure on all NaN
                if np.all(np.isnan(mdata.filled(np.nan))):
                    vmin, vmax = (np.nan, np.nan)
                else:
                    raise
        else:
            vmin, vmax = interval.get_limits(mdata.filled(0))
    elif percentiles is not None:
        vmin, vmax = np.percentile(mdata[~mdata.mask], percentiles) 

    if centred_cuts:
        abscut = np.maximum(np.abs(vmin), np.abs(vmax))
        vmin, vmax = -abscut, abscut

    # How are values between vmin and vmax mapped to corresponding
    # positions along the colorbar?
    if scale == 'linear':
        stretch = viz.LinearStretch
    elif scale == 'log':
        stretch = viz.LogStretch
    elif scale in ('asinh', 'arcsinh'):
        stretch = viz.AsinhStretch
    elif scale == 'sqrt':
        stretch = viz.SqrtStretch
    else:
        raise ValueError('Unknown scale: {}'.format(scale))

    # Create an object that will be used to map pixel values
    # in the range vmin..vmax to normalized colormap indexes.
    if verbose:
        print(f"Vmin = {vmin} / Vmax = {vmax}")
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=stretch(), clip=False)

    return norm

def set_axtext(im, ax, label, cuts=True, pos=[0.5, 0.02], fontsize=14, zorder=10):
    """Setting the text within an imshow image
    Provides the label and cuts within the panel

    This requires the axis.
    """
    l1,l2 = im.get_clim()
    facdpi = get_facdpi()
    if cuts:
        return ax.text(0.02,0.96, "{0:4.2f}/{1:4.2f}".format(l1, l2), verticalalignment='top',
                rotation='vertical',transform=ax.transAxes, fontsize=10*facdpi, zorder=zorder,
                bbox=dict(boxstyle="square", ec=(0,0,0), fc=(1,1,1), zorder=zorder))

    return ax.text(pos[0], pos[1], label, fontsize=fontsize*facdpi, transform=ax.transAxes, 
            horizontalalignment='center', zorder=zorder,
            bbox=dict(boxstyle="square", ec=(0,0,0), fc=(1,1,1), zorder=zorder))

def set_figtext(fig, label, pos=[0.5, 0.02], ax=None, fontsize=14):
    """Setting the text within an imshow image
    Provides the label and cuts within the panel
    """
    if ax is not None:
        pos = ax.transAxes.transform((pos[0], pos[1]))
        pos = fig.transFigure.inverted().transform((pos[0], pos[1]))
    plt.figtext(pos[0], pos[1], label, fontsize=fontsize, transform=fig.transFigure, 
            horizontalalignment='center', zorder=999.,
            bbox=dict(boxstyle="square", ec=(0,0,0), fc=(1,1,1), zorder=999.))

def make_muse_sample_grid(phangs, aspect="asec", **kwargs):
    """Make a sample grid. aspect can be 'asec', 'kpc' or 'equal'.
    If 'equal' it will use ncols and nows. Otherwise it will guess it from the
    frames_per_row, which gives a list of numbers, each number being how many
    frames per row. E.g., [2, 3, 2] means 2 frames on 1st row, 3 on 2nd row, 2
    again on third and last row.
    """

    return make_map_grid(shapelist=phangs.get_muse_fov(unit=aspect), **kwargs)

def make_map_grid(shapelist=None, ncols=5, nrows=4,
                  figsize=15., frames_per_row=None, 
                  left=0.2, right=0.2, bottom=0.2, top=0.2,
                  hspace=0.2, wspace=0.2, alignX='centre',
                  alignY='centre', aspect=None, dpi=100):
    """Make a sample grid. aspect can be 'asec', 'kpc' or 'equal'.
    If 'equal' it will use ncols and nows. Otherwise it will guess it from the
    frames_per_row, which gives a list of numbers, each number being how many
    frames per row. E.g., [2, 3, 2] means 2 frames on 1st row, 3 on 2nd row, 2
    again on third and last row.
    """
#    class C: pass
#    list_keyw = ['figsize', 'top', 'bottom', 'left', 'right', 'hspace', 'wspace']
#    myc = C()
#    facdpi = 100. / dpi
#    for keyw in list_keyw:
#        setattr(myc, keyw, locals()[keyw] * facdpi)

    if aspect == "equal":
        gref = np.array([[1., 1.]] * ncols * nrows).T
        frames_per_row = np.array([ncols] * nrows)
    else:
        gref = shapelist

    if frames_per_row is None:
        frames_per_row = np.array([ncols] * nrows)
    else:
        frames_per_row = np.array(frames_per_row)
        nrows = len(frames_per_row)
        ncols = [None] * nrows

    # Decrypting the align parameter
    if isinstance(alignX, str):
        alignX = [alignX] * nrows
    if isinstance(alignY, str):
        alignY = [alignY] * nrows
    if np.size(alignY) != nrows: 
        print(f"ERROR: {np.size(alignY)} is different than ncols={nrows}")
        return None, None, None

    fulln = np.append([0], frames_per_row)
    cs = np.cumsum(fulln)
    new_ref = []
    for j in range(nrows):
        new_ref.append(gref[:,cs[j]:cs[j+1]])

    # Sizes in units of input data scale
    # Sizes per row in X
    listX = np.array([np.sum(new_ref[j][0]) for j in range(nrows)]) 
    # Max size in X
    sizeX = np.max(listX)
    # Max of sizes in Y
    list_maxY = np.array([np.max(new_ref[j][1]) for j in range(nrows)]) 
    # Sum of max sizes in Y
    sizeY = np.sum(list_maxY)

    marginsX = left + right + (frames_per_row - 1) * wspace
    mmarginsX = np.max(marginsX)
    mmarginsY = top + bottom + (nrows - 1) * hspace 
    fs = np.minimum((figsize - mmarginsX) / sizeX,
                    (figsize - mmarginsY) / sizeY)
    # Full size of figure
    # fs is the size of 1 pixel on the figure size (15 here)
    # fs is thus in Figure_size per pixel
    # Hence sizeX is the number of pixels in that direction (max)
    # Same for sizeY
    # sizefigX is thus the total extent of the plots in figure units
    # same for sizefigY (max extent)
    sizefigX = sizeX * fs + mmarginsX
    sizefigY = sizeY * fs + mmarginsY
    xmargin = wspace / sizefigX
    ymargin = hspace / sizefigY
    # Diff of size in Y
    dYrow = list_maxY * fs / sizefigY
    # Diff of size in X
    dXrow = (sizefigX - listX * fs - marginsX) / sizefigX
    # Scale is thus the per pixel size in for a unit of 1 figure size
    scale = [1. * fs / sizefigX, 1. * fs / sizefigY]
    
    # Opening the figure with the full size
    plt.close('all')
    fig = plt.figure(figsize=(sizefigX, sizefigY))

    plt.clf()
    axs = []
    ypos = 1. - top / sizefigY
    for j in range(nrows):
        xpos = left / sizefigX
        for i in range(frames_per_row[j]):
            dx, dy = new_ref[j][:,i]
            dx /= sizefigX / fs
            dy /= sizefigY / fs
            if alignY[j] == 'centre':
                ddy = (dYrow[j] - dy) / 2.
            elif alignY[j] == 'bottom':
                ddy = dYrow[j] - dy
            else:
                ddy = 0.
            if alignX[j] == 'centre':
                ddx = dXrow[j] / (2. * frames_per_row[j])
            elif alignX[j] == 'left':
                ddx = 0.
            else:
                ddx = dXrow[j] / frames_per_row[j]
            axs.append(fig.add_axes([xpos + ddx, ypos - dy - ddy, dx, dy]))
            xpos += dx + xmargin + 2 * ddx
        ypos -= dYrow[j] + ymargin

    return fig, axs, scale

def double_schechter(m, dm, a1=-0.35, a2=-1.47, phi1=3.96e-3, phi2=0.79e-3, Ms=10**10.66):
    """Return a double schechter law
    """
    return np.exp(- m / Ms) * (phi1 * (m / Ms)**a1 + phi2 * (m / Ms)**a2) * dm / Ms

def main_sequence(mstar):
    """Return a reference main sequence in log10(sfr) versus log10(mstar)
    """
    return (-0.32 * (np.log10(mstar) - 10.0) - 10.17) + np.log10(mstar)

def hist2d_sdss(logm, logsfr, logsfr_lim=[-1.0, 1.5], logm_lim=[9, 11], norm_mass=True, kernel=3):
    """Provides a histogram view of sdss
    Note the kernel convolution (default with sigma=3 pixels)
    """
    ind = np.where((logsfr > logsfr_lim[0]) & (logsfr < logsfr_lim[1]) 
                    & (logm > logm_lim[0]) & (logm < logm_lim[1]))
    h, xe, ye = np.histogram2d(logm[ind], logsfr[ind], bins=[120, 120], normed=True)
    if kernel is not None:
        from scipy.ndimage import gaussian_filter
        h = gaussian_filter(h, kernel)
    if norm_mass: 
        nm = h.shape[1]
        xe10 = 10**xe
        dm = np.diff(xe10)
        ma = (xe10[:-1] + xe10[1:]) / 2.
        for i in range(nm):
            h[i] /= double_schechter(ma[i], dm[i])
    return h, xe, ye

def annotate_rgb(text=['R', 'G', 'B'], cols=['red', 'green', 'blue'], 
                 pos=[0.5,0.1], align="y", delta=0.05, 
                 **kwargs):
    facdpi = get_facdpi()
    fontsize = kwargs.pop("fontsize", 15) * facdpi
    plt.figtext(pos[0], pos[1], text[1], c=cols[1], fontsize=fontsize, ha='center', **kwargs)
    if align == 'y':
        plt.figtext(pos[0], pos[1] + delta, text[2], c=cols[2], fontsize=fontsize, ha='center', **kwargs)
        plt.figtext(pos[0], pos[1] - delta, text[0], c=cols[0], fontsize=fontsize, ha='center', **kwargs)
    else:
        plt.figtext(pos[0] - delta, pos[1], text[2], c=cols[2], fontsize=fontsize, ha='center', **kwargs)
        plt.figtext(pos[0] + delta, pos[1], text[0], c=cols[0], fontsize=fontsize, ha='center', **kwargs)

def add_scalebar(ax, text, size, fontsize=12, weight='semibold',
                 pos=[0.18, 0.9], color='w', size_vertical=2, **kwargs):
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    import matplotlib.font_manager as fm
    fontprops = fm.FontProperties(size=fontsize, weight=weight)
    scalebar = AnchoredSizeBar(ax.transData, size, text, 'center', 
                            pad=0.1, borderpad=0.1, frameon=False, sep=1,
                            fontproperties=fontprops,
                            bbox_to_anchor=(pos[0], pos[1]),
                            bbox_transform=ax.transAxes,
                            size_vertical=size_vertical, color=color,
                            label_top=True, **kwargs)
    ax.add_artist(scalebar)

def get_facdpi(default_dpi=100):
    return 100. / plt.gcf().get_dpi()

def annotate_axis_xy(ax, scalex=None, scaley=None, text=None, units="asec", 
                     howmany=1, pos=[0.5, 0.01], ec='w', lw=3, fc="none", fs=12):

    facdpi = get_facdpi()
    limy = ax.get_ylim()
    fullY = limy[1] - limy[0]
    limx = ax.get_xlim()
    fullX = limx[1] - limx[0]
    if scalex is None:
        if scaley is None:
            print("Please provide either of scalex or scaley")
            return
        else:
            rotation = "vertical"
            scale_bar = howmany * scaley * fullY
    else:
        rotation = "horizontal"
        scale_bar = howmany * scalex * fullX

    posn = [pos[0] * fullX, pos[1] * fullY]
    if rotation == "horizontal":
        pos = [posn[0] - scale_bar/2., posn[1]]
        ax.add_artist(plt.Rectangle((pos[0], pos[1]), 0, scale_bar, ec=ec, lw=lw*facdpi, fc=fc))
    else:
        pos = [posn[0], posn[1] - scale_bar/2.]
        ax.add_artist(plt.Rectangle((pos[0], pos[1]), 0, scale_bar, ec=ec, lw=lw*facdpi, fc=fc))
    if text is None:
        ax.text(posn[0]+fullY/15., posn[1], f"{howmany} {units}", 
                rotation=rotation, fontsize=fs, ha='center', va='center', color=ec)
    else:
        ax.text(posn[0], posn[1]+fullX/15., text, rotation=rotation, fontsize=fs, 
                ha='center', va='center', color=ec)

def annotate_fig_x(fig, scalex, text=None, units="kpc", arrowstyle="<->",
                   howmany=5, pos=[0.5, 0.01], lw=3, fs=12, offt=0.015,
                   color="k", rotation=0, **kwargs):
    """Show a scale (with double arrows) for a given plot where we know the scale
    in fraction of the figure width

    Input
    -----
    fig (figure)
    scalex (float): scale in units of the figure size fraction in X
    howmany (int): number in units to show
    units (str): units to write on the scale (kpc, arsec)
    pos ([float, float]): position of the arrow in fraction of figure scale
    """
#    offy= fig.get_figheight() * pos[1]
#    facx= fig.get_figwidth() 
#    offx = pos[0] * facx
#    ar = plt.annotate("", xy=(offx - howmany * scalex * facx / 2., offy), 
#                      xytext=(offx + howmany * scalex * facx / 2., offy), 
#                      xycoords="figure fraction", color=color, rotation=rotation,
#                      arrowprops=dict(arrowstyle=arrowstyle, lw=lw, connectionstyle="arc3",
#                                      color=color, **kwargs))
    ar = plt.annotate("", xy=(pos[0] - howmany * scalex / 2., pos[1]), 
                      xytext=(pos[0] + howmany * scalex / 2., pos[1]), 
                      xycoords="figure fraction", color=color, rotation=rotation,
                      arrowprops=dict(arrowstyle=arrowstyle, lw=lw, connectionstyle="arc3",
                                      color=color, **kwargs))

    if text is None:
        plt.figtext(pos[0], pos[1]+offt, f"{howmany} {units}", fontsize=fs, c=color,
                    va='center', ha='center', rotation=rotation, **kwargs)
    else:
        plt.figtext(pos[0], pos[1]+offt, text, fontsize=fs, c=color,
                    va='center', ha='center', rotation=rotation, **kwargs)

def annotate_fig_y(fig, scaley, text=None, units="kpc", arrowstyle="<->",
                   howmany=5, pos=[0.5, 0.01], lw=3, fs=12, offt=0.01,
                   color="k", rotation=0, **kwargs):
    """Show a scale (with double arrows) for a given plot where we know the scale
    in fraction of the figure width

    Input
    -----
    fig (figure)
    scaley (float): scale in units of the figure size fraction in X
    howmany (int): number in units to show
    units (str): units to write on the scale (kpc, arsec)
    pos ([float, float]): position of the arrow in fraction of figure scale
    """
    ar = plt.annotate("", xy=(pos[0], pos[1] - howmany * scaley / 2.), 
                      xytext=(pos[0], pos[1] + howmany * scaley / 2.), 
                      xycoords="figure fraction", color=color,
                      arrowprops=dict(arrowstyle=arrowstyle, lw=lw, 
                                      connectionstyle="arc3", color=color), 
                      rotation=rotation)

    if text is None:
        plt.figtext(pos[0]+offt, pos[1], f"{howmany} {units}", 
                    fontsize=fs, va='center', ha='center', c=color, rotation=rotation)
    else:
        plt.figtext(pos[0]+offt, pos[1], text, fontsize=fs, 
                    ha='center', va='center', c=color, rotation=rotation)

def make_apy_rgb(r, g, b, factors=[1., 1., 1.], backwhite=True, minwhite=True,
                 **kwargs):
    """Use make_lupton_rgb from astropy to create an RGB scheme
    """
    from astropy.visualization import make_lupton_rgb
    r = r * factors[0]
    g = g * factors[1]
    b = b * factors[2]
    selnan = np.isnan(r) | np.isnan(g) | np.isnan(b)
    rgb = make_lupton_rgb(r, g, b, **kwargs)
    if backwhite:
        rgb[selnan, :] = 255
    if minwhite:
        rgb[rgb.sum(axis=2) == 0] = 255

    return r, g, b, rgb

def make_rgb(r, g, b, scale='linear', factors=[1., 1., 1.], 
             vmin=[None]*3, vmax=[None]*3, perc_rgb=None, clip_value=0.,
             sigma_conv=0.0, kerfac=8.0):
    """Make a cube from 3 images in R, G and B

    Returns the RGB and individual arrays
    Beware of the nan and 0 filtering out
    """
    # shape and first array in 3D
    ny, nx = r.shape

    if clip_value is not None:
        r[r <= clip_value] = np.nan
        g[g <= clip_value] = np.nan
        b[b <= clip_value] = np.nan

    # Returning the full arrays
    if scale is not None:
        if isinstance(scale, str) : scale = [scale]
        if len(scale) == 1: scale = scale * 3
        if perc_rgb is None: perc_rgb = [def_perc]
        if len(perc_rgb) == 1 : perc_rgb = perc_rgb * 3
        r = get_plot_norm(r, percentiles=perc_rgb[0], scale=scale[0], vmin=vmin[0], vmax=vmax[0])(r)
        g = get_plot_norm(g, percentiles=perc_rgb[1], scale=scale[1], vmin=vmin[1], vmax=vmax[1])(g)
        b = get_plot_norm(b, percentiles=perc_rgb[2], scale=scale[2], vmin=vmin[2], vmax=vmax[2])(b)

    r = mask_0_and_invalid(r) * factors[0]
    g = mask_0_and_invalid(g) * factors[1]
    b = mask_0_and_invalid(b) * factors[2]

    # Mask all layers with nan
    mask2d = r.mask | g.mask | b.mask
    mask3d = np.repeat(mask2d[:, :, np.newaxis], 3, axis=2)

    # Transfer data to the arrays
    R = np.zeros((ny, nx, 3))
    G = np.zeros_like(R)
    B = np.zeros_like(R)

    # Convolution if sigma_conv > 0
    if sigma_conv > 0:
        from astropy.convolution import (Gaussian2DKernel, convolve_fft)
        sizek = np.int(sigma_conv * kerfac)
        gaussian_k = Gaussian2DKernel(sigma_conv, x_size=sizek, y_size=sizek)
        r = convolve_fft(r, gaussian_k, psf_pad=True, fft_pad=True, boundary='fill', 
                          normalize_kernel=True, preserve_nan=True)
        g = convolve_fft(g, gaussian_k, psf_pad=True, fft_pad=True, boundary='fill', 
                          normalize_kernel=True, preserve_nan=True)
        b = convolve_fft(b, gaussian_k, psf_pad=True, fft_pad=True, boundary='fill', 
                          normalize_kernel=True, preserve_nan=True)

    R[:, :, 0] = r
    G[:, :, 1] = g
    B[:, :, 2] = b

    # Getting masked arrays again
    r = np.ma.masked_array(r, mask=mask2d)
    g = np.ma.masked_array(g, mask=mask2d)
    b = np.ma.masked_array(b, mask=mask2d)

    RGB = R + G + B
    RGB = np.dstack((RGB, ~mask2d))

    return r, g, b, RGB

def make_colorize_rgb(r, g, b, colors=['#ff1500', '#fbff00', '#03b3ff'], scale='linear', 
        perc_rgb=None, clip_value=0., inverse=False, vmin=[None]*3, vmax=[None]*3):
    """Make a cube from 3 images in R, G and B

    Returns the RGB and individual arrays
    Beware of the nan and 0 filtering out
    """
    # shape and first array in 3D
    ny, nx = r.shape

    if clip_value is not None:
        r[r <= clip_value] = np.nan
        g[g <= clip_value] = np.nan
        b[b <= clip_value] = np.nan

    # Returning the full arrays
    if isinstance(scale, str) : scale = [scale]
    if len(scale) == 1: scale = scale * 3
    if perc_rgb is None: perc_rgb = [def_perc]
    if len(perc_rgb) == 1 : perc_rgb = perc_rgb * 3
#    r = get_plot_norm(r, percentiles=perc_rgb[0], scale=scale[0], vmin=vmin[0], vmax=vmax[0])(r)
#    g = get_plot_norm(g, percentiles=perc_rgb[1], scale=scale[1], vmin=vmin[1], vmax=vmax[1])(g)
#    b = get_plot_norm(b, percentiles=perc_rgb[2], scale=scale[2], vmin=vmin[2], vmax=vmax[2])(b)

    # Transforming into RGB
    R = mcf.greyRGBize_image(r, rescalefn=scale[0]); 
    G = mcf.greyRGBize_image(g, rescalefn=scale[1]); 
    B = mcf.greyRGBize_image(b, rescalefn=scale[2]); 

#    # Mask all layers with nan
#    mask2d = r.mask | g.mask | b.mask
#    mask3d = np.repeat(mask2d[:, :, np.newaxis], 3, axis=2)

    # Colorize
    Rc = mcf.colorize_image(R, colors[0], colorintype='hex')
    Gc = mcf.colorize_image(G, colors[1], colorintype='hex')
    Bc = mcf.colorize_image(B, colors[2], colorintype='hex')

    # Combine
    RGB = mcf.combine_multicolor([Rc, Gc, Bc], gamma=2.2, inverse=inverse)

    # RGB = R + G + B
#    RGB = np.dstack((RGB, ~mask2d))

    return r, g, b, RGB, colors

def take_rules_for_plot(label, stretch='linear', zscale=False, 
                        percentiles=[5, 95], centred_cuts=False, 
                        use_defaults=True):
    """Using some rules for the imshow of maps

    Using either keywords or labels. Beware: not robust!!
    Priority is on LABELS
    """

    # Default
    pl_cmap = default_colormap
    pl_stretch = stretch
    pl_zc = zscale
    pl_perc = percentiles
    pl_cc = centred_cuts

    # if we find the label
    if use_defaults:
        if label in maps_rules_dict.keys():
            pl_cmap = maps_rules_dict[label][0]
            pl_stretch = maps_rules_dict[label][1]
            pl_zc = maps_rules_dict[label][2]
            pl_perc = maps_rules_dict[label][3]
            pl_cc = maps_rules_dict[label][4]
        # or the keyword
        else:
            for keyw in maps_keyw_dict.keys():
                if keyw in label:
                    pl_cmap = maps_keyw_dict[keyw][0]
                    pl_stretch = maps_keyw_dict[keyw][1]
                    pl_zc = maps_keyw_dict[keyw][2]
                    pl_perc = maps_keyw_dict[keyw][3]
                    pl_cc = maps_keyw_dict[keyw][4]
                    break

    return pl_cmap, pl_stretch, pl_zc, pl_perc, pl_cc

def make_rgb_imshow(r, g, b):
    fig, ax = plt.subplots()
    ax_r, ax_g, ax_b = make_rgb_axes(ax, pad=0.02)
    im_r, im_g, im_b, im_rgb = make_rgb(r, g, b)

# ================================================
class MuseDAPMaps(object):
    """Main class for such MAPS
    """
    def __init__(self, targetname, phangs, version="native", jwst="anchored", **kwargs):
        """Initialise the map structure

        Input
        -----
        targetname (str): name of the galaxy
        DR (str): number for the release. Relates to input dictionary
        mapsuffix (str): suffixes for the maps
        suffix (str): specific suffix to add to find the maps
        folder_maps (str): folder where the maps are
        folder_images (str): folder where the RGB reconstructed images are
        name_fits (str): name of the fits in case it needs to be defined
            By default it will use the standard naming for MUSE release
        """
        self.targetname = targetname
        self.version = version
        self.jwst = jwst

        # Folder of where the fits maps are
        self.folder_maps = kwargs.pop("folder_maps", f"{phangs.dr_folder}{version}_MAPS/")
        self.folder_images = kwargs.pop("folder_images", f"{phangs.dr_folder}{version}_IMAGES/")
        self.folder_jwst = kwargs.pop("folder_jwst", f"{phangs.jwst_folder}{jwst}/")
        self.suffix, self.mapsuffix, self.jwst_suffix = phangs.get_map_suffix(targetname, jwst=jwst, version=version)

        self.name_fits = kwargs.pop("name_fits", 
                                   (f"{self.folder_maps}{self.targetname}"
                                    f"{self.mapsuffix}.fits"))

        self._open_fits()

    def read_rgb_bands(self, rgb_band_dict=default_rgb_bands, factor=1.,
                       clip_value=0.):
        """Reading the 3 RGB bands
        """
        self.rgb_band_dict = rgb_band_dict
        self.Rname = (f"{self.folder_images}{self.targetname}_"
                     f"IMAGE_FOV_{rgb_band_dict['R']}_WCS_Pall_mad{self.suffix}.fits")
        self.Gname = (f"{self.folder_images}{self.targetname}_"
                     f"IMAGE_FOV_{rgb_band_dict['G']}_WCS_Pall_mad{self.suffix}.fits")
        self.Bname = (f"{self.folder_images}{self.targetname}_"
                     f"IMAGE_FOV_{rgb_band_dict['B']}_WCS_Pall_mad{self.suffix}.fits")
        if not os.path.isfile(self.Rname):
            print("ERROR: Rband file does not exist")
            R = np.empty((0,0))
        if not os.path.isfile(self.Gname):
            print("ERROR: Gband file does not exist")
            G = np.empty((0,0))
        if not os.path.isfile(self.Bname):
            print("ERROR: Bband file does not exist")
            B = np.empty((0,0))

        Rdata = pyfits.getdata(self.Rname).astype(np.float64)
        Gdata = pyfits.getdata(self.Gname).astype(np.float64)
        Bdata = pyfits.getdata(self.Bname).astype(np.float64)
        if Rdata.shape != Bdata.shape:
            print("ERROR: R and B images do not have the same size")
            Bdata = Rdata
        if Rdata.shape != Gdata.shape:
            print("ERROR: R and g images do not have the same size")
            Gdata = Rdata

        # Clipping ?
        if clip_value is not None:
            Rdata = np.clip(Rdata, clip_value, None)
            Gdata = np.clip(Gdata, clip_value, None)
            Bdata = np.clip(Bdata, clip_value, None)
        self.Rdata = Rdata
        self.Gdata = Gdata
        self.Bdata = Bdata

    def read_jwst_bands(self, bands=[770, 1130, 2100], jwst_band_dict=default_jwst_bands, factor=1.,
                        clip_value=0., suffix=None):
        """Reading 3 RGB bands
        """
        self.jwst_band_dict = jwst_band_dict
        if suffix is None:
            suffix = self.jwst_suffix
        self.Rname = (f"{self.folder_jwst}{self.targetname.lower()}_"
                     f"{jwst_band_dict[bands[0]]}_{suffix}.fits")
        self.Gname = (f"{self.folder_jwst}{self.targetname.lower()}_"
                     f"{jwst_band_dict[bands[1]]}_{suffix}.fits")
        self.Bname = (f"{self.folder_jwst}{self.targetname.lower()}_"
                     f"{jwst_band_dict[bands[2]]}_{suffix}.fits")
        if not os.path.isfile(self.Rname):
            print("ERROR: Rband file does not exist")
            R = np.empty((0,0))
        if not os.path.isfile(self.Gname):
            print("ERROR: Gband file does not exist")
            G = np.empty((0,0))
        if not os.path.isfile(self.Bname):
            print("ERROR: Bband file does not exist")
            B = np.empty((0,0))

        Rdata, hr = pyfits.getdata(self.Rname, header=True)
        Gdata, hg = pyfits.getdata(self.Gname, header=True)
        Bdata, hb = pyfits.getdata(self.Bname, header=True)
        if Rdata.shape != Bdata.shape:
            print("ERROR: R and B images do not have the same size")
            Bdata = Rdata
        if Rdata.shape != Gdata.shape:
            print("ERROR: R and g images do not have the same size")
            Gdata = Rdata

        # Clipping ?
        if clip_value is not None:
            Rdata = np.clip(Rdata, clip_value, None)
            Gdata = np.clip(Gdata, clip_value, None)
            Bdata = np.clip(Bdata, clip_value, None)
        self.Rdata = Rdata
        self.Gdata = Gdata
        self.Bdata = Bdata
        self.jwst_r_scale = np.abs(hr['CD1_1']) * 3600.
        self.jwst_g_scale = np.abs(hg['CD1_1']) * 3600.
        self.jwst_b_scale = np.abs(hb['CD1_1']) * 3600.


    def _open_fits(self):
        if not self._check_namefits:
            self._maps = None
        elif os.path.isfile(self.name_fits):
            self._maps = pyfits.open(self.name_fits)
        else:
            self._maps = None

    @property
    def maps(self):
        if not hasattr(self, '_maps'):
            self._open_fits()
        return self._maps

    @property
    def _check_namefits(self):
        if os.path.isfile(self.name_fits):
            return True
        else:
            print(f"Warning = Filename {self.name_fits} does not exist")
            return False

    def _check_map_name(self, name=None):
        return name in self.map_names

    @property
    def nmaps(self):
        return len(self.map_names)

    @property
    def map_names(self):
        return [self.maps[i].header['EXTNAME'] for i in range(1, len(self.maps))]

    def print_map_names(self):
        for i, name in enumerate(self.map_names):
            print(f"Map {i+1:03d} = {name}")

    def read_datamap(self, map_name=None, n=1, err=False):
        """Reading the maps from a certain quantity
        """
        if map_name is None:
            if n < 1 or n > self.nmaps:
                print("ERROR: data number {n} not within the [1..{self.nmaps}] range")
                return None, None
            map_name = self.map_names[n-1]

        # If RGB, return the RGB matrix
        if map_name == "RGB":
            return np.ma.masked_array(self.RGBAdata, np.isnan(self.RGBAdata))
            
        if map_name not in self.map_names:
            print("ERROR: data name {map_name} not in list of map names")
            return None, None

        # Returning a masked data using Nan as a criterion
        data = self.maps[map_name].data
        if err is True:
            edata = self.maps[get_name_err_map(map_name)]
        else:
            edata = None

        return [data, edata]

    def save_datamap(self, map_name=None, n=1, overwrite=False):
        """Save data (and err) map into a fits image

        map_name (str): name of the map
        n (int): number of map (if no name is given)
        overwrite (bool): False by default
        """
        if map_name is None:
            if n < 1 or n > self.nmaps:
                print("ERROR: data number {n} not within the [1..{self.nmaps}] range")
                return None, None
            map_name = self.map_names[n-1]

        if map_name not in self.map_names:
            print("ERROR: data name {map_name} not in list of map names")
            return

        # Extracting the data
        data, err = self.read_datamap(map_name)
        if data is None:
            print("Problem extracting the data from the data maps")
            return

        # Creating the primary header copying the header
        empty_primary = pyfits.PrimaryHDU(header=self.maps[0].header)

        # Creating the hdu
        if err is not None:
            hdulist = pyfits.HDUList([empty_primary, data, err])
        else:
            hdulist = pyfits.HDUList([empty_primary, data])

        fname, ext = os.path.splitext(self.name_fits)
        hdulist.writeto(f"{fname}_{map_name}{ext}", overwrite=overwrite)

    def show_map(self, list_labels=[], rgb=True, ncols=4, width=15, text_info="", 
                 stretch='linear', zscale=False, percentiles=[2, 99], 
                 factor_RGB=1.0, use_defaults=True, perc_rgb=[2, 99]):
        """Build the figure with the list of maps
        """

        # Decide to add the RGB or not
        labels = copy.copy(list_labels)
        if rgb:
            self.read_rgb_bands(factor=factor_RGB, perc_rgb=perc_rgb)
            list_labels.insert(0, "RGB")
            labels.insert(0, f"R={self.rgb_band_dict['R']}, "
                          f"G={self.rgb_band_dict['G']}, "
                          f"B={self.rgb_band_dict['B']}")

        # number of maps to decide how many rows/cols
        nmaps = len(list_labels)

        if nmaps < 1:
            print("ERROR: list is empty, no map to show - Aborting")
            return

        nrows = (nmaps - 1) // ncols + 1
        if nrows == 1:
            ncols = nmaps

        # size of the figures
        # Basically = 15 for width / ncols
        #             nrows * (15/ncols) + 1
        # First get the array axis ratio
        shape = self.read_datamap(name=list_labels[0])[0].shape[:2]

        height = nrows * (width * shape[0] / (shape[1] * ncols)) + 1
        fig = plt.figure(num=1, figsize=(width, height))
        ax = plt.clf()
        text_info = f"{text_info} [{self.version}]"
        fig.suptitle(f'{self.targetname}{text_info}', fontsize=16)
        gs = GridSpec(nrows, ncols, wspace=0, hspace=0)
        axs = []
        ims = []
        for i in range(nmaps):
            # Adding the axis
            a = i // ncols 
            b = i % ncols
            axs.append(fig.add_subplot(gs[a, b]))
            # Reading the data
            data, _ = self.read_datamap(name=list_labels[i])[0].astype(np.float64)
            pl_cmap, pl_stretch, pl_zc, pl_perc, pl_cc = take_rules_for_plot(list_labels[i], 
                                                           stretch, zscale, percentiles, 
                                                           use_defaults=use_defaults)
            # Showing the data with labels
            norm = get_plot_norm(data, zscale=pl_zc, scale=pl_stretch, 
                                 percentiles=pl_perc, centred_cuts=pl_cc)
            ims.append(axs[-1].imshow(data, cmap=pl_cmap, interpolation='nearest', 
                                      origin='lower', norm=norm))
            set_axtext(ims[-1], axs[-1], labels[i])
            axs[-1].set_xticks([])
            axs[-1].set_yticks([])

        plt.tight_layout()
