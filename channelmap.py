from gofish import imagecube
import plotly.express as px
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

def rebin1d(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0])
    return arr.reshape(shape).mean(-1)

def rebin2d(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(-2)

def rebin3d(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1],
             new_shape[2], arr.shape[2] // new_shape[2])
    return arr.reshape(shape).mean(-1).mean(-2).mean(-3)

def concatenate_cmaps(cmap1, cmap2, ratio=None, ntot=None):
    """
    Concatenate two colormaps.
    https://matplotlib.org/stable/tutorials/colors/colormap-manipulation.html

    Args:
        cmap1 (str): Name of the first colormap (bottom) to concatenate.
        cmap2 (str): Name of the second colormap (top) to concatenate.
        ratio (Optional[float]): The ratio between the first and second colormap.
        ntot (Optional[int]): The number of levels in the concatenated colormap.
    """
    ratio = 0.5 if ratio is None else ratio
    ntot = 256 if ntot is None else ntot

    bottom = cm.get_cmap(cmap1, ntot)
    top = cm.get_cmap(cmap2, ntot)
    nbottom = int(ratio*ntot)
    ntop = ntot-nbottom
    newcolors = np.vstack((bottom(np.linspace(0, 1, nbottom)),
                       top(np.linspace(0, 1, ntop))))
    newcmp = ListedColormap(newcolors, name='newcolormap')    
    newcmp = np.around(newcmp(range(ntot)),decimals=4)
    colorscale = [[f, 'rgb({}, {}, {})'.format(*newcmp[ff])]
              for ff, f in enumerate(np.around(np.linspace(0, 1, newcmp.shape[0]),decimals=4))]
    return colorscale

def channelmap(path, vmin=None, vmed=None, vmax=None, nv=None, nx=None, ny=None,
               cmap=None, show_figure=False):
    """
    Create interactive channel map.

    Args:
        path (str): Relative path to the FITS cube.
        vmin (Optional[float]): The lower bound of the flux. 
        vmax (Optional[float]): The upper bound of the flux.
        nv (Optional[float]): Number of velocity channels.
        nx (Optional[float]): Number of x pixels.
        ny (Optional[float]): Number of y pixels.
        cmap (Optional[str]): Color map to use.
        show_figure (Optional[bool]): If True, show channel map.
    Returns:
        Interactive channel map in a html format.
    """
    # Read in the FITS data.
    cube = imagecube(path)

    vmin = 0. if vmin is None else vmin
    vmed = 5.*cube.rms if vmed is None else vmed
    vmax = cube.data.max()*0.5 if vmax is None else vmax
    funit = 'Jy/beam'
    if vmax < 0.5 :
        cube.data *= 1.0e3
        vmin *= 1.0e3
        vmed *= 1.0e3
        vmax *= 1.0e3
        funit = 'mJy/beam'

    nv = cube.data.shape[0] if nv is None else nv
    nx = 400 if nx is None else nx
    ny = 400 if ny is None else ny

    vaxis = np.around(rebin1d(cube.velax,[nv]),decimals=3)
    xaxis = np.around(rebin1d(cube.xaxis,[nx]),decimals=3)
    yaxis = np.around(rebin1d(cube.yaxis,[ny]),decimals=3)
    toplot = np.around(rebin3d(cube.data,[nv,nx,ny]),decimals=3)

    cmap = concatenate_cmaps('binary','inferno',ratio=vmed/vmax) if cmap is None else cmap

    fig = px.imshow(toplot, color_continuous_scale=cmap, origin='lower', 
                    x=xaxis, y=yaxis,
                    zmin=vmin, zmax=vmax, 
                    labels=dict(x="RA offset [arcsec]", y="Dec offset [arcsec]", 
                                color="Intensity ["+funit+"]", animation_frame="channel"),
                    animation_frame=0,
                   )
    fig.update_xaxes(autorange="reversed")
    fig.update_xaxes(ticks="outside")
    fig.update_yaxes(ticks="outside")
    for i, frame in enumerate(fig.frames):
        frame['layout'].update(title_text="v= {:.2f} km/s".format(vaxis[i]/1.0e3),
                               title_x=0.5,
                              )
    fig.write_html(path.replace('.fits', '_channel.html'), include_plotlyjs='cdn')

    if show_figure:
        fig.show()
    return

