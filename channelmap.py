from gofish import imagecube
import plotly.express as px

def rebin1d(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0])
    return arr.reshape(shape).mean(-1)


def rebin3d(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1],
             new_shape[2], arr.shape[2] // new_shape[2])
    return arr.reshape(shape).mean(-1).mean(-2).mean(-3)

def channelmap(path, vmin=None, vmax=None, nv=None, nx=None, ny=None,
               ):
    """
    Create interactive channel map.

    Args:
        path (str): Relative path to the FITS cube.
        vmin (Optional[float]): The lower bound of the flux. 
        vmax (Optional[float]): The upper bound of the flux.
        nv (Optional[float]): Number of velocity channels.
        nx (Optional[float]): Number of x pixels.
        ny (Optional[float]): Number of y pixels.

    Returns:
        Interactive channel map in a html format.
    """
    # Read in the FITS data.
    cube = imagecube(path)

    vmin = 0 if vmin is None else vmin
    vmax = cube.data.max()*0.5 if vmax is None else vmax
    funit = 'Jy/beam'
    if vmax < 0.5 :
        cube.data *= 1.0e3
        vmin *= 1.0e3
        vmax *= 1.0e3
        funit = 'mJy/beam'

    nv = cube.data.shape[0] if nv is None else nv
    nx = 400 if nx is None else nx
    ny = 400 if ny is None else ny

    vaxis = rebin1d(cube.velax,[nv])
    xaxis = rebin1d(cube.xaxis,[nx])
    yaxis = rebin1d(cube.yaxis,[ny])
    toplot = rebin3d(cube.data,[nv,nx,ny])

    fig = px.imshow(toplot, color_continuous_scale='inferno', origin='lower', 
                    x=xaxis, y=yaxis,
                    zmin=vmin, zmax=vmax, 
                    labels=dict(x="RA offset [arcsec]", y="Dec offset [arcsec]", 
                                color="Intensity ["+funit+"]", animation_frame="channel"),
                    animation_frame=0,
                   )
    fig.update_xaxes(autorange="reversed")
    for i, frame in enumerate(fig.frames):
        frame['layout'].update(title_text="v= {:.2f} km/s".format(vaxis[i]/1.0e3),
                               title_x=0.5,
                              )
    fig.write_html(path.replace('.fits', '_channel.html'), include_plotlyjs='cdn')
    return

