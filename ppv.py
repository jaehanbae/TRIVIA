import numpy as np
import cmasher as cmr
import plotly.graph_objects as go # https://plotly.com/python/3d-scatter-plots/
from gofish import imagecube

def ppv(path, clip=5., N=None, cmin=None, cmax=None, constant_opacity=None, ntrace=20, 
        marker_size=2, cmap=None, hoverinfo='x+y+z', colorscale=None, xaxis_title=None, 
        yaxis_title=None, zaxis_title=None, xaxis_backgroundcolor=None, xaxis_gridcolor=None,
        yaxis_backgroundcolor=None, yaxis_gridcolor=None,
        zaxis_backgroundcolor=None, zaxis_gridcolor=None,
        xmin=None, xmax=None, ymin=None, ymax=None, zmin=None, zmax=None,
        projection_x=False, projection_y=False, projection_z=True,
        show_colorbar=True, camera_eye_x=-1., camera_eye_y=-2., camera_eye_z=1.,
        show_figure=False, write_pdf=True, write_html=True):
    """
    Create a three-dimensional position-position-velocity diagram.

    Args:
        path (str): Relative path to the FITS cube.
        clip (Optional[float]): Clip the cube having cube.data > clip * cube.rms
        N (Optional[integer]): Downsample the data. 
        cmin (Optional[float]): The lower bound of the velocity for the colorscale in km/s. 
        cmax (Optional[float]): The upper bound of the velocity for the colorscale in km/s. 
        constant_opacity (Optional[float]): If not None, use a constant opacity of the given value.
        ntrace (Optional[integer]): Number of opacity layers.
        markersize (Optional[integer]): Size of the marker in the PPV diagram.
        cmap (Optional[str]): Name of the colormap to use for the PPV diagram.
        hoverinfo (Optional[str]): Determines which trace information appear on hover.
                   Any combination of "x", "y", "z", "text", "name" joined with a "+" 
                   or "all" or "none" or "skip". If `none` or `skip` are set, no 
                   information is displayed upon hovering. But, if `none` is set, 
                   click and hover events are still fired.
        xaxis_title (Optional[str]): X-axis title.
        yaxis_title (Optional[str]): Y-axis title.
        zaxis_title (Optional[str]): Z-axis title.
        xaxis_backgroundcolor (Optional[str]): X-axis background color.
        xaxis_gridcolor (Optional[str]): X-axis grid color.
        yaxis_backgroundcolor (Optional[str]): Y-axis background color.
        yaxis_gridcolor (Optional[str]): Y-axis grid color.
        zaxis_backgroundcolor (Optional[str]): Z-axis background color.
        zaxis_gridcolor (Optional[str]): Z-axis grid color.
        xmin (Optional[float]): The lower bound of PPV diagram X range.
        xmax (Optional[float]): The upper bound of PPV diagram X range.
        ymin (Optional[float]): The lower bound of PPV diagram Y range.
        ymax (Optional[float]): The upper bound of PPV diagram Y range.
        zmin (Optional[float]): The lower bound of PPV diagram Z range.
        zmax (Optional[float]): The upper bound of PPV diagram Z range.
        projection_x (Optional[bool]): Whether or not to add projection on the Y-Z plane.
        projection_y (Optional[bool]): Whether or not to add projection on the X-Z plane.
        projection_z (Optional[bool]): Whether or not to add projection on the X-Y plane.
        show_colorbar (Optional[bool]): Whether or not to plot a colorbar.
        camera_eye_x (Optional[float]): The x component of the 'eye' camera vector.
        camera_eye_y (Optional[float]): The y component of the 'eye' camera vector.
        camera_eye_z (Optional[float]): The z component of the 'eye' camera vector.
        show_figure (Optional[bool]): If True, show PPV diagram.
        write_pdf (Optional[bool]): If True, write PPV diagram in a pdf file.
        write_html (Optional[bool]): If True, write PPV diagram in a html file.
    Returns:
        PPV diagram in a pdf and/or a html format.
    """
    # Read in the FITS data.
    cube = imagecube(path)

    # Generate a mask based on SNR.
    SNR_mask = cube.data > clip * cube.rms

    # Sigma-clipped LOS velocity, RA, Dec, intensity arrays.
    v = (cube.velax[:, None, None] * np.ones(cube.data.shape))[SNR_mask]/1e3
    x = (cube.xaxis[None, None, :] * np.ones(cube.data.shape))[SNR_mask]
    y = (cube.yaxis[None, :, None] * np.ones(cube.data.shape))[SNR_mask]
    i = cube.data[SNR_mask]

    # Take N random voxel.
    N = np.int(np.max([v.size/1.0e5,1])) if N is None else N
    if N > 1:
        idx = np.arange(v.size) 
        np.random.shuffle(idx)
        v = v[idx][::N]
        x = x[idx][::N]
        y = y[idx][::N]
        i = i[idx][::N]

    # Normalize the intensity.
    i = (i - i.min())/(i.max() - i.min())

    # Determine the opacity of the data points.
    cuts = np.linspace(0, 1, ntrace+1)
    opacity = np.logspace(-1., 0.5, cuts.size - 1)
    if constant_opacity is not None:
        opacity[:] = constant_opacity
    datas = []

    colorscale = generate_colorscale('cmr.pride') if colorscale is None else cmap
    cmin = 0.5*(cube.velax.min() + cube.velax.max())/1.0e3 - 4 if cmin is None else cmin
    cmax = 0.5*(cube.velax.min() + cube.velax.max())/1.0e3 + 4 if cmax is None else cmax

    # 3d scatter plot
    for a, alpha in enumerate(opacity):
        mask = np.logical_and(i >= cuts[a], i < cuts[a+1])
        datas += [go.Scatter3d(x=x[mask], y=y[mask], z=v[mask], mode='markers',
                               marker=dict(size=marker_size, color=v[mask], colorscale=colorscale,
                                           cauto=False, cmin=cmin, cmax=cmax,
                                           opacity=min(1.0, alpha)),
                               hoverinfo=hoverinfo,
#                              name='I ='+(str('% 4.2f' % cuts[a]))+' -'+(str('% 4.2f' % cuts[a+1]))
                              )
                 ]

    xaxis_title = 'RA offset [arcsec]' if xaxis_title is None else xaxis_title
    yaxis_title = 'Dec offset [arcsec]' if yaxis_title is None else yaxis_title
    zaxis_title = 'velocity [km/s]' if zaxis_title is None else zaxis_title
    xaxis_backgroundcolor = 'white' if xaxis_backgroundcolor is None else xaxis_backgroundcolor
    xaxis_gridcolor = 'gray' if xaxis_gridcolor is None else xaxis_gridcolor
    yaxis_backgroundcolor = 'white' if yaxis_backgroundcolor is None else yaxis_backgroundcolor
    yaxis_gridcolor = 'gray' if yaxis_gridcolor is None else yaxis_gridcolor
    zaxis_backgroundcolor = 'white' if zaxis_backgroundcolor is None else zaxis_backgroundcolor
    zaxis_gridcolor = 'gray' if zaxis_gridcolor is None else zaxis_gridcolor
    xmin = cube.FOV/2.0 if xmin is None else xmin
    xmax = -cube.FOV/2.0 if xmax is None else xmax
    ymin = -cube.FOV/2.0 if ymin is None else ymin
    ymax = cube.FOV/2.0 if ymax is None else ymax
    zmin = cube.velax.min()/1e3 if zmin is None else zmin
    zmax = cube.velax.max()/1e3 if zmax is None else zmax

    # layout
    layout = go.Layout(scene=dict(xaxis_title=xaxis_title, 
                                  yaxis_title=yaxis_title,
                                  zaxis_title=zaxis_title,
                                  xaxis_backgroundcolor=xaxis_backgroundcolor, 
                                  xaxis_gridcolor=xaxis_gridcolor,
                                  yaxis_backgroundcolor=yaxis_backgroundcolor, 
                                  yaxis_gridcolor=yaxis_gridcolor,
                                  zaxis_backgroundcolor=zaxis_backgroundcolor, 
                                  zaxis_gridcolor=zaxis_gridcolor,
                                  xaxis_range=[xmin, xmax],
                                  yaxis_range=[ymin, ymax],
                                  zaxis_range=[zmin, zmax],
                                  aspectmode='cube'),
                       margin=dict(l=0, r=0, b=0, t=0), showlegend=False)

    fig = go.Figure(data=datas, layout=layout)

    fig.update_traces(projection_x=dict(show=projection_x, opacity=1), 
                      projection_y=dict(show=projection_y, opacity=1),
                      projection_z=dict(show=projection_z, opacity=1),
                     )

    if show_colorbar:
        fig.update_traces(marker_colorbar=dict(thickness=20, 
                                               tickvals=np.arange(cmin,cmax+1,1),
                                               tickformat='.2f',
                                               title='v [km/s]',
                                               title_side='right',
                                               len=0.5
                                              )
                         )

    camera = dict(up=dict(x=0, y=0, z=1),
                  center=dict(x=0, y=0, z=0),
                  eye=dict(x=camera_eye_x, y=camera_eye_y, z=camera_eye_z)
                 )

    fig.update_layout(scene_camera=camera)

    if show_figure:
        fig.show()
    if write_pdf:
        fig.write_image(path.replace('.fits', '_ppv.pdf'))
    if write_html:
        fig.write_html(path.replace('.fits', '_ppv.html'), include_plotlyjs='cdn')
    return

def generate_colorscale(cmap):
    """
    Convert a CMasher color table into a plotly-compatible color table.
    See https://cmasher.readthedocs.io/

    Args:
        cmap (str): CMasher color table name. e.g., 'cmr.pride'

    Returns:
        A list containing plotly-compatible color table.
    """
    cmarr = np.array(cmr.take_cmap_colors('cmr.pride', 128))
    colorscale = [[f, 'rgb({}, {}, {})'.format(*cmarr[ff])]
                  for ff, f in enumerate(np.linspace(0, 1, cmarr.shape[0]))]
    return colorscale
