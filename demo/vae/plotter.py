"""Plotters for the VAE example."""

__all__ = ["plot_test_sample_image",
           "plot_elbo",
           "plot_latent_image",
           "overlay_datapoints", "remove_overlay"]

from collections import defaultdict
import typing

from unpythonic.env import env

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_probability as tfp

from extrafeathers import plotmagic

from .cvae import CVAE

def plot_test_sample_image(test_sample: tf.Tensor, *,
                           model: typing.Optional[CVAE] = None,
                           epoch: typing.Optional[int] = None,
                           figno: int = 1) -> None:
    """Plot image of test sample and the corresponding prediction (by feeding the sample through the CVAE).

    `test_sample`: tensor of size `[n, 28, 28, 1]`, where `n = 16` (test sample size).
    `model`: `CVAE` instance, or `None` to use the default instance.
    `epoch`: if specified, included in the figure title.
    `figno`: matplotlib figure number.
    """
    if model is None:
        from . import main
        model = main.model
    assert isinstance(model, CVAE)

    batch_size, n_pixels_y, n_pixels_x, n_channels = tf.shape(test_sample).numpy()
    assert batch_size == 16, f"This function currently assumes a test sample of size 16, got {batch_size}"
    assert n_channels == 1, f"This function currently assumes grayscale images, got {n_channels} channels"

    mean, logvar = model.encode(test_sample)
    ignored_eps, z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)

    n = 4  # how many images per row/column; sqrt(batch_size)
    image_width = (2 * n + 1) * n_pixels_x  # extra empty column at center, as separator
    image_height = n * n_pixels_y
    image = np.zeros((image_height, image_width))

    for i in range(batch_size):
        x_orig = test_sample[i, :, :, 0]
        x_hat = predictions[i, :, :, 0]
        row, base_col = divmod(i, n)
        col1 = base_col  # original image (input)
        col2 = base_col + n + 1  # reconstructed image
        image[row * n_pixels_y: (row + 1) * n_pixels_y,
              col1 * n_pixels_x: (col1 + 1) * n_pixels_x] = x_orig.numpy()
        image[row * n_pixels_y: (row + 1) * n_pixels_y,
              col2 * n_pixels_x: (col2 + 1) * n_pixels_x] = x_hat.numpy()

    fig = plt.figure(figno)
    if not fig.axes:
        plt.subplot(1, 1, 1)  # create Axes
        fig.set_figwidth(8)
        fig.set_figheight(4)
    ax = fig.axes[0]
    ax.cla()
    plt.sca(ax)
    fig.tight_layout()  # prevent axes crawling
    ax.imshow(image, cmap="Greys_r")
    ax.axis("off")

    epoch_str = f"; epoch {epoch}" if epoch is not None else ""
    ax.set_title(f"Test sample (left: input $\\mathbf{{x}}$, right: prediction $\\hat{{\\mathbf{{x}}}}$){epoch_str}")
    fig.tight_layout()
    plt.draw()
    plotmagic.pause(0.1)  # force redraw


def plot_elbo(epochs, train_elbos, test_elbos, *,
              epoch: typing.Optional[int] = None,
              figno: int = 1) -> None:
    """Plot the evidence lower bound for the training and test sets as a function of the epoch number."""
    fig = plt.figure(figno)
    if not fig.axes:
        plt.subplot(1, 1, 1)  # create Axes
        fig.set_figwidth(6)
        fig.set_figheight(4)
    ax = fig.axes[0]
    ax.cla()
    plt.sca(ax)
    fig.tight_layout()  # <-- important to do this also here to prevent axes crawling

    ax.plot(epochs, train_elbos, label="train")
    ax.plot(epochs, test_elbos, label="test")

    # Zoom to top 80% of data mass
    q = np.quantile(train_elbos, 0.2)
    datamax = max(np.max(train_elbos), np.max(test_elbos))
    ax.set_ylim(q, datamax)

    ax.xaxis.grid(visible=True, which="both")
    ax.yaxis.grid(visible=True, which="both")
    # https://stackoverflow.com/questions/30914462/how-to-force-integer-tick-labels
    # https://matplotlib.org/stable/api/ticker_api.html#matplotlib.ticker.MaxNLocator
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    ax.set_xlabel("epoch")
    ax.set_ylabel("ELBO")
    ax.legend(loc="best")

    fig.tight_layout()
    plt.draw()
    plotmagic.pause(0.1)


def normal_grid(n: int = 20, *, kind: str = "quantile", eps: float = 3):
    """Make a grid on `[-εσ, +εσ]` for evaluating normally distributed quantities.

    μ = 0, σ = 1; shift and scale the result manually if necessary.

    `n`: number of points

    `grid`: grid spacing type; one of "linear" or "quantile" (default)

            "quantile" has normally distributed density, placing more emphasis
            on the region near the origin, where most of the gaussian probability
            mass is concentrated. This grid is linear in cumulative probability.

            "linear" is just a linear spacing. It effectively emphasizes the faraway
            regions, since the gaussian does not have much probability mass there.

    `eps`:  ε for lower/upper limit ±εσ. E.g. the default 3 means ±3σ.
    """
    assert kind in ("linear", "quantile")

    gaussian = tfp.distributions.Normal(0, 1)
    pmin = gaussian.cdf(-eps)  # cdf(x) := P[X ≤ x], so this is P[x ≤ -εσ], where σ = 1

    if kind == "quantile":  # quantile(p) := {x | P[X ≤ x] = p}
        # xx = gaussian.quantile(np.linspace(p, 1 - p, n)).numpy()  # yields +inf at ≥ +6σ
        xx = gaussian.quantile(np.linspace(pmin, 0.5, n // 2 + 1)).numpy()
        xx_left = xx[:-1]
        xx_right = -xx_left[::-1]
        if n % 2 == 0:
            xx = np.concatenate((xx_left, xx_right))
        else:
            xx = np.concatenate((xx_left, [0.0], xx_right))
    else:  # kind == "linear":
        xmin = gaussian.quantile(pmin)
        xmax = -xmin
        xx = np.linspace(xmin, xmax, n)

    assert np.shape(xx)[0] == n
    return xx


def plot_latent_image(n: int = 20, *,
                      grid: str = "quantile",
                      eps: float = 3,
                      model: typing.Optional[CVAE] = None,
                      digit_size: int = 28,
                      epoch: typing.Optional[int] = None,
                      figno: int = 1) -> env:
    """Plot n × n digit images decoded from the latent space.

    `n`, `grid`, `eps`: passed to `normal_grid` (`grid` is the `kind`)

                        A quantile grid is linear in cumulative probability according to the
                        latent prior. However, using the prior is subtly wrong, and the marginal
                        posterior of z should be used instead; see Lin et al.

    `model`: `CVAE` instance, or `None` to use the default instance.
    `digit_size`: width/height of each digit image (square-shaped), in pixels.
                  Must match what the model was trained for.
    `epoch`: if specified, included in the figure title.
    `figno`: matplotlib figure number.
    """
    if model is None:
        from . import main
        model = main.model
    assert isinstance(model, CVAE)

    image_width = digit_size * n
    image_height = image_width
    image = np.zeros((image_height, image_width))

    zz = normal_grid(n, kind=grid, eps=eps)
    grid_x = zz
    grid_y = zz

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z = np.array([[xi, yi]])
            x_decoded = model.sample(z)
            digit = tf.reshape(x_decoded[0], (digit_size, digit_size))
            # flip y so we can use origin="lower" for plotting the complete tiled image
            image[i * digit_size: (i + 1) * digit_size,
                  j * digit_size: (j + 1) * digit_size] = digit.numpy()[::-1, :]

    fig = plt.figure(figno)
    if not fig.axes:
        plt.subplot(1, 1, 1)  # create Axes
        fig.set_figwidth(10)
        fig.set_figheight(10)
    remove_overlay(figno)
    ax = fig.axes[0]
    ax.cla()
    plt.sca(ax)
    fig.tight_layout()  # <-- important to do this also here to prevent axes crawling
    ax.imshow(image, origin="lower", cmap="Greys_r")
    # print(ax._position.bounds)  # DEBUG

    # Show latent space coordinates
    #
    # We use ticks aligned to the image centers ticks for the bare latent space plot.
    # Each image corresponds to its center point.
    #
    # If a dataset overlay is displayed later, the ticks will be adjusted so that they
    # cover the whole data area, which looks better.
    #
    startx = digit_size / 2
    endx = image_width - (digit_size / 2)

    tick_positions_x = np.array(startx + np.linspace(0, 1, len(grid_x)) * (endx - startx), dtype=int)
    tick_positions_y = tick_positions_x
    ax.set_xticks(tick_positions_x, [f"{x:0.3g}" for x in grid_x], rotation="vertical")
    ax.set_yticks(tick_positions_y, [f"{y:0.3g}" for y in grid_y])

    ax.set_xlabel(r"$z_{1}$")
    ax.set_ylabel(r"$z_{2}$")

    epoch_str = f"; epoch {epoch}" if epoch is not None else ""
    ax.set_title(f"Latent space ({grid} grid, up to ±{eps}σ){epoch_str}")

    fig.tight_layout()

    plt.draw()
    plotmagic.pause(0.1)  # force redraw

    return env(n=n, model=model, digit_size=digit_size, grid=grid, eps=eps, figno=figno)


def remove_overlay(figno: int = 1):
    """Remove previous datapoint overlay in figure `figno`, if any."""
    fig = plt.figure(figno)

    # We're making a new overlay; clean up old stuff added to figure `figno` by this function.
    for cb in _overlay_colorbars.pop(figno, []):
        cb.remove()
    for cid in _overlay_callbacks.pop(figno, []):
        fig.canvas.mpl_disconnect(cid)
    if len(fig.axes) > 1:
        for ax in fig.axes[1:]:
            ax.remove()
    fig.set_figwidth(fig.get_figheight())
    fig.tight_layout()

    plt.draw()  # force update of extents
    plotmagic.pause(0.1)


_overlay_colorbars = defaultdict(list)
_overlay_callbacks = defaultdict(list)
def overlay_datapoints(x: tf.Tensor, labels: tf.Tensor, figdata: env, alpha: float = 0.1) -> None:
    """Overlay the codepoints corresponding to a dataset `x` and `labels` onto the latent space plot.

    `figdata`: metadata describing the figure on which to overlay the plot.
               This is the return value of `plot_latent_image`.
    `alpha`: opacity of the scatterplot points.
    """
    n = figdata.n
    model = figdata.model
    digit_size = figdata.digit_size
    grid = figdata.grid
    eps = figdata.eps
    figno = figdata.figno

    assert isinstance(model, CVAE)

    # Find latent distribution parameters for the given data.
    # We'll plot the means.
    mean, logvar = model.encode(x)

    # We need some gymnastics to plot on top of an imshow image; it's easiest to
    # overlay a new Axes with a transparent background.
    # https://stackoverflow.com/questions/16829436/overlay-matplotlib-imshow-with-line-plots-that-are-arranged-in-a-grid
    fig = plt.figure(figno)
    remove_overlay(figno)
    if not fig.axes:
        raise ValueError(f"Figure {figno} has no existing Axes; nothing to overlay on.")
    ax = fig.axes[0]  # the Axes we're overlaying the dataset on
    # axs = fig.axes  # list of all Axes objects in this Figure

    # print([int(x) for x in fig.axes[0].get_xlim()])

    # Compute position for overlay:
    #
    # Determine the centers of images at two opposite corners of the sheet,
    # in data coordinates of the imshow plot.
    image_width = digit_size * n

    # # In this variant, the data area ends at the center of the edgemost images.
    # # Doesn't look good, the dataset is cut off before the image area ends.
    # xmin = digit_size / 2
    # xmax = image_width - (digit_size / 2)
    #
    # Use whole data area - looks much better.
    xmin = 0
    xmax = image_width

    ymin = xmin
    ymax = xmax
    xy0 = [xmin, ymin]
    xy1 = [xmax, ymax]

    # Adjust the ticks of the parent plot to match. Now the centermost image corresponds
    # to its center point; cornermost images correspond to their outermost corner points.
    startx = 0
    endx = image_width - 1
    zz = normal_grid(n, kind=grid, eps=eps)
    grid_x = zz
    grid_y = zz
    tick_positions_x = np.array(startx + np.linspace(0, 1, len(grid_x)) * (endx - startx), dtype=int)
    tick_positions_y = tick_positions_x
    ax.set_xticks(tick_positions_x, [f"{x:0.3g}" for x in grid_x], rotation="vertical")
    ax.set_yticks(tick_positions_y, [f"{y:0.3g}" for y in grid_y])

    # Convert to figure coordinates.
    def data_to_fig(xy):
        """Convert Matplotlib data coordinates (of current axis) to figure coordinates."""
        # https://matplotlib.org/stable/tutorials/advanced/transforms_tutorial.html
        xy_ax = ax.transLimits.transform(xy)  # data coordinates -> axes coordinates
        xy_disp = ax.transAxes.transform(xy_ax)  # axes -> display
        xy_fig = fig.transFigure.inverted().transform(xy_disp)  # display -> figure
        # print(f"data: {xy}")
        # print(f"ax:   {xy_ax}")
        # print(f"disp: {xy_disp}")
        # print(f"fig:  {xy_fig}")
        return xy_fig

    def compute_overlay_position():  # in figure coordinates
        x0, y0 = data_to_fig(xy0)
        x1, y1 = data_to_fig(xy1)
        box = [x0, y0, (x1 - x0), (y1 - y0)]
        return box

    # Set up the new Axes, no background (`set_axis_off`), and plot the overlay.
    box = compute_overlay_position()
    newax = fig.add_axes(box, label="<custom overlay>")
    newax.set_axis_off()

    # https://matplotlib.org/stable/users/explain/event_handling.html
    def onresize(event):
        fig.tight_layout()
        plt.draw()
        plotmagic.pause(0.001)
        box = compute_overlay_position()
        newax.set_position(box)
    cid = fig.canvas.mpl_connect('resize_event', onresize)  # return value = callback id for `mpl_disconnect`
    _overlay_callbacks[figno].append(cid)

    # # Instead of using a global alpha, we could also customize a colormap like this
    # # (to make alpha vary as a function of the data value):
    # rgb_colors = mpl.colormaps.get("viridis").colors  # or some other base colormap; or make a custom one
    # rgba_colors = [[r, g, b, alpha] for r, g, b in rgb_colors]
    # my_cmap = mpl.colors.ListedColormap(rgba_colors, name="viridis_translucent")
    # # mpl.colormaps.register(my_cmap, force=True)  # no need to register it as we can pass it directly.

    if grid == "quantile":
        # Invert the quantile spacing numerically, to make the positioning match the example images.
        # TODO: implement a custom ScaleTransform for data-interpolated axes? Useful both here and in `hdrplot`.
        n_interp = 10001
        raw_zi = normal_grid(n_interp, kind=grid, eps=eps)  # data value
        linear_zi = np.linspace(-eps, eps, n_interp)  # where that value is on a display with linear coordinates
        def to_linear_display_coordinate(zi):
            """raw value of z_i -> display position on a linear axis with interval [-eps, eps]"""
            return np.interp(zi, xp=raw_zi, fp=linear_zi, left=np.nan, right=np.nan)  # nan = don't plot
        linear_z1 = to_linear_display_coordinate(mean[:, 0])
        linear_z2 = to_linear_display_coordinate(mean[:, 1])
    else:  # grid == "linear":
        linear_z1 = mean[:, 0]
        linear_z2 = mean[:, 1]

    # https://stackoverflow.com/questions/14777066/matplotlib-discrete-colorbar
    cmap = mpl.colormaps.get("viridis")  # or just `mpl.cm.viridis`
    minlabel = np.min(labels)
    maxlabel = np.max(labels)
    color_bounds = np.arange(minlabel, (maxlabel + 1) + 1)
    color_norm = mpl.colors.BoundaryNorm(color_bounds, cmap.N)

    newax.scatter(linear_z1, linear_z2, c=labels, norm=color_norm, alpha=alpha)
    # newax.scatter(linear_z1, linear_z2, c=labels, cmap=my_cmap)
    # newax.patch.set_alpha(0.25)  # patch = Axes background
    newax.set_xlim(-eps, eps)
    newax.set_ylim(-eps, eps)

    # The alpha value of the scatter points messes up the colorbar (making the entries translucent),
    # so we need to customize the colorbar (instead of using the `scatter` return value as the mappable).
    #
    # # One way is to plot an invisible copy of the label values, and base the colorbar on that:
    # # https://stackoverflow.com/questions/16595138/standalone-colorbar-matplotlib
    # fakeax = fig.add_axes([0.0, 0.0, 0.0, 0.0])
    # fakeax.set_visible(False)
    # fakedata = np.array([[minlabel, maxlabel]])
    # fakeplot = fakeax.imshow(fakedata, norm=color_norm)  # cmap=... if needed
    # cb = fig.colorbar(fakeplot, ax=ax)
    #
    # Another way is to supply `norm` and optionally `cmap` (see docstring of `mpl.colorbar.Colorbar`):
    cb = fig.colorbar(None, ax=ax, norm=color_norm,  # cmap=... if needed
                      ticks=color_bounds + 0.5, format="%d")
    _overlay_colorbars[figno].append(cb)

    # Widen the figure to accommodate for the colorbar (at the end, to force a resize)
    fig.set_figwidth(fig.get_figheight() * 1.2)
    onresize(None)  # force-update overlay position once, even if no resizing took place

    plt.draw()
    plotmagic.pause(0.1)  # force redraw
