"""Plotters for the VAE example."""

__all__ = ["plot_test_sample_image",
           "plot_elbo",
           "plot_latent_image",
           "overlay_datapoints", "remove_overlay"]

from collections import defaultdict
import math
import typing

from unpythonic import timer
from unpythonic.env import env

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_probability as tfp

import sklearn.manifold  # for visualizing latent spaces of dimension > 2
import sklearn.preprocessing

from extrafeathers import plotmagic

from .cvae import CVAE

def plot_test_sample_image(test_sample: tf.Tensor, *,
                           model: typing.Optional[CVAE] = None,
                           epoch: typing.Optional[int] = None,
                           figno: int = 1,
                           cols: typing.Optional[int] = None,
                           zoom: float = 1.0) -> None:
    """Plot image of test sample and the corresponding prediction (by feeding the sample through the CVAE).

    `test_sample`: tensor of size `[n, 28, 28, 1]`.
    `model`: `CVAE` instance, or `None` to use the default instance.
    `epoch`: if specified, included in the figure title.
    `figno`: matplotlib figure number.
    `cols`: number of columns in plot; `None` means `cols = floor(sqrt(n))`.
    `zoom`: figure size tuning factor. The default is fine for `cols` = 4 ... 10.
    """
    if model is None:
        from . import main
        model = main.model
    assert isinstance(model, CVAE)

    batch_size, n_pixels_y, n_pixels_x, n_channels = tf.shape(test_sample).numpy()
    assert n_channels == 1, f"This function currently assumes grayscale images, got {n_channels} channels"

    mean, logvar = model.encoder(test_sample, training=False)
    ignored_eps, z = model.reparameterize(mean, logvar)
    predictions = model.sample(z, training=False)

    # If `cols` is not given, auto-pick the most appropriate square layout.
    # For illustration, consider for example:
    #   n = 15 → rows=4, cols=4, one empty slot on last row
    #   n = 16 → rows=4, cols=4, filled exactly
    #   n = 17 → rows=5, cols=4, only one image on last row
    if cols is None:
        cols = math.floor(math.sqrt(batch_size))
    rows = math.ceil(batch_size / cols)

    image_width = (2 * cols + 1) * n_pixels_x  # extra empty column at center, as separator
    image_height = rows * n_pixels_y
    image = np.zeros((image_height, image_width))

    for i in range(batch_size):
        x_orig = test_sample[i, :, :, 0]
        x_hat = predictions[i, :, :, 0]
        row, base_col = divmod(i, cols)
        col1 = base_col  # original image (input)
        col2 = base_col + cols + 1  # reconstructed image
        image[row * n_pixels_y: (row + 1) * n_pixels_y,
              col1 * n_pixels_x: (col1 + 1) * n_pixels_x] = x_orig.numpy()
        image[row * n_pixels_y: (row + 1) * n_pixels_y,
              col2 * n_pixels_x: (col2 + 1) * n_pixels_x] = x_hat.numpy()

    fig = plt.figure(figno)
    if not fig.axes:
        plt.subplot(1, 1, 1)  # create Axes
        fig.set_figwidth(zoom * float(2 * cols + 1))
        fig.set_figheight(zoom * float(cols))
    ax = fig.axes[0]
    ax.cla()
    plt.sca(ax)
    fig.tight_layout()  # prevent axes crawling

    ax.imshow(image, cmap="Greys_r", vmin=0.0, vmax=1.0)
    ax.axis("off")
    epoch_str = f"; epoch {epoch}" if epoch is not None else ""
    ax.set_title(f"Test sample (left: input $\\mathbf{{x}}$, right: prediction $\\hat{{\\mathbf{{x}}}}$){epoch_str}")

    fig.tight_layout()
    plt.draw()
    plotmagic.pause(0.1)  # force redraw


_twin = None
def plot_elbo(epochs, train_elbos, test_elbos, *,
              epoch: typing.Optional[int] = None,
              lr_epochs: typing.Optional[typing.List[float]],
              lrs: typing.Optional[typing.List[float]],
              figno: int = 1) -> None:
    """Plot the evidence lower bound for the training and test sets as a function of the epoch number.

    To plot also the learning rate, pass also `lr_epochs` and `lrs`. Here `lr_epochs` is the epoch number,
    as a float, corresponding to each entry in `lrs`. It is ok for `lr_epochs` to be non-integers; this is
    particularly useful with learning rate schedules that update the learning rate during an epoch.
    """
    fig = plt.figure(figno)
    if not fig.axes:
        plt.subplot(1, 1, 1)  # create Axes
        fig.set_figwidth(6)
        fig.set_figheight(4)
    ax = fig.axes[0]

    global _twin
    if _twin is None:
        _twin = fig.axes[0].twinx()
    twin = _twin

    ax.cla()
    plt.sca(ax)
    fig.tight_layout()  # <-- important to do this also here to prevent axes crawling

    p1, = ax.plot(epochs, train_elbos, "C0", label="train")
    p2, = ax.plot(epochs, test_elbos, "C1", label="test")

    # Zoom to top 80% of data mass (but keep the test elbos visible, if this would hide them)
    q = np.quantile(train_elbos, 0.2)
    if epoch >= 10:
        # Typically, the initial steep climb is over in ~10 epochs.
        ymin = min(q, train_elbos[9], test_elbos[9])
    else:
        ymin = q
    datamax = max(np.max(train_elbos), np.max(test_elbos))

    # When training with fp16, the ELBO loss often starts as NaN.
    if np.isnan(ymin) and np.isnan(datamax):
        ymin = 0.0
        datamax = 0.0
    elif np.isnan(datamax):
        datamax = max(ymin, 0.0)
    elif np.isnan(ymin):
        ymin = min(datamax, 0.0)

    ax.set_ylim(ymin, datamax)

    ax.xaxis.grid(visible=True, which="both")
    ax.yaxis.grid(visible=True, which="both")
    # https://stackoverflow.com/questions/30914462/how-to-force-integer-tick-labels
    # https://matplotlib.org/stable/api/ticker_api.html#matplotlib.ticker.MaxNLocator
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    ax.set_xlabel("epoch")
    ax.set_ylabel("ELBO")

    plot_also_learning_rates = (lr_epochs is not None and lrs is not None)
    if plot_also_learning_rates:
        # https://matplotlib.org/stable/gallery/spines/multiple_yaxis_with_spines.html#sphx-glr-gallery-spines-multiple-yaxis-with-spines-py
        p3, = twin.plot(lr_epochs, lrs, "C2", label="LR")
        twin.yaxis.label.set_color(p3.get_color())
        twin.tick_params(axis='y', colors=p3.get_color())

    # ax.yaxis.label.set_color(p1.get_color())
    # ax.tick_params(axis='y', colors=p1.get_color())

    if plot_also_learning_rates:
        ax.legend(loc="best", handles=[p1, p2, p3])
    else:
        ax.legend(loc="best", handles=[p1, p2])

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

    (This works for any square images encoded into a 2-dimensional latent space.)

    `n`, `grid`, `eps`: passed to `normal_grid` (`grid` is the `kind`)

                        A quantile grid is linear in cumulative probability according to the
                        latent prior. However, using the prior is subtly wrong, and the marginal
                        posterior of z should be used instead; see Lin et al.

    `model`: `CVAE` instance, or `None` to use the default instance.
    `digit_size`: width/height of each digit image (square-shaped), in pixels.
                  Must match what the model was trained for.
    `epoch`: if specified, included in the figure title.
    `figno`: matplotlib figure number.

    The return value is an `unpythonic.env.env` that can be passed to
    `overlay_datapoints` as the `figdata` parameter to add a translucent
    dataset overlay onto the plot.
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
            x_decoded = model.sample(z, training=False)
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
    ax.imshow(image, origin="lower", cmap="Greys_r", vmin=0.0, vmax=1.0)
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

    # We're making a new overlay; clean up old stuff added to figure `figno` by `overlay_datapoints`.
    for cid in _overlay_callbacks.pop(figno, []):
        fig.canvas.mpl_disconnect(cid)
    if len(fig.axes) > 1:
        for ax in fig.axes[1:]:
            ax.remove()
    fig.set_figwidth(fig.get_figheight())

    fig.tight_layout()
    plt.draw()  # force update of extents
    plotmagic.pause(0.1)


_overlay_callbacks = defaultdict(list)
def overlay_datapoints(x: tf.Tensor, labels: tf.Tensor, figdata: env, alpha: float = 0.1) -> None:
    """Overlay the codepoints corresponding to a dataset `x` and `labels` onto the latent space plot.

    (This works for any square images encoded into a 2-dimensional latent space.)

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
    #
    # # mean, logvar = model.encoder(x)  # without batching, this runs out of GPU memory when running on GPU
    # #
    # # The batch size here is just a processing convenience, so ideally (to run as fast as possible) we
    # # should make it as large as the GPU VRAM can accommodate (accounting for any extra VRAM required by
    # # temporary results during encoding). We know at least that a size of 64 fits, but 60k doesn't. So
    # # let's just choose something in between.
    # def encode_batched(inputs, batch_size=1024):
    #     batches = tf.data.Dataset.from_tensor_slices(inputs).batch(batch_size)
    #     means = []
    #     logvars = []
    #     for x in batches:
    #         mean, logvar = model.encoder(x)
    #         means.append(mean)
    #         logvars.append(logvar)
    #     mean = tf.concat(means, axis=0)
    #     logvar = tf.concat(logvars, axis=0)
    #     return mean, logvar
    # mean, logvar = encode_batched(x)
    #
    # Since the encoder part is a `Model`, we can just:
    #   https://keras.io/getting_started/faq/#whats-the-difference-between-model-methods-predict-and-call
    mean, logvar = model.encoder.predict(x, batch_size=1024)

    # --------------------------------------------------------------------------------
    # We need some gymnastics to plot on top of an imshow image; it's easiest to
    # overlay a new Axes with a transparent background.
    #
    # https://stackoverflow.com/questions/16829436/overlay-matplotlib-imshow-with-line-plots-that-are-arranged-in-a-grid

    # Remove old overlay, if any.
    fig = plt.figure(figno)
    remove_overlay(figno)
    if not fig.axes:
        raise ValueError(f"Figure {figno} has no existing Axes; nothing to overlay on.")
    ax = fig.axes[0]  # the Axes we're overlaying the dataset on
    # axs = fig.axes  # list of all Axes objects in this Figure

    # print([int(x) for x in fig.axes[0].get_xlim()])

    # Compute the desired position for the overlay, in pixels of the imshow image.
    image_width = digit_size * n

    # # In this variant, we use the centers of images at two opposite corners of the sheet.
    # # The data area ends at the center of the edgemost images.
    # # Doesn't look good, the dataset overlay is cut off before the image area ends.
    # xmin = digit_size / 2
    # xmax = image_width - (digit_size / 2)
    #
    # In this variant, we use the whole image area for the overlay - looks much better.
    xmin = 0
    xmax = image_width  # endpoint, so after the last pixel.

    ymin = xmin
    ymax = xmax
    xy0 = [xmin, ymin]
    xy1 = [xmax, ymax]

    # For the second variant, we now adjust the ticks of the parent plot to match. The result is that now
    # the centermost image corresponds to its center point; and the cornermost images correspond to their
    # outermost corner points.
    startx = 0
    endx = image_width - 1
    zz = normal_grid(n, kind=grid, eps=eps)
    grid_x = zz
    grid_y = zz
    tick_positions_x = np.array(startx + np.linspace(0, 1, len(grid_x)) * (endx - startx), dtype=int)
    tick_positions_y = tick_positions_x
    ax.set_xticks(tick_positions_x, [f"{x:0.3g}" for x in grid_x], rotation="vertical")
    ax.set_yticks(tick_positions_y, [f"{y:0.3g}" for y in grid_y])

    # --------------------------------------------------------------------------------
    # Create the overlay, and keep it positioned when the window size changes.

    def data_to_fig(xy):
        """Convert Matplotlib data coordinates (of current axis) to figure coordinates."""
        # https://matplotlib.org/stable/tutorials/advanced/transforms_tutorial.html
        xy_ax = ax.transLimits.transform(xy)  # data coordinates -> axes coordinates
        xy_pixels = ax.transAxes.transform(xy_ax)  # axes -> display
        xy_fig = fig.transFigure.inverted().transform(xy_pixels)  # display -> figure
        # print(f"data: {xy}")
        # print(f"ax:   {xy_ax}")
        # print(f"disp: {xy_pixels}")
        # print(f"fig:  {xy_fig}")
        return xy_fig

    def compute_overlay_position_in_figure_coordinates():
        x0, y0 = data_to_fig(xy0)
        x1, y1 = data_to_fig(xy1)
        box = [x0, y0, (x1 - x0), (y1 - y0)]
        return box

    # Set up the new Axes, no background (`set_axis_off`).
    box = compute_overlay_position_in_figure_coordinates()
    newax = fig.add_axes(box, label="<custom overlay>")
    newax.set_axis_off()

    # https://matplotlib.org/stable/users/explain/event_handling.html
    def onresize(event):
        fig.tight_layout()
        plt.draw()
        plotmagic.pause(0.001)
        box = compute_overlay_position_in_figure_coordinates()
        newax.set_position(box)

        # Update scatter plot marker sizes.
        # The data needed here is computed further below when we set up the plot.
        #
        # Matplotlib `scatter` marker sizes `s` are given in units of points². This is defined so that
        # if the marker shape is square, `s` is its area. A circular marker is the inscribed circle,
        # which has area  (π / 4) s = π r²  so  s = 4 r² = (2 r)²  or in other words, `s` is the square
        # of the diameter.
        # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html
        # https://stackoverflow.com/questions/14827650/pyplot-scatter-plot-marker-size
        #
        trans = newax.transData.transform  # linear (`newax`) data coordinates -> display coordinates (pixels)
        # Here `mradius_pixels[:, 0]` is only for debug, to show the length of one linear data unit
        # in display coordinates (pixels).
        mradius_pixels = trans(p2_linear) - trans(p1_linear)
        ppd = 72. / fig.dpi  # relative size of a typographic point
        mradius_points = mradius_pixels[:, 1] * ppd
        s = 4 * mradius_points**2
        scatterplot.set_sizes(s)
    cid = fig.canvas.mpl_connect('resize_event', onresize)  # return value = callback id for `mpl_disconnect`
    _overlay_callbacks[figno].append(cid)

    # # For documentation: if we need to fire a redraw from a `draw_event` handler, it must be delayed.
    # # https://stackoverflow.com/a/48174228
    # # https://github.com/matplotlib/matplotlib/issues/10334
    # def redraw_later(fig):
    #     timer = fig.canvas.new_timer(interval=10)
    #     timer.single_shot = True
    #     timer.add_callback(lambda: fig.canvas.draw_idle())
    #     timer.start()

    # --------------------------------------------------------------------------------
    # Set up a transformation between raw z space (as displayed on the ticks of the parent plot)
    # and linear (`newax`) data coordinates, and use it to position the overlay datapoints.

    n_interp = 10001
    zi_linear = np.linspace(-eps, eps, n_interp)  # where zi_raw (below) is in linear data coordinates
    if grid == "quantile":
        # Invert the quantile spacing numerically.
        # TODO: implement a custom ScaleTransform for data-interpolated axes? Useful both here and in `hdrplot`.
        zi_raw = normal_grid(n_interp, kind=grid, eps=eps)
        def linear_z(zi):  # raw z_i to linear (`newax`) data coordinates on interval [-eps, eps]
            # We pad with NaN to disable plotting of any points outside the overlay area.
            return np.interp(zi, xp=zi_raw, fp=zi_linear, left=np.nan, right=np.nan)

        z1_linear = linear_z(mean[:, 0])
        z2_linear = linear_z(mean[:, 1])

        # Jacobian of the transformation, at interpolation interval midpoints, as function of raw z.
        dzilinear_dziraw = (zi_linear[1:] - zi_linear[:-1]) / (zi_raw[1:] - zi_raw[:-1])
        zzraw_for_jacobian = (zi_raw[1:] + zi_raw[:-1]) / 2  # raw z where we have jacobian data
        def display_jacobian(zi):  # d[z_linear]/d[z_raw] as function of z_raw
            return np.interp(zi, xp=zzraw_for_jacobian, fp=dzilinear_dziraw)
    else:  # grid == "linear":
        z1_linear = mean[:, 0]
        z2_linear = mean[:, 1]
        def display_jacobian(zi):
            return np.ones_like(zi)

    # --------------------------------------------------------------------------------
    # Show the standard deviation of the approximate posterior qϕ(z|x) (i.e. encoding model)
    # in the marker size.
    #
    # TODO: This doesn't yet work correctly upon zooming into a region.

    # By definition, the log-variance is  logvar := log(σ²),  so
    sigma = np.exp(logvar / 2)

    # In "quantile" mode, parts of the data space are magnified, depending on the position of the data point.
    # So the mean of the approximate posterior also affects the visual marker size.
    #
    # Markers are equal-aspect (circular), so we need to scalarize the 2D μ and σ to use them to control
    # the marker size. We use the euclidean length of the vector just for simplicity.
    def euclidean_length(z):
        return np.sqrt(np.sum(z**2, axis=1))
    mdistance_raw = euclidean_length(mean)  # ‖μ‖ in raw z space
    mradius_raw = euclidean_length(sigma)  # ‖σ‖ in raw z space (could also visualize e.g. ‖3σ‖)

    # Given the marker radius in raw z space, compute the marker radius in linear (`newax`) data space.
    # Some standard deviations are very small (1e-6), so we force a minimum marker size
    # to make all data points visible.
    # Also, at the start of the training, with the initial random weights in the network,
    # some standard derivations may be excessively large.
    tick_interval_linear = (2 * eps) / (n - 1)  # whole axis = 2 * eps, with n ticks
    min_mradius_linear = tick_interval_linear / 32
    max_mradius_linear = (2 * (2 * eps)**2)**0.5  # covering the whole data area from corner to opposite corner
    assert min_mradius_linear <= max_mradius_linear
    mradius_linear = np.clip(mradius_raw * display_jacobian(mdistance_raw),
                             min_mradius_linear,
                             max_mradius_linear)

    # # DEBUG
    # print(np.min(mdistance_raw), np.max(mdistance_raw))
    # print(np.min(mradius_raw), np.max(mradius_raw))
    # print(np.min(jacobian(mdistance_raw)), np.max(jacobian(mdistance_raw)))
    # print(np.min(mradius_raw / jacobian(mdistance_raw)), np.max(mradius_raw / jacobian(mdistance_raw)))
    # print(np.min(mradius_linear), np.max(mradius_linear))

    # As the final step, we need to convert from `newax` data units into typographic points; see e.g.
    # https://stackoverflow.com/a/48174228
    ONE = np.ones_like(mradius_linear)
    ZERO = np.zeros_like(mradius_linear)
    p2_linear = np.column_stack((ONE, mradius_linear))
    p1_linear = np.column_stack((ZERO, ZERO))
    # The rest is done in `onresize`, as the marker sizes must be updated when the window size changes.

    # --------------------------------------------------------------------------------
    # Set up colors, make the actual scatterplot, add a colorbar to show data labels.

    # # Instead of using a global alpha, we could also customize a colormap like this
    # # (to make alpha vary as a function of the data value):
    # rgb_colors = mpl.colormaps.get("viridis").colors  # or some other base colormap; or make a custom one
    # rgba_colors = [[r, g, b, alpha] for r, g, b in rgb_colors]
    # my_cmap = mpl.colors.ListedColormap(rgba_colors, name="viridis_translucent")
    # # mpl.colormaps.register(my_cmap, force=True)  # no need to register, we can pass it directly as `cmap`.

    # Make the colorbar discrete (since it represents data labels)
    # https://stackoverflow.com/questions/14777066/matplotlib-discrete-colorbar
    cmap = mpl.colormaps.get("viridis")  # or just `mpl.cm.viridis`
    minlabel = np.min(labels)
    maxlabel = np.max(labels)
    # Labels 0...9 need an upper bound of 10 to have a region for the "9" (in the BoundaryNorm,
    # the region 9...10 maps to 9). The other +1 is for one-past-end.
    color_bounds = np.arange(minlabel, (maxlabel + 1) + 1)
    color_norm = mpl.colors.BoundaryNorm(color_bounds, cmap.N)

    # Make the actual scatter plot
    scatterplot = newax.scatter(z1_linear, z2_linear,
                                s=np.ones_like(z1_linear),
                                c=labels, norm=color_norm,
                                alpha=alpha)
    # newax.scatter(z1_linear, z2_linear, c=labels, cmap=my_cmap)
    # newax.patch.set_alpha(0.25)  # patch = Axes background
    newax.set_xlim(-eps, eps)
    newax.set_ylim(-eps, eps)

    # The alpha value of the scatterplot points makes also the corresponding colorbar entries translucent,
    # so we need to customize the colorbar (instead of using the `scatter` return value as the mappable).
    #
    # # The old way is to plot an invisible copy of the label values, and base the colorbar on that:
    # # https://stackoverflow.com/questions/16595138/standalone-colorbar-matplotlib
    # fakeax = fig.add_axes([0.0, 0.0, 0.0, 0.0])
    # fakeax.set_visible(False)
    # fakedata = np.array([[minlabel, maxlabel]])
    # fakeplot = fakeax.imshow(fakedata, norm=color_norm)  # cmap=... if needed
    # cb = fig.colorbar(fakeplot, ax=ax)
    #
    # Another, better way is to supply `norm` and optionally `cmap` (see docstring of `mpl.colorbar.Colorbar`).
    # We also shift the ticks to the midpoint of each discrete region of the colorbar, to ease readability.
    cb = fig.colorbar(None, ax=ax, norm=color_norm,  # cmap=... if needed    # noqa: F841
                      ticks=color_bounds + 0.5, format="%d")

    # --------------------------------------------------------------------------------
    # Final adjustments.

    # Widen the figure to accommodate for the colorbar (at the end, to force a resize)
    fig.set_figwidth(fig.get_figheight() * 1.2)
    onresize(None)  # force-update overlay position once, even if no resizing took place

    plt.draw()
    plotmagic.pause(0.1)  # force redraw


# TODO: make a version compatible with `plot_latent_image` so that we can `overlay_datapoints` on this (to see the variances).
def plot_manifold(x: tf.Tensor,
                  labels: tf.Tensor,
                  alpha: float = 0.1,
                  model: typing.Optional[CVAE] = None,
                  epoch: typing.Optional[int] = None,
                  figno: int = 1) -> None:
    """Plot learned manifold in 3+ dimensional latent space, via dimension reduction to 2D.

    `x`: tensor of shape [N, 28, 28, 1], containing training and/or test images.
    `labels`: tensor of shape [N], integer labels corresponding to the data points.
    `alpha`: opacity of the scatterplot points.
    `model`: `CVAE` instance, or `None` to use the default instance.
    `epoch`: if specified, included in the figure title.
    `figno`: matplotlib figure number.

    This makes a scatterplot with the integer labels, applying the following dimension
    reduction algorithms from `scikit-learn`, each in its own subplot:

      - LTSA: Local Tangent Space Alignment
      - MLLE: Modified Locally Linear Embedding
      - MDS: MultiDimensional Scaling
      - SE: Spectral Embedding a.k.a. laplacian eigenmap; equivalent to diffusion map.

    These particular methods were chosen because each produces different-looking results
    for some toy datasets, and they are all rather fast (compared to t-SNE). Still,
    dimension reduction in general is slow; e.g. for 4000 datapoints, on the dev machine:
        LTSA: 12.6s, MLLE: 4.2s, MDS: 25.9s, SE: 2.8s

    We suggest something like::

        plot_manifold(test_images[:4000, :, :, :], test_labels[:4000])

    to keep things semi-fast.

    For more information, see:
        https://scikit-learn.org/stable/modules/manifold.html
    """
    if model is None:
        from . import main
        model = main.model
    assert isinstance(model, CVAE)

    # Compute latent representation (now `z` has ≥ 3 dimensions)
    mean, logvar = model.encoder.predict(x, batch_size=1024)
    ignored_eps, z = model.reparameterize(mean, logvar)

    # Reduce dimension, using several different methods. Helpful code examples:
    #   https://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html
    #   https://scikit-learn.org/stable/auto_examples/manifold/plot_manifold_sphere.html
    #   https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html
    # TODO: tune dimension reduction parameters (esp. the number of neighbors)
    k = 100  # number of neighbors
    d = 2    # target dimension
    ltsa = sklearn.manifold.LocallyLinearEmbedding(n_components=d,
                                                   n_neighbors=k,
                                                   method="ltsa")  # Local Tangent Space Alignment
    mlle = sklearn.manifold.LocallyLinearEmbedding(n_components=d,
                                                   n_neighbors=k,
                                                   method="modified")  # Modified Locally Linear Embedding
    mds = sklearn.manifold.MDS(n_components=d,
                               n_init=1,
                               max_iter=120,
                               n_jobs=2,
                               normalized_stress="auto")  # MultiDimensional Scaling  # TODO: tune params?
    spectral = sklearn.manifold.SpectralEmbedding(n_components=d,
                                                  random_state=0,
                                                  eigen_solver="arpack")  # laplacian eigenmap; equivalent to diffusion map

    transformers = (("LTSA", ltsa), ("MLLE", mlle), ("MDS", mds), ("SE", spectral))
    z = sklearn.preprocessing.MinMaxScaler().fit_transform(z)
    data = []  # dimension-reduced latent data
    for name, transformer in transformers:
        print(f"Reducing dimension with transformer {name}...")
        with timer() as tim:
            data.append((name, transformer.fit_transform(z, labels)))
        print(f"    Done in {tim.dt:0.6g}s.")
    print("Plotting manifold.")

    # Plot the result
    #
    fig = plt.figure(figno)
    if not fig.axes:
        plt.subplot(2, 2, 1)  # create Axes
        fig.set_figwidth(10)
        fig.set_figheight(10)
    fig.tight_layout()  # prevent axes crawling

    for sub, (name, zhat) in enumerate(data):
        print(f"Plotting results of transformer {name}...")
        with timer() as tim:
            ax = plt.subplot(2, 2, 1 + sub)
            ax.cla()
            plt.sca(ax)
            for digit in range(min(labels), max(labels) + 1):
                ax.scatter(*zhat[labels == digit].T,
                           marker=f"${digit}$",
                           s=60,
                           color=plt.cm.Dark2(digit),
                           alpha=0.425,
                           zorder=2)
            ax.axis("off")
            ax.set_title(name)
        print(f"    Done in {tim.dt:0.6g}s.")
    print("Manifold plotting done.")

    epoch_str = f"; epoch {epoch}" if epoch is not None else ""
    plt.suptitle(f"Latent manifold{epoch_str}")

    fig.tight_layout()
    plt.draw()
    plotmagic.pause(0.1)  # force redraw
