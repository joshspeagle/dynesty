#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A set of built-in plotting functions to help visualize ``dynesty`` nested
sampling :class:`~dynesty.results.Results`.

"""

from __future__ import (print_function, division)
from six.moves import range

import logging
import types
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator, NullLocator
from matplotlib.colors import LinearSegmentedColormap, colorConverter
from matplotlib.ticker import ScalarFormatter
from scipy import spatial
from scipy.ndimage import gaussian_filter as norm_kde
from scipy.stats import gaussian_kde
import warnings
from .utils import resample_equal, unitcheck
from .utils import quantile as _quantile

try:
    str_type = types.StringTypes
    float_type = types.FloatType
    int_type = types.IntType
except:
    str_type = str
    float_type = float
    int_type = int

__all__ = ["runplot", "traceplot", "cornerpoints", "cornerplot",
           "boundplot", "cornerbound", "_hist2d"]


def runplot(results, span=None, logplot=False, kde=True, nkde=1000,
            color='blue', plot_kwargs=None, label_kwargs=None, lnz_error=True,
            lnz_truth=None, truth_color='red', truth_kwargs=None,
            max_x_ticks=8, max_y_ticks=3, use_math_text=True,
            mark_final_live=True, fig=None):
    """
    Plot live points, ln(likelihood), ln(weight), and ln(evidence)
    as a function of ln(prior volume).

    Parameters
    ----------
    results : :class:`~dynesty.results.Results` instance
        A :class:`~dynesty.results.Results` instance from a nested
        sampling run.

    span : iterable with shape (4,), optional
        A list where each element is either a length-2 tuple containing
        lower and upper bounds *or* a float from `(0., 1.]` giving the
        fraction below the maximum. If a fraction is provided,
        the bounds are chosen to be equal-tailed. An example would be::

            span = [(0., 10.), 0.001, 0.2, (5., 6.)]

        Default is `(0., 1.05 * max(data))` for each element.

    logplot : bool, optional
        Whether to plot the evidence on a log scale. Default is `False`.

    kde : bool, optional
        Whether to use kernel density estimation to estimate and plot
        the PDF of the importance weights as a function of log-volume
        (as opposed to the importance weights themselves). Default is
        `True`.

    nkde : int, optional
        The number of grid points used when plotting the kernel density
        estimate. Default is `1000`.

    color : str or iterable with shape (4,), optional
        A `~matplotlib`-style color (either a single color or a different
        value for each subplot) used when plotting the lines in each subplot.
        Default is `'blue'`.

    plot_kwargs : dict, optional
        Extra keyword arguments that will be passed to `plot`.

    label_kwargs : dict, optional
        Extra keyword arguments that will be sent to the
        `~matplotlib.axes.Axes.set_xlabel` and
        `~matplotlib.axes.Axes.set_ylabel` methods.

    lnz_error : bool, optional
        Whether to plot the 1, 2, and 3-sigma approximate error bars
        derived from the ln(evidence) error approximation over the course
        of the run. Default is `True`.

    lnz_truth : float, optional
        A reference value for the evidence that will be overplotted on the
        evidence subplot if provided.

    truth_color : str or iterable with shape (ndim,), optional
        A `~matplotlib`-style color used when plotting :data:`lnz_truth`.
        Default is `'red'`.

    truth_kwargs : dict, optional
        Extra keyword arguments that will be used for plotting
        :data:`lnz_truth`.

    max_x_ticks : int, optional
        Maximum number of ticks allowed for the x axis. Default is `8`.

    max_y_ticks : int, optional
        Maximum number of ticks allowed for the y axis. Default is `4`.

    use_math_text : bool, optional
        Whether the axis tick labels for very large/small exponents should be
        displayed as powers of 10 rather than using `e`. Default is `False`.

    mark_final_live : bool, optional
        Whether to indicate the final addition of recycled live points
        (if they were added to the resulting samples) using
        a dashed vertical line. Default is `True`.

    fig : (`~matplotlib.figure.Figure`, `~matplotlib.axes.Axes`), optional
        If provided, overplot the run onto the provided figure.
        Otherwise, by default an internal figure is generated.

    Returns
    -------
    runplot : (`~matplotlib.figure.Figure`, `~matplotlib.axes.Axes`)
        Output summary plot.

    """

    # Initialize values.
    if label_kwargs is None:
        label_kwargs = dict()
    if plot_kwargs is None:
        plot_kwargs = dict()
    if truth_kwargs is None:
        truth_kwargs = dict()

    # Set defaults.
    plot_kwargs['linewidth'] = plot_kwargs.get('linewidth', 5)
    plot_kwargs['alpha'] = plot_kwargs.get('alpha', 0.7)
    truth_kwargs['linestyle'] = truth_kwargs.get('linestyle', 'solid')
    truth_kwargs['linewidth'] = truth_kwargs.get('linewidth', 3)

    # Extract results.
    niter = results['niter']  # number of iterations
    logvol = results['logvol']  # ln(prior volume)
    logl = results['logl'] - max(results['logl'])  # ln(normalized likelihood)
    logwt = results['logwt'] - results['logz'][-1]  # ln(importance weight)
    logz = results['logz']  # ln(evidence)
    logzerr = results['logzerr']  # error in ln(evidence)
    logzerr[~np.isfinite(logzerr)] = 0.
    nsamps = len(logwt)  # number of samples

    # Check whether the run was "static" or "dynamic".
    try:
        nlive = results['samples_n']
        mark_final_live = False
    except:
        nlive = np.ones(niter) * results['nlive']
        if nsamps - niter == results['nlive']:
            nlive_final = np.arange(1, results['nlive'] + 1)[::-1]
            nlive = np.append(nlive, nlive_final)

    # Check if the final set of live points were added to the results.
    if mark_final_live:
        if nsamps - niter == results['nlive']:
            live_idx = niter
        else:
            warnings.warn("The number of iterations and samples differ "
                          "by an amount that isn't the number of final "
                          "live points. `mark_final_live` has been disabled.")
            mark_final_live = False

    # Determine plotting bounds for each subplot.
    data = [nlive, np.exp(logl), np.exp(logwt), np.exp(logz)]
    if kde:
        # Derive kernel density estimate.
        wt_kde = gaussian_kde(resample_equal(-logvol, data[2]))  # KDE
        logvol_new = np.linspace(logvol[0], logvol[-1], nkde)  # resample
        data[2] = wt_kde.pdf(-logvol_new)  # evaluate KDE PDF
    if span is None:
        span = [(0., 1.05 * max(d)) for d in data]
        no_span = True
    else:
        no_span = False
    span = list(span)
    if len(span) != 4:
        raise ValueError("More bounds provided in `span` than subplots!")
    for i, _ in enumerate(span):
        try:
            ymin, ymax = span[i]
        except:
            span[i] = (max(data[i]) * span[i], max(data[i]))
    if lnz_error and no_span:
        if logplot:
            zspan = (np.exp(logz[-1] - 1.3 * 3. * logzerr[-1]),
                     np.exp(logz[-1] + 1.3 * 3. * logzerr[-1]))
        else:
            zspan = (0., 1.05 * np.exp(logz[-1] + 3. * logzerr[-1]))
        span[3] = zspan

    # Setting up default plot layout.
    if fig is None:
        fig, axes = pl.subplots(4, 1, figsize=(16, 16))
        xspan = [(0., -min(logvol)) for _ax in axes]
        yspan = span
    else:
        fig, axes = fig
        try:
            axes.reshape(4, 1)
        except:
            raise ValueError("Provided axes do not match the required shape "
                             "for plotting samples.")
        # If figure is provided, keep previous bounds if they were larger.
        xspan = [ax.get_xlim() for ax in axes]
        yspan = [ax.get_ylim() for ax in axes]
        # One exception: if the bounds are the plotting default `(0., 1.)`,
        # overwrite them.
        xspan = [t if t != (0., 1.) else (None, None) for t in xspan]
        yspan = [t if t != (0., 1.) else (None, None) for t in yspan]

    # Set up bounds for plotting.
    for i in range(4):
        if xspan[i][0] is None:
            xmin = None
        else:
            xmin = min(0., xspan[i][0])
        if xspan[i][1] is None:
            xmax = -min(logvol)
        else:
            xmax = max(-min(logvol), xspan[i][1])
        if yspan[i][0] is None:
            ymin = None
        else:
            ymin = min(span[i][0], yspan[i][0])
        if yspan[i][1] is None:
            ymax = span[i][1]
        else:
            ymax = max(span[i][1], yspan[i][1])
        axes[i].set_xlim([xmin, xmax])
        axes[i].set_ylim([ymin, ymax])

    # Plotting.
    labels = ['Live Points', 'Likelihood\n(normalized)',
              'Importance\nWeight', 'Evidence']
    if kde:
        labels[2] += ' PDF'

    for i, d in enumerate(data):

        # Establish axes.
        ax = axes[i]
        # Set color(s)/colormap(s).
        if isinstance(color, str_type):
            c = color
        else:
            c = color[i]
        # Setup axes.
        if max_x_ticks == 0:
            ax.xaxis.set_major_locator(NullLocator())
        else:
            ax.xaxis.set_major_locator(MaxNLocator(max_x_ticks))
        if max_y_ticks == 0:
            ax.yaxis.set_major_locator(NullLocator())
        else:
            ax.yaxis.set_major_locator(MaxNLocator(max_y_ticks))
        # Label axes.
        sf = ScalarFormatter(useMathText=use_math_text)
        ax.yaxis.set_major_formatter(sf)
        ax.set_xlabel(r"$-\ln X$", **label_kwargs)
        ax.set_ylabel(labels[i], **label_kwargs)
        # Plot run.
        if logplot and i == 3:
            ax.semilogy(-logvol, d, color=c, **plot_kwargs)
            yspan = [ax.get_ylim() for _ax in axes]
        elif kde and i == 2:
            ax.plot(-logvol_new, d, color=c, **plot_kwargs)
        else:
            ax.plot(-logvol, d, color=c, **plot_kwargs)
        if i == 3 and lnz_error:
            [ax.fill_between(-logvol, np.exp(logz + s*logzerr),
                             np.exp(logz - s*logzerr), color=c, alpha=0.2)
             for s in range(1, 4)]
        # Mark addition of final live points.
        if mark_final_live:
            ax.axvline(-logvol[live_idx], color=c, ls="dashed", lw=2,
                       **plot_kwargs)
            if i == 0:
                ax.axhline(live_idx, color=c, ls="dashed", lw=2,
                           **plot_kwargs)
        # Add truth value(s).
        if i == 3 and lnz_truth is not None:
            ax.axhline(np.exp(lnz_truth), color=truth_color, **truth_kwargs)

    return fig, axes


def traceplot(results, span=None, quantiles=[0.025, 0.5, 0.975],
              smooth=0.02, thin=1, dims=None,
              post_color='blue', post_kwargs=None, kde=True, nkde=1000,
              trace_cmap='plasma', trace_color=None, trace_kwargs=None,
              connect=False, connect_highlight=10, connect_color='red',
              connect_kwargs=None, max_n_ticks=5, use_math_text=False,
              labels=None, label_kwargs=None,
              show_titles=False, title_fmt=".2f", title_kwargs=None,
              truths=None, truth_color='red', truth_kwargs=None,
              verbose=False, fig=None):
    """
    Plot traces and marginalized posteriors for each parameter.

    Parameters
    ----------
    results : :class:`~dynesty.results.Results` instance
        A :class:`~dynesty.results.Results` instance from a nested
        sampling run. **Compatible with results derived from**
        `nestle <http://kylebarbary.com/nestle/>`_.

    span : iterable with shape (ndim,), optional
        A list where each element is either a length-2 tuple containing
        lower and upper bounds or a float from `(0., 1.]` giving the
        fraction of (weighted) samples to include. If a fraction is provided,
        the bounds are chosen to be equal-tailed. An example would be::

            span = [(0., 10.), 0.95, (5., 6.)]

        Default is `0.999999426697` (5-sigma credible interval) for each
        parameter.

    quantiles : iterable, optional
        A list of fractional quantiles to overplot on the 1-D marginalized
        posteriors as vertical dashed lines. Default is `[0.025, 0.5, 0.975]`
        (the 95%/2-sigma credible interval).

    smooth : float or iterable with shape (ndim,), optional
        The standard deviation (either a single value or a different value for
        each subplot) for the Gaussian kernel used to smooth the 1-D
        marginalized posteriors, expressed as a fraction of the span.
        Default is `0.02` (2% smoothing). If an integer is provided instead,
        this will instead default to a simple (weighted) histogram with
        `bins=smooth`.

    thin : int, optional
        Thin the samples so that only each `thin`-th sample is plotted.
        Default is `1` (no thinning).

    dims : iterable of shape (ndim,), optional
        The subset of dimensions that should be plotted. If not provided,
        all dimensions will be shown.

    post_color : str or iterable with shape (ndim,), optional
        A `~matplotlib`-style color (either a single color or a different
        value for each subplot) used when plotting the histograms.
        Default is `'blue'`.

    post_kwargs : dict, optional
        Extra keyword arguments that will be used for plotting the
        marginalized 1-D posteriors.

    kde : bool, optional
        Whether to use kernel density estimation to estimate and plot
        the PDF of the importance weights as a function of log-volume
        (as opposed to the importance weights themselves). Default is
        `True`.

    nkde : int, optional
        The number of grid points used when plotting the kernel density
        estimate. Default is `1000`.

    trace_cmap : str or iterable with shape (ndim,), optional
        A `~matplotlib`-style colormap (either a single colormap or a
        different colormap for each subplot) used when plotting the traces,
        where each point is colored according to its weight. Default is
        `'plasma'`.

    trace_color : str or iterable with shape (ndim,), optional
        A `~matplotlib`-style color (either a single color or a
        different color for each subplot) used when plotting the traces.
        This overrides the `trace_cmap` option by giving all points
        the same color. Default is `None` (not used).

    trace_kwargs : dict, optional
        Extra keyword arguments that will be used for plotting the traces.

    connect : bool, optional
        Whether to draw lines connecting the paths of unique particles.
        Default is `False`.

    connect_highlight : int or iterable, optional
        If `connect=True`, highlights the paths of a specific set of
        particles. If an integer is passed, :data:`connect_highlight`
        random particle paths will be highlighted. If an iterable is passed,
        then the particle paths corresponding to the provided indices
        will be highlighted.

    connect_color : str, optional
        The color of the highlighted particle paths. Default is `'red'`.

    connect_kwargs : dict, optional
        Extra keyword arguments used for plotting particle paths.

    max_n_ticks : int, optional
        Maximum number of ticks allowed. Default is `5`.

    use_math_text : bool, optional
        Whether the axis tick labels for very large/small exponents should be
        displayed as powers of 10 rather than using `e`. Default is `False`.

    labels : iterable with shape (ndim,), optional
        A list of names for each parameter. If not provided, the default name
        used when plotting will follow :math:`x_i` style.

    label_kwargs : dict, optional
        Extra keyword arguments that will be sent to the
        `~matplotlib.axes.Axes.set_xlabel` and
        `~matplotlib.axes.Axes.set_ylabel` methods.

    show_titles : bool, optional
        Whether to display a title above each 1-D marginalized posterior
        showing the 0.5 quantile along with the upper/lower bounds associated
        with the 0.025 and 0.975 (95%/2-sigma credible interval) quantiles.
        Default is `True`.

    title_fmt : str, optional
        The format string for the quantiles provided in the title. Default is
        `'.2f'`.

    title_kwargs : dict, optional
        Extra keyword arguments that will be sent to the
        `~matplotlib.axes.Axes.set_title` command.

    truths : iterable with shape (ndim,), optional
        A list of reference values that will be overplotted on the traces and
        marginalized 1-D posteriors as solid horizontal/vertical lines.
        Individual values can be exempt using `None`. Default is `None`.

    truth_color : str or iterable with shape (ndim,), optional
        A `~matplotlib`-style color (either a single color or a different
        value for each subplot) used when plotting `truths`.
        Default is `'red'`.

    truth_kwargs : dict, optional
        Extra keyword arguments that will be used for plotting the vertical
        and horizontal lines with `truths`.

    verbose : bool, optional
        Whether to print the values of the computed quantiles associated with
        each parameter. Default is `False`.

    fig : (`~matplotlib.figure.Figure`, `~matplotlib.axes.Axes`), optional
        If provided, overplot the traces and marginalized 1-D posteriors
        onto the provided figure. Otherwise, by default an
        internal figure is generated.

    Returns
    -------
    traceplot : (`~matplotlib.figure.Figure`, `~matplotlib.axes.Axes`)
        Output trace plot.

    """

    # Initialize values.
    if title_kwargs is None:
        title_kwargs = dict()
    if label_kwargs is None:
        label_kwargs = dict()
    if trace_kwargs is None:
        trace_kwargs = dict()
    if connect_kwargs is None:
        connect_kwargs = dict()
    if post_kwargs is None:
        post_kwargs = dict()
    if truth_kwargs is None:
        truth_kwargs = dict()

    # Set defaults.
    connect_kwargs['alpha'] = connect_kwargs.get('alpha', 0.7)
    post_kwargs['alpha'] = post_kwargs.get('alpha', 0.6)
    trace_kwargs['s'] = trace_kwargs.get('s', 3)
    trace_kwargs['edgecolor'] = trace_kwargs.get('edgecolor', None)
    trace_kwargs['edgecolors'] = trace_kwargs.get('edgecolors', None)
    truth_kwargs['linestyle'] = truth_kwargs.get('linestyle', 'solid')
    truth_kwargs['linewidth'] = truth_kwargs.get('linewidth', 2)

    # Extract weighted samples.
    samples = results['samples']
    logvol = results['logvol']
    try:
        weights = np.exp(results['logwt'] - results['logz'][-1])
    except:
        weights = results['weights']
    if kde:
        # Derive kernel density estimate.
        wt_kde = gaussian_kde(resample_equal(-logvol, weights))  # KDE
        logvol_grid = np.linspace(logvol[0], logvol[-1], nkde)  # resample
        wt_grid = wt_kde.pdf(-logvol_grid)  # evaluate KDE PDF
        wts = np.interp(-logvol, -logvol_grid, wt_grid)  # interpolate
    else:
        wts = weights

    # Deal with 1D results. A number of extra catches are also here
    # in case users are trying to plot other results besides the `Results`
    # instance generated by `dynesty`.
    samples = np.atleast_1d(samples)
    if len(samples.shape) == 1:
        samples = np.atleast_2d(samples)
    else:
        assert len(samples.shape) == 2, "Samples must be 1- or 2-D."
        samples = samples.T
    assert samples.shape[0] <= samples.shape[1], "There are more " \
                                                 "dimensions than samples!"

    # Slice samples based on provided `dims`.
    if dims is not None:
        samples = samples[dims]
    ndim, nsamps = samples.shape

    # Check weights.
    if weights.ndim != 1:
        raise ValueError("Weights must be 1-D.")
    if nsamps != weights.shape[0]:
        raise ValueError("The number of weights and samples disagree!")

    # Check ln(volume).
    if logvol.ndim != 1:
        raise ValueError("Ln(volume)'s must be 1-D.")
    if nsamps != logvol.shape[0]:
        raise ValueError("The number of ln(volume)'s and samples disagree!")

    # Check sample IDs.
    if connect:
        try:
            samples_id = results['samples_id']
            uid = np.unique(samples_id)
        except:
            raise ValueError("Sample IDs are not defined!")
        try:
            ids = connect_highlight[0]
            ids = connect_highlight
        except:
            ids = np.random.choice(uid, size=connect_highlight, replace=False)

    # Determine plotting bounds for marginalized 1-D posteriors.
    if span is None:
        span = [0.999999426697 for i in range(ndim)]
    span = list(span)
    if len(span) != ndim:
        raise ValueError("Dimension mismatch between samples and span.")
    for i, _ in enumerate(span):
        try:
            xmin, xmax = span[i]
        except:
            q = [0.5 - 0.5 * span[i], 0.5 + 0.5 * span[i]]
            span[i] = _quantile(samples[i], q, weights=weights)

    # Setting up labels.
    if labels is None:
        labels = [r"$x_{"+str(i+1)+"}$" for i in range(ndim)]

    # Setting up smoothing.
    if (isinstance(smooth, int_type) or isinstance(smooth, float_type)):
        smooth = [smooth for i in range(ndim)]

    # Setting up default plot layout.
    if fig is None:
        fig, axes = pl.subplots(ndim, 2, figsize=(12, 3*ndim))
    else:
        fig, axes = fig
        try:
            axes.reshape(ndim, 2)
        except:
            raise ValueError("Provided axes do not match the required shape "
                             "for plotting samples.")

    # Plotting.
    for i, x in enumerate(samples):

        # Plot trace.

        # Establish axes.
        if np.shape(samples)[0] == 1:
            ax = axes[1]
        else:
            ax = axes[i, 0]
        # Set color(s)/colormap(s).
        if trace_color is not None:
            if isinstance(trace_color, str_type):
                color = trace_color
            else:
                color = trace_color[i]
        else:
            color = wts[::thin]
        if isinstance(trace_cmap, str_type):
            cmap = trace_cmap
        else:
            cmap = trace_cmap[i]
        # Setup axes.
        ax.set_xlim([0., -min(logvol)])
        ax.set_ylim([min(x), max(x)])
        if max_n_ticks == 0:
            ax.xaxis.set_major_locator(NullLocator())
            ax.yaxis.set_major_locator(NullLocator())
        else:
            ax.xaxis.set_major_locator(MaxNLocator(max_n_ticks))
            ax.yaxis.set_major_locator(MaxNLocator(max_n_ticks))
        # Label axes.
        sf = ScalarFormatter(useMathText=use_math_text)
        ax.yaxis.set_major_formatter(sf)
        ax.set_xlabel(r"$-\ln X$", **label_kwargs)
        ax.set_ylabel(labels[i], **label_kwargs)
        # Generate scatter plot.
        ax.scatter(-logvol[::thin], x[::thin], c=color, cmap=cmap,
                   **trace_kwargs)
        if connect:
            # Add lines highlighting specific particle paths.
            for j in ids:
                sel = (samples_id[::thin] == j)
                ax.plot(-logvol[::thin][sel], x[::thin][sel],
                        color=connect_color, **connect_kwargs)
        # Add truth value(s).
        if truths is not None and truths[i] is not None:
            try:
                [ax.axhline(t, color=truth_color, **truth_kwargs)
                 for t in truths[i]]
            except:
                ax.axhline(truths[i], color=truth_color, **truth_kwargs)

        # Plot marginalized 1-D posterior.

        # Establish axes.
        if np.shape(samples)[0] == 1:
            ax = axes[0]
        else:
            ax = axes[i, 1]
        # Set color(s).
        if isinstance(post_color, str_type):
            color = post_color
        else:
            color = post_color[i]
        # Setup axes
        ax.set_xlim(span[i])
        if max_n_ticks == 0:
            ax.xaxis.set_major_locator(NullLocator())
            ax.yaxis.set_major_locator(NullLocator())
        else:
            ax.xaxis.set_major_locator(MaxNLocator(max_n_ticks))
            ax.yaxis.set_major_locator(NullLocator())
        # Label axes.
        sf = ScalarFormatter(useMathText=use_math_text)
        ax.xaxis.set_major_formatter(sf)
        ax.set_xlabel(labels[i], **label_kwargs)
        # Generate distribution.
        s = smooth[i]
        if isinstance(s, int_type):
            # If `s` is an integer, plot a weighted histogram with
            # `s` bins within the provided bounds.
            n, b, _ = ax.hist(x, bins=s, weights=weights, color=color,
                              range=np.sort(span[i]), **post_kwargs)
            x0 = np.array(list(zip(b[:-1], b[1:]))).flatten()
            y0 = np.array(list(zip(n, n))).flatten()
        else:
            # If `s` is a float, oversample the data relative to the
            # smoothing filter by a factor of 10, then use a Gaussian
            # filter to smooth the results.
            bins = int(round(10. / s))
            n, b = np.histogram(x, bins=bins, weights=weights,
                                range=np.sort(span[i]))
            n = norm_kde(n, 10.)
            x0 = 0.5 * (b[1:] + b[:-1])
            y0 = n
            ax.fill_between(x0, y0, color=color, **post_kwargs)
        ax.set_ylim([0., max(y0) * 1.05])
        # Plot quantiles.
        if quantiles is not None and len(quantiles) > 0:
            qs = _quantile(x, quantiles, weights=weights)
            for q in qs:
                ax.axvline(q, lw=2, ls="dashed", color=color)
            if verbose:
                print("Quantiles:")
                print(labels[i], [blob for blob in zip(quantiles, qs)])
        # Add truth value(s).
        if truths is not None and truths[i] is not None:
            try:
                [ax.axvline(t, color=truth_color, **truth_kwargs)
                 for t in truths[i]]
            except:
                ax.axvline(truths[i], color=truth_color, **truth_kwargs)
        # Set titles.
        if show_titles:
            title = None
            if title_fmt is not None:
                ql, qm, qh = _quantile(x, [0.025, 0.5, 0.975], weights=weights)
                q_minus, q_plus = qm - ql, qh - qm
                fmt = "{{0:{0}}}".format(title_fmt).format
                title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
                title = title.format(fmt(qm), fmt(q_minus), fmt(q_plus))
                title = "{0} = {1}".format(labels[i], title)
                ax.set_title(title, **title_kwargs)

    return fig, axes


def cornerpoints(results, dims=None, thin=1, span=None,
                 cmap='plasma', color=None,
                 kde=True, nkde=1000, plot_kwargs=None, labels=None,
                 label_kwargs=None, truths=None, truth_color='red',
                 truth_kwargs=None, max_n_ticks=5, use_math_text=False,
                 fig=None):
    """
    Generate a (sub-)corner plot of (weighted) samples.

    Parameters
    ----------
    results : :class:`~dynesty.results.Results` instance
        A :class:`~dynesty.results.Results` instance from a nested
        sampling run. **Compatible with results derived from**
        `nestle <http://kylebarbary.com/nestle/>`_.

    dims : iterable of shape (ndim,), optional
        The subset of dimensions that should be plotted. If not provided,
        all dimensions will be shown.

    thin : int, optional
        Thin the samples so that only each `thin`-th sample is plotted.
        Default is `1` (no thinning).

    span : iterable with shape (ndim,), optional
        A list where each element is either a length-2 tuple containing
        lower and upper bounds or a float from `(0., 1.]` giving the
        fraction of (weighted) samples to include. If a fraction is provided,
        the bounds are chosen to be equal-tailed. An example would be::

            span = [(0., 10.), 0.95, (5., 6.)]

        Default is `1.` for all parameters (no bound).

    cmap : str, optional
        A `~matplotlib`-style colormap used when plotting the points,
        where each point is colored according to its weight. Default is
        `'plasma'`.

    color : str, optional
        A `~matplotlib`-style color used when plotting the points.
        This overrides the `cmap` option by giving all points
        the same color. Default is `None` (not used).

    kde : bool, optional
        Whether to use kernel density estimation to estimate and plot
        the PDF of the importance weights as a function of log-volume
        (as opposed to the importance weights themselves). Default is
        `True`.

    nkde : int, optional
        The number of grid points used when plotting the kernel density
        estimate. Default is `1000`.

    plot_kwargs : dict, optional
        Extra keyword arguments that will be used for plotting the points.

    labels : iterable with shape (ndim,), optional
        A list of names for each parameter. If not provided, the default name
        used when plotting will follow :math:`x_i` style.

    label_kwargs : dict, optional
        Extra keyword arguments that will be sent to the
        `~matplotlib.axes.Axes.set_xlabel` and
        `~matplotlib.axes.Axes.set_ylabel` methods.

    truths : iterable with shape (ndim,), optional
        A list of reference values that will be overplotted on the traces and
        marginalized 1-D posteriors as solid horizontal/vertical lines.
        Individual values can be exempt using `None`. Default is `None`.

    truth_color : str or iterable with shape (ndim,), optional
        A `~matplotlib`-style color (either a single color or a different
        value for each subplot) used when plotting `truths`.
        Default is `'red'`.

    truth_kwargs : dict, optional
        Extra keyword arguments that will be used for plotting the vertical
        and horizontal lines with `truths`.

    max_n_ticks : int, optional
        Maximum number of ticks allowed. Default is `5`.

    use_math_text : bool, optional
        Whether the axis tick labels for very large/small exponents should be
        displayed as powers of 10 rather than using `e`. Default is `False`.

    fig : (`~matplotlib.figure.Figure`, `~matplotlib.axes.Axes`), optional
        If provided, overplot the points onto the provided figure object.
        Otherwise, by default an internal figure is generated.

    Returns
    -------
    cornerpoints : (`~matplotlib.figure.Figure`, `~matplotlib.axes.Axes`)
        Output (sub-)corner plot of (weighted) samples.

    """

    # Initialize values.
    if truth_kwargs is None:
        truth_kwargs = dict()
    if label_kwargs is None:
        label_kwargs = dict()
    if plot_kwargs is None:
        plot_kwargs = dict()

    # Set defaults.
    plot_kwargs['s'] = plot_kwargs.get('s', 1)
    plot_kwargs['edgecolor'] = plot_kwargs.get('edgecolor', None)
    plot_kwargs['edgecolors'] = plot_kwargs.get('edgecolors', None)
    truth_kwargs['linestyle'] = truth_kwargs.get('linestyle', 'solid')
    truth_kwargs['linewidth'] = truth_kwargs.get('linewidth', 2)
    truth_kwargs['alpha'] = truth_kwargs.get('alpha', 0.7)

    # Extract weighted samples.
    samples = results['samples']
    logvol = results['logvol']
    try:
        weights = np.exp(results['logwt'] - results['logz'][-1])
    except:
        weights = results['weights']
    if kde:
        # Derive kernel density estimate.
        wt_kde = gaussian_kde(resample_equal(-logvol, weights))  # KDE
        logvol_grid = np.linspace(logvol[0], logvol[-1], nkde)  # resample
        wt_grid = wt_kde.pdf(-logvol_grid)  # evaluate KDE PDF
        weights = np.interp(-logvol, -logvol_grid, wt_grid)  # interpolate

    # Deal with 1D results. A number of extra catches are also here
    # in case users are trying to plot other results besides the `Results`
    # instance generated by `dynesty`.
    samples = np.atleast_1d(samples)
    if len(samples.shape) == 1:
        samples = np.atleast_2d(samples)
    else:
        assert len(samples.shape) == 2, "Samples must be 1- or 2-D."
        samples = samples.T
    assert samples.shape[0] <= samples.shape[1], "There are more " \
                                                 "dimensions than samples!"

    # Slice samples based on provided `dims`.
    if dims is not None:
        samples = samples[dims]
    ndim, nsamps = samples.shape

    # Check weights.
    if weights.ndim != 1:
        raise ValueError("Weights must be 1-D.")
    if nsamps != weights.shape[0]:
        raise ValueError("The number of weights and samples disagree!")

    # Determine plotting bounds.
    if span is not None:
        if len(span) != ndim:
            raise ValueError("Dimension mismatch between samples and span.")
        for i, _ in enumerate(span):
            try:
                xmin, xmax = span[i]
            except:
                q = [0.5 - 0.5 * span[i], 0.5 + 0.5 * span[i]]
                span[i] = _quantile(samples[i], q, weights=weights)

    # Set labels
    if labels is None:
        labels = [r"$x_{"+str(i+1)+"}$" for i in range(ndim)]

    # Set colormap.
    if color is None:
        color = weights

    # Setup axis layout (from `corner.py`).
    factor = 2.0  # size of side of one panel
    lbdim = 0.5 * factor  # size of left/bottom margin
    trdim = 0.2 * factor  # size of top/right margin
    whspace = 0.05  # size of width/height margin
    plotdim = factor * (ndim - 1.) + factor * (ndim - 2.) * whspace
    dim = lbdim + plotdim + trdim  # total size

    # Initialize figure.
    if fig is None:
        fig, axes = pl.subplots(ndim - 1, ndim - 1, figsize=(dim, dim))
    else:
        try:
            fig, axes = fig
            axes = np.array(axes).reshape((ndim - 1, ndim - 1))
        except:
            raise ValueError("Mismatch between axes and dimension.")

    # Format figure.
    lb = lbdim / dim
    tr = (lbdim + plotdim) / dim
    fig.subplots_adjust(left=lb, bottom=lb, right=tr, top=tr,
                        wspace=whspace, hspace=whspace)

    # Plot the 2-D projected samples.
    for i, x in enumerate(samples[1:]):
        for j, y in enumerate(samples[:-1]):
            try:
                ax = axes[i, j]
            except:
                ax = axes
            # Setup axes.
            if span is not None:
                ax.set_xlim(span[j])
                ax.set_ylim(span[i])
            if j > i:
                ax.set_frame_on(False)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            if max_n_ticks == 0:
                ax.xaxis.set_major_locator(NullLocator())
                ax.yaxis.set_major_locator(NullLocator())
            else:
                ax.xaxis.set_major_locator(MaxNLocator(max_n_ticks,
                                                       prune="lower"))
                ax.yaxis.set_major_locator(MaxNLocator(max_n_ticks,
                                                       prune="lower"))
            # Label axes.
            sf = ScalarFormatter(useMathText=use_math_text)
            ax.xaxis.set_major_formatter(sf)
            ax.yaxis.set_major_formatter(sf)
            if i < ndim - 2:
                ax.set_xticklabels([])
            else:
                [l.set_rotation(45) for l in ax.get_xticklabels()]
                ax.set_xlabel(labels[j], **label_kwargs)
                ax.xaxis.set_label_coords(0.5, -0.3)
            if j > 0:
                ax.set_yticklabels([])
            else:
                [l.set_rotation(45) for l in ax.get_yticklabels()]
                ax.set_ylabel(labels[i+1], **label_kwargs)
                ax.yaxis.set_label_coords(-0.3, 0.5)
            # Plot distribution.
            in_bounds = np.ones_like(y).astype('bool')
            if span is not None and span[i] is not None:
                in_bounds *= ((x >= span[i][0]) & (x <= span[i][1]))
            if span is not None and span[j] is not None:
                in_bounds *= ((y >= span[j][0]) & (y <= span[j][1]))
            ax.scatter(y[in_bounds][::thin], x[in_bounds][::thin],
                       c=color, cmap=cmap, **plot_kwargs)
            # Add truth values
            if truths is not None:
                if truths[j] is not None:
                    try:
                        [ax.axvline(t, color=truth_color,  **truth_kwargs)
                         for t in truths[j]]
                    except:
                        ax.axvline(truths[j], color=truth_color,
                                   **truth_kwargs)
                if truths[i+1] is not None:
                    try:
                        [ax.axhline(t, color=truth_color, **truth_kwargs)
                         for t in truths[i+1]]
                    except:
                        ax.axhline(truths[i+1], color=truth_color,
                                   **truth_kwargs)

    return (fig, axes)


def cornerplot(results, dims=None, span=None, quantiles=[0.025, 0.5, 0.975],
               color='black', smooth=0.02, quantiles_2d=None, hist_kwargs=None,
               hist2d_kwargs=None, labels=None, label_kwargs=None,
               show_titles=False, title_fmt=".2f", title_kwargs=None,
               truths=None, truth_color='red', truth_kwargs=None,
               max_n_ticks=5, top_ticks=False, use_math_text=False,
               verbose=False, fig=None):
    """
    Generate a corner plot of the 1-D and 2-D marginalized posteriors.

    Parameters
    ----------
    results : :class:`~dynesty.results.Results` instance
        A :class:`~dynesty.results.Results` instance from a nested
        sampling run. **Compatible with results derived from**
        `nestle <http://kylebarbary.com/nestle/>`_.

    dims : iterable of shape (ndim,), optional
        The subset of dimensions that should be plotted. If not provided,
        all dimensions will be shown.

    span : iterable with shape (ndim,), optional
        A list where each element is either a length-2 tuple containing
        lower and upper bounds or a float from `(0., 1.]` giving the
        fraction of (weighted) samples to include. If a fraction is provided,
        the bounds are chosen to be equal-tailed. An example would be::

            span = [(0., 10.), 0.95, (5., 6.)]

        Default is `0.999999426697` (5-sigma credible interval).

    quantiles : iterable, optional
        A list of fractional quantiles to overplot on the 1-D marginalized
        posteriors as vertical dashed lines. Default is `[0.025, 0.5, 0.975]`
        (spanning the 95%/2-sigma credible interval).

    color : str or iterable with shape (ndim,), optional
        A `~matplotlib`-style color (either a single color or a different
        value for each subplot) used when plotting the histograms.
        Default is `'black'`.

    smooth : float or iterable with shape (ndim,), optional
        The standard deviation (either a single value or a different value for
        each subplot) for the Gaussian kernel used to smooth the 1-D and 2-D
        marginalized posteriors, expressed as a fraction of the span.
        Default is `0.02` (2% smoothing). If an integer is provided instead,
        this will instead default to a simple (weighted) histogram with
        `bins=smooth`.

    quantiles_2d : iterable with shape (nquant,), optional
        The quantiles used for plotting the smoothed 2-D distributions.
        If not provided, these default to 0.5, 1, 1.5, and 2-sigma contours
        roughly corresponding to quantiles of `[0.1, 0.4, 0.65, 0.85]`.

    hist_kwargs : dict, optional
        Extra keyword arguments to send to the 1-D (smoothed) histograms.

    hist2d_kwargs : dict, optional
        Extra keyword arguments to send to the 2-D (smoothed) histograms.

    labels : iterable with shape (ndim,), optional
        A list of names for each parameter. If not provided, the default name
        used when plotting will follow :math:`x_i` style.

    label_kwargs : dict, optional
        Extra keyword arguments that will be sent to the
        `~matplotlib.axes.Axes.set_xlabel` and
        `~matplotlib.axes.Axes.set_ylabel` methods.

    show_titles : bool, optional
        Whether to display a title above each 1-D marginalized posterior
        showing the 0.5 quantile along with the upper/lower bounds associated
        with the 0.025 and 0.975 (95%/2-sigma credible interval) quantiles.
        Default is `True`.

    title_fmt : str, optional
        The format string for the quantiles provided in the title. Default is
        `'.2f'`.

    title_kwargs : dict, optional
        Extra keyword arguments that will be sent to the
        `~matplotlib.axes.Axes.set_title` command.

    truths : iterable with shape (ndim,), optional
        A list of reference values that will be overplotted on the traces and
        marginalized 1-D posteriors as solid horizontal/vertical lines.
        Individual values can be exempt using `None`. Default is `None`.

    truth_color : str or iterable with shape (ndim,), optional
        A `~matplotlib`-style color (either a single color or a different
        value for each subplot) used when plotting `truths`.
        Default is `'red'`.

    truth_kwargs : dict, optional
        Extra keyword arguments that will be used for plotting the vertical
        and horizontal lines with `truths`.

    max_n_ticks : int, optional
        Maximum number of ticks allowed. Default is `5`.

    top_ticks : bool, optional
        Whether to label the top (rather than bottom) ticks. Default is
        `False`.

    use_math_text : bool, optional
        Whether the axis tick labels for very large/small exponents should be
        displayed as powers of 10 rather than using `e`. Default is `False`.

    verbose : bool, optional
        Whether to print the values of the computed quantiles associated with
        each parameter. Default is `False`.

    fig : (`~matplotlib.figure.Figure`, `~matplotlib.axes.Axes`), optional
        If provided, overplot the traces and marginalized 1-D posteriors
        onto the provided figure. Otherwise, by default an
        internal figure is generated.

    Returns
    -------
    cornerplot : (`~matplotlib.figure.Figure`, `~matplotlib.axes.Axes`)
        Output corner plot.

    """

    # Initialize values.
    if quantiles is None:
        quantiles = []
    if truth_kwargs is None:
        truth_kwargs = dict()
    if label_kwargs is None:
        label_kwargs = dict()
    if title_kwargs is None:
        title_kwargs = dict()
    if hist_kwargs is None:
        hist_kwargs = dict()
    if hist2d_kwargs is None:
        hist2d_kwargs = dict()

    # Set defaults.
    hist_kwargs['alpha'] = hist_kwargs.get('alpha', 0.6)
    hist2d_kwargs['alpha'] = hist2d_kwargs.get('alpha', 0.6)
    hist2d_kwargs['levels'] = hist2d_kwargs.get('levels', quantiles_2d)
    truth_kwargs['linestyle'] = truth_kwargs.get('linestyle', 'solid')
    truth_kwargs['linewidth'] = truth_kwargs.get('linewidth', 2)
    truth_kwargs['alpha'] = truth_kwargs.get('alpha', 0.7)

    # Extract weighted samples.
    samples = results['samples']
    try:
        weights = np.exp(results['logwt'] - results['logz'][-1])
    except:
        weights = results['weights']

    # Deal with 1D results. A number of extra catches are also here
    # in case users are trying to plot other results besides the `Results`
    # instance generated by `dynesty`.
    samples = np.atleast_1d(samples)
    if len(samples.shape) == 1:
        samples = np.atleast_2d(samples)
    else:
        assert len(samples.shape) == 2, "Samples must be 1- or 2-D."
        samples = samples.T
    assert samples.shape[0] <= samples.shape[1], "There are more " \
                                                 "dimensions than samples!"

    # Slice samples based on provided `dims`.
    if dims is not None:
        samples = samples[dims]
    ndim, nsamps = samples.shape

    # Check weights.
    if weights.ndim != 1:
        raise ValueError("Weights must be 1-D.")
    if nsamps != weights.shape[0]:
        raise ValueError("The number of weights and samples disagree!")

    # Determine plotting bounds.
    if span is None:
        span = [0.999999426697 for i in range(ndim)]
    span = list(span)
    if len(span) != ndim:
        raise ValueError("Dimension mismatch between samples and span.")
    for i, _ in enumerate(span):
        try:
            xmin, xmax = span[i]
        except:
            q = [0.5 - 0.5 * span[i], 0.5 + 0.5 * span[i]]
            span[i] = _quantile(samples[i], q, weights=weights)

    # Set labels
    if labels is None:
        labels = [r"$x_{"+str(i+1)+"}$" for i in range(ndim)]

    # Setting up smoothing.
    if (isinstance(smooth, int_type) or isinstance(smooth, float_type)):
        smooth = [smooth for i in range(ndim)]

    # Setup axis layout (from `corner.py`).
    factor = 2.0  # size of side of one panel
    lbdim = 0.5 * factor  # size of left/bottom margin
    trdim = 0.2 * factor  # size of top/right margin
    whspace = 0.05  # size of width/height margin
    plotdim = factor * ndim + factor * (ndim - 1.) * whspace  # plot size
    dim = lbdim + plotdim + trdim  # total size

    # Initialize figure.
    if fig is None:
        fig, axes = pl.subplots(ndim, ndim, figsize=(dim, dim))
    else:
        try:
            fig, axes = fig
            axes = np.array(axes).reshape((ndim, ndim))
        except:
            raise ValueError("Mismatch between axes and dimension.")

    # Format figure.
    lb = lbdim / dim
    tr = (lbdim + plotdim) / dim
    fig.subplots_adjust(left=lb, bottom=lb, right=tr, top=tr,
                        wspace=whspace, hspace=whspace)

    # Plotting.
    for i, x in enumerate(samples):
        if np.shape(samples)[0] == 1:
            ax = axes
        else:
            ax = axes[i, i]

        # Plot the 1-D marginalized posteriors.

        # Setup axes
        ax.set_xlim(span[i])
        if max_n_ticks == 0:
            ax.xaxis.set_major_locator(NullLocator())
            ax.yaxis.set_major_locator(NullLocator())
        else:
            ax.xaxis.set_major_locator(MaxNLocator(max_n_ticks,
                                                   prune="lower"))
            ax.yaxis.set_major_locator(NullLocator())
        # Label axes.
        sf = ScalarFormatter(useMathText=use_math_text)
        ax.xaxis.set_major_formatter(sf)
        if i < ndim - 1:
            if top_ticks:
                ax.xaxis.set_ticks_position("top")
                [l.set_rotation(45) for l in ax.get_xticklabels()]
            else:
                ax.set_xticklabels([])
        else:
            [l.set_rotation(45) for l in ax.get_xticklabels()]
            ax.set_xlabel(labels[i], **label_kwargs)
            ax.xaxis.set_label_coords(0.5, -0.3)
        # Generate distribution.
        sx = smooth[i]
        if isinstance(sx, int_type):
            # If `sx` is an integer, plot a weighted histogram with
            # `sx` bins within the provided bounds.
            n, b, _ = ax.hist(x, bins=sx, weights=weights, color=color,
                              range=np.sort(span[i]), **hist_kwargs)
        else:
            # If `sx` is a float, oversample the data relative to the
            # smoothing filter by a factor of 10, then use a Gaussian
            # filter to smooth the results.
            bins = int(round(10. / sx))
            n, b = np.histogram(x, bins=bins, weights=weights,
                                range=np.sort(span[i]))
            n = norm_kde(n, 10.)
            b0 = 0.5 * (b[1:] + b[:-1])
            n, b, _ = ax.hist(b0, bins=b, weights=n,
                              range=np.sort(span[i]), color=color,
                              **hist_kwargs)
        ax.set_ylim([0., max(n) * 1.05])
        # Plot quantiles.
        if quantiles is not None and len(quantiles) > 0:
            qs = _quantile(x, quantiles, weights=weights)
            for q in qs:
                ax.axvline(q, lw=2, ls="dashed", color=color)
            if verbose:
                print("Quantiles:")
                print(labels[i], [blob for blob in zip(quantiles, qs)])
        # Add truth value(s).
        if truths is not None and truths[i] is not None:
            try:
                [ax.axvline(t, color=truth_color, **truth_kwargs)
                 for t in truths[i]]
            except:
                ax.axvline(truths[i], color=truth_color, **truth_kwargs)
        # Set titles.
        if show_titles:
            title = None
            if title_fmt is not None:
                ql, qm, qh = _quantile(x, [0.025, 0.5, 0.975], weights=weights)
                q_minus, q_plus = qm - ql, qh - qm
                fmt = "{{0:{0}}}".format(title_fmt).format
                title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
                title = title.format(fmt(qm), fmt(q_minus), fmt(q_plus))
                title = "{0} = {1}".format(labels[i], title)
                ax.set_title(title, **title_kwargs)

        for j, y in enumerate(samples):
            if np.shape(samples)[0] == 1:
                ax = axes
            else:
                ax = axes[i, j]

            # Plot the 2-D marginalized posteriors.

            # Setup axes.
            if j > i:
                ax.set_frame_on(False)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            elif j == i:
                continue

            if max_n_ticks == 0:
                ax.xaxis.set_major_locator(NullLocator())
                ax.yaxis.set_major_locator(NullLocator())
            else:
                ax.xaxis.set_major_locator(MaxNLocator(max_n_ticks,
                                                       prune="lower"))
                ax.yaxis.set_major_locator(MaxNLocator(max_n_ticks,
                                                       prune="lower"))
            # Label axes.
            sf = ScalarFormatter(useMathText=use_math_text)
            ax.xaxis.set_major_formatter(sf)
            ax.yaxis.set_major_formatter(sf)
            if i < ndim - 1:
                ax.set_xticklabels([])
            else:
                [l.set_rotation(45) for l in ax.get_xticklabels()]
                ax.set_xlabel(labels[j], **label_kwargs)
                ax.xaxis.set_label_coords(0.5, -0.3)
            if j > 0:
                ax.set_yticklabels([])
            else:
                [l.set_rotation(45) for l in ax.get_yticklabels()]
                ax.set_ylabel(labels[i], **label_kwargs)
                ax.yaxis.set_label_coords(-0.3, 0.5)
            # Generate distribution.
            sy = smooth[j]
            check_ix = isinstance(sx, int_type)
            check_iy = isinstance(sy, int_type)
            if check_ix and check_iy:
                fill_contours = False
                plot_contours = False
            else:
                fill_contours = True
                plot_contours = True
            hist2d_kwargs['fill_contours'] = hist2d_kwargs.get('fill_contours',
                                                               fill_contours)
            hist2d_kwargs['plot_contours'] = hist2d_kwargs.get('plot_contours',
                                                               plot_contours)
            _hist2d(y, x, ax=ax, span=[span[j], span[i]],
                    weights=weights, color=color, smooth=[sy, sx],
                    **hist2d_kwargs)
            # Add truth values
            if truths is not None:
                if truths[j] is not None:
                    try:
                        [ax.axvline(t, color=truth_color, **truth_kwargs)
                         for t in truths[j]]
                    except:
                        ax.axvline(truths[j], color=truth_color,
                                   **truth_kwargs)
                if truths[i] is not None:
                    try:
                        [ax.axhline(t, color=truth_color, **truth_kwargs)
                         for t in truths[i]]
                    except:
                        ax.axhline(truths[i], color=truth_color,
                                   **truth_kwargs)

    return (fig, axes)


def boundplot(results, dims, it=None, idx=None, prior_transform=None,
              periodic=None, reflective=None, ndraws=5000, color='gray',
              plot_kwargs=None, labels=None, label_kwargs=None, max_n_ticks=5,
              use_math_text=False, show_live=False, live_color='darkviolet',
              live_kwargs=None, span=None, fig=None):
    """
    Return the bounding distribution used to propose either (1) live points
    at a given iteration or (2) a specific dead point during
    the course of a run, projected onto the two dimensions specified
    by `dims`.

    Parameters
    ----------
    results : :class:`~dynesty.results.Results` instance
        A :class:`~dynesty.results.Results` instance from a nested
        sampling run.

    dims : length-2 tuple
        The dimensions used to plot the bounding.

    it : int, optional
        If provided, returns the bounding distribution at the specified
        iteration of the nested sampling run. **Note that this option and
        `idx` are mutually exclusive.**

    idx : int, optional
        If provided, returns the bounding distribution used to propose the
        dead point at the specified iteration of the nested sampling run.
        **Note that this option and `it` are mutually exclusive.**

    prior_transform : func, optional
        The function transforming samples within the unit cube back to samples
        in the native model space. If provided, the transformed bounding
        distribution will be plotted in the native model space.

    periodic : iterable, optional
        A list of indices for parameters with periodic boundary conditions.
        These parameters *will not* have their positions constrained to be
        within the unit cube, enabling smooth behavior for parameters
        that may wrap around the edge. Default is `None` (i.e. no periodic
        boundary conditions).

    reflective : iterable, optional
        A list of indices for parameters with reflective boundary conditions.
        These parameters *will not* have their positions constrained to be
        within the unit cube, enabling smooth behavior for parameters
        that may reflect at the edge. Default is `None` (i.e. no reflective
        boundary conditions).

    ndraws : int, optional
        The number of random samples to draw from the bounding distribution
        when plotting. Default is `5000`.

    color : str, optional
        The color of the points randomly sampled from the bounding
        distribution. Default is `'gray'`.

    plot_kwargs : dict, optional
        Extra keyword arguments used when plotting the bounding draws.

    labels : iterable with shape (ndim,), optional
        A list of names for each parameter. If not provided, the default name
        used when plotting will follow :math:`x_i` style.

    label_kwargs : dict, optional
        Extra keyword arguments that will be sent to the
        `~matplotlib.axes.Axes.set_xlabel` and
        `~matplotlib.axes.Axes.set_ylabel` methods.

    max_n_ticks : int, optional
        Maximum number of ticks allowed. Default is `5`.

    use_math_text : bool, optional
        Whether the axis tick labels for very large/small exponents should be
        displayed as powers of 10 rather than using `e`. Default is `False`.

    show_live : bool, optional
        Whether the live points at a given iteration (for `it`) or
        associated with the bounding (for `idx`) should be highlighted.
        Default is `False`. In the dynamic case, only the live points
        associated with the batch used to construct the relevant bound
        are plotted.

    live_color : str, optional
        The color of the live points. Default is `'darkviolet'`.

    live_kwargs : dict, optional
        Extra keyword arguments used when plotting the live points.

    span : iterable with shape (2,), optional
        A list where each element is a length-2 tuple containing
        lower and upper bounds. Default is `None` (no bound).

    fig : (`~matplotlib.figure.Figure`, `~matplotlib.axes.Axes`), optional
        If provided, overplot the draws onto the provided figure.
        Otherwise, by default an internal figure is generated.

    Returns
    -------
    bounding_plot : (`~matplotlib.figure.Figure`, `~matplotlib.axes.Axes`)
        Output plot of the bounding distribution.

    """

    # Initialize values.
    if plot_kwargs is None:
        plot_kwargs = dict()
    if label_kwargs is None:
        label_kwargs = dict()
    if live_kwargs is None:
        live_kwargs = dict()

    # Check that either `idx` or `it` has been specified.
    if (it is None and idx is None) or (it is not None and idx is not None):
        raise ValueError("You must specify either an iteration or an index!")

    # Set defaults.
    plot_kwargs['marker'] = plot_kwargs.get('marker', 'o')
    plot_kwargs['linestyle'] = plot_kwargs.get('linestyle', 'None')
    plot_kwargs['markersize'] = plot_kwargs.get('markersize', 1)
    plot_kwargs['alpha'] = plot_kwargs.get('alpha', 0.4)
    live_kwargs['marker'] = live_kwargs.get('marker', 'o')
    live_kwargs['linestyle'] = live_kwargs.get('linestyle', 'None')
    live_kwargs['markersize'] = live_kwargs.get('markersize', 1)

    # Extract bounding distributions.
    try:
        bounds = results['bound']
    except:
        raise ValueError("No bounds were saved in the results!")
    nsamps = len(results['samples'])

    # Gather boundary conditions.
    nonbounded = np.ones(bounds[0].n, dtype='bool')
    if periodic is not None:
        nonbounded[periodic] = False
    if reflective is not None:
        nonbounded[reflective] = False

    if it is not None:
        if it >= nsamps:
            raise ValueError("The iteration requested goes beyond the "
                             "number of iterations in the run.")
        # Extract bound iterations.
        try:
            bound_iter = np.array(results['bound_iter'])
        except:
            raise ValueError("Cannot reconstruct the bound used at the "
                             "specified iteration since bound "
                             "iterations were not saved in the results.")

        # Find bound at the specified iteration.
        if it == 0:
            pidx = 0
        else:
            pidx = bound_iter[it]
    else:
        if idx >= nsamps:
            raise ValueError("The index requested goes beyond the "
                             "number of samples in the run.")
        try:
            samples_bound = results['samples_bound']
        except:
            raise ValueError("Cannot reconstruct the bound used to "
                             "compute the specified dead point since "
                             "sample bound indices were not saved "
                             "in the results.")
        # Grab relevant bound.
        pidx = samples_bound[idx]

    # Get desired bound.
    bound = bounds[pidx]

    # Do we want to show the live points at the specified iteration?
    # If so, we need to rewind our bound to check.
    # (We could also go forward; this is an arbitrary choice.)
    if show_live:
        try:
            # We can only reconstruct the run if the final set of live points
            # were added to the results. This is true by default for dynamic
            # nested sampling runs but not guaranteeed for standard runs.
            nlive = results['nlive']
            niter = results['niter']
            if nsamps - niter != nlive:
                raise ValueError("Cannot reconstruct bound because the "
                                 "final set of live points are not included "
                                 "in the results.")
            # Grab our final set of live points (with proper IDs).
            samples = results['samples_u']
            samples_id = results['samples_id']
            ndim = samples.shape[1]
            live_u = np.empty((nlive, ndim))
            live_u[samples_id[-nlive:]] = samples[-nlive:]
            # Find generating bound ID if necessary.
            if it is None:
                it = results['samples_it'][idx]
            # Run our sampling backwards.
            for i in range(1, niter - it + 1):
                r = -(nlive + i)
                uidx = samples_id[r]
                live_u[uidx] = samples[r]
        except:
            # In the dynamic sampling case, we will show the live points used
            # during the batch associated with a particular iteration/bound.
            batch = results['samples_batch'][it]  # select batch
            nbatch = results['batch_nlive'][batch]  # nlive in the batch
            bsel = results['samples_batch'] == batch  # select batch
            niter_eff = sum(bsel) - nbatch  # "effective" iterations in batch
            # Grab our final set of live points (with proper IDs).
            samples = results['samples_u'][bsel]
            samples_id = results['samples_id'][bsel]
            samples_id -= min(samples_id)  # re-index to start at zero
            ndim = samples.shape[1]
            live_u = np.empty((nbatch, ndim))
            live_u[samples_id[-nbatch:]] = samples[-nbatch:]
            # Find generating bound ID if necessary.
            if it is None:
                it = results['samples_it'][idx]
            it_eff = sum(bsel[:it+1])  # effective iteration in batch
            # Run our sampling backwards.
            for i in range(1, niter_eff - it_eff + 1):
                r = -(nbatch + i)
                uidx = samples_id[r]
                live_u[uidx] = samples[r]

    # Draw samples from the bounding distribution.
    try:
        # If bound is "fixed", go ahead and draw samples from it.
        psamps = bound.samples(ndraws)
    except:
        # If bound is based on the distribution of live points at a
        # specific iteration, we need to reconstruct what those were.
        if not show_live:
            try:
                # Only reconstruct the run if we haven't done it already.
                nlive = results['nlive']
                niter = results['niter']
                if nsamps - niter != nlive:
                    raise ValueError("Cannot reconstruct bound because the "
                                     "final set of live points are not "
                                     "included in the results.")
                # Grab our final set of live points (with proper IDs).
                samples = results['samples_u']
                samples_id = results['samples_id']
                ndim = samples.shape[1]
                live_u = np.empty((nlive, ndim))
                live_u[samples_id[-nlive:]] = samples[-nlive:]
                # Run our sampling backwards.
                if it is None:
                    it = results['samples_it'][idx]
                for i in range(1, niter - it + 1):
                    r = -(nlive + i)
                    uidx = samples_id[r]
                    live_u[uidx] = samples[r]
            except:
                raise ValueError("Live point tracking currently not "
                                 "implemented for dynamic sampling results.")
        # Construct a KDTree to speed up nearest-neighbor searches.
        kdtree = spatial.KDTree(live_u)
        # Draw samples.
        psamps = bound.samples(ndraws, live_u, kdtree=kdtree)

    # Projecting samples to input dimensions and possibly
    # the native model space.
    if prior_transform is None:
        x1, x2 = psamps[:, dims].T
        if show_live:
            l1, l2 = live_u[:, dims].T
    else:
        # Remove points outside of the unit cube as appropriate.
        sel = [unitcheck(point, nonbounded) for point in psamps]
        vsamps = np.array(list(map(prior_transform, psamps[sel])))
        x1, x2 = vsamps[:, dims].T
        if show_live:
            lsamps = np.array(list(map(prior_transform, live_u)))
            l1, l2 = lsamps[:, dims].T

    # Setting up default plot layout.
    if fig is None:
        fig, axes = pl.subplots(1, 1, figsize=(6, 6))
    else:
        fig, axes = fig
        try:
            axes.plot()
        except:
            raise ValueError("Provided axes do not match the required shape "
                             "for plotting samples.")

    # Plotting.
    axes.plot(x1, x2, color=color, zorder=1, **plot_kwargs)
    if show_live:
        axes.plot(l1, l2, color=live_color, zorder=2, **live_kwargs)

    # Setup axes
    if span is not None:
        axes.set_xlim(span[0])
        axes.set_ylim(span[1])
    if max_n_ticks == 0:
        axes.xaxis.set_major_locator(NullLocator())
        axes.yaxis.set_major_locator(NullLocator())
    else:
        axes.xaxis.set_major_locator(MaxNLocator(max_n_ticks))
        axes.yaxis.set_major_locator(MaxNLocator(max_n_ticks))
    # Label axes.
    sf = ScalarFormatter(useMathText=use_math_text)
    axes.xaxis.set_major_formatter(sf)
    axes.yaxis.set_major_formatter(sf)
    if labels is not None:
        axes.set_xlabel(labels[0], **label_kwargs)
        axes.set_ylabel(labels[1], **label_kwargs)
    else:
        axes.set_xlabel(r"$x_{"+str(dims[0]+1)+"}$", **label_kwargs)
        axes.set_ylabel(r"$x_{"+str(dims[1]+1)+"}$", **label_kwargs)

    return fig, axes


def cornerbound(results, it=None, idx=None, dims=None, prior_transform=None,
                periodic=None, reflective=None, ndraws=5000, color='gray',
                plot_kwargs=None, labels=None, label_kwargs=None, max_n_ticks=5,
                use_math_text=False, show_live=False, live_color='darkviolet',
                live_kwargs=None, span=None, fig=None):
    """
    Return the bounding distribution used to propose either (1) live points
    at a given iteration or (2) a specific dead point during
    the course of a run, projected onto all pairs of dimensions.

    Parameters
    ----------
    results : :class:`~dynesty.results.Results` instance
        A :class:`~dynesty.results.Results` instance from a nested
        sampling run.

    it : int, optional
        If provided, returns the bounding distribution at the specified
        iteration of the nested sampling run. **Note that this option and
        `idx` are mutually exclusive.**

    idx : int, optional
        If provided, returns the bounding distribution used to propose the
        dead point at the specified iteration of the nested sampling run.
        **Note that this option and `it` are mutually exclusive.**

    dims : iterable of shape (ndim,), optional
        The subset of dimensions that should be plotted. If not provided,
        all dimensions will be shown.

    prior_transform : func, optional
        The function transforming samples within the unit cube back to samples
        in the native model space. If provided, the transformed bounding
        distribution will be plotted in the native model space.

    periodic : iterable, optional
        A list of indices for parameters with periodic boundary conditions.
        These parameters *will not* have their positions constrained to be
        within the unit cube, enabling smooth behavior for parameters
        that may wrap around the edge. Default is `None` (i.e. no periodic
        boundary conditions).

    reflective : iterable, optional
        A list of indices for parameters with reflective boundary conditions.
        These parameters *will not* have their positions constrained to be
        within the unit cube, enabling smooth behavior for parameters
        that may reflect at the edge. Default is `None` (i.e. no reflective
        boundary conditions).

    ndraws : int, optional
        The number of random samples to draw from the bounding distribution
        when plotting. Default is `5000`.

    color : str, optional
        The color of the points randomly sampled from the bounding
        distribution. Default is `'gray'`.

    plot_kwargs : dict, optional
        Extra keyword arguments used when plotting the bounding draws.

    labels : iterable with shape (ndim,), optional
        A list of names for each parameter. If not provided, the default name
        used when plotting will be in :math:`x_i` style.

    label_kwargs : dict, optional
        Extra keyword arguments that will be sent to the
        `~matplotlib.axes.Axes.set_xlabel` and
        `~matplotlib.axes.Axes.set_ylabel` methods.

    max_n_ticks : int, optional
        Maximum number of ticks allowed. Default is `5`.

    use_math_text : bool, optional
        Whether the axis tick labels for very large/small exponents should be
        displayed as powers of 10 rather than using `e`. Default is `False`.

    show_live : bool, optional
        Whether the live points at a given iteration (for `it`) or
        associated with the bounding (for `idx`) should be highlighted.
        Default is `False`. In the dynamic case, only the live points
        associated with the batch used to construct the relevant bound
        are plotted.

    live_color : str, optional
        The color of the live points. Default is `'darkviolet'`.

    live_kwargs : dict, optional
        Extra keyword arguments used when plotting the live points.

    span : iterable with shape (2,), optional
        A list where each element is a length-2 tuple containing
        lower and upper bounds. Default is `None` (no bound).

    fig : (`~matplotlib.figure.Figure`, `~matplotlib.axes.Axes`), optional
        If provided, overplot the draws onto the provided figure.
        Otherwise, by default an internal figure is generated.


    Returns
    -------
    cornerbound : (`~matplotlib.figure.Figure`, `~matplotlib.axes.Axes`)
        Output corner plot of the bounding distribution.

    """

    # Initialize values.
    if label_kwargs is None:
        label_kwargs = dict()
    if plot_kwargs is None:
        plot_kwargs = dict()
    if live_kwargs is None:
        live_kwargs = dict()

    # Check that either `idx` or `it` is specified.
    if (it is None and idx is None) or (it is not None and idx is not None):
        raise ValueError("You must specify either an iteration or an index!")

    # Set defaults.
    plot_kwargs['marker'] = plot_kwargs.get('marker', 'o')
    plot_kwargs['linestyle'] = plot_kwargs.get('linestyle', 'None')
    plot_kwargs['markersize'] = plot_kwargs.get('markersize', 1)
    plot_kwargs['alpha'] = plot_kwargs.get('alpha', 0.4)
    live_kwargs['marker'] = live_kwargs.get('marker', 'o')
    live_kwargs['linestyle'] = live_kwargs.get('linestyle', 'None')
    live_kwargs['markersize'] = live_kwargs.get('markersize', 1)

    # Extract bounding distributions.
    try:
        bounds = results['bound']
    except:
        raise ValueError("No bounds were saved in the results!")
    nsamps = len(results['samples'])

    # Gather boundary conditions.
    nonbounded = np.ones(bounds[0].n, dtype='bool')
    if periodic is not None:
        nonbounded[periodic] = False
    if reflective is not None:
        nonbounded[reflective] = False

    if it is not None:
        if it >= nsamps:
            raise ValueError("The iteration requested goes beyond the "
                             "number of iterations in the run.")
        # Extract bound iterations.
        try:
            bound_iter = np.array(results['bound_iter'])
        except:
            raise ValueError("Cannot reconstruct the bound used at the "
                             "specified iteration since bound "
                             "iterations were not saved in the results.")

        # Find bound at the specified iteration.
        if it == 0:
            pidx = 0
        else:
            pidx = bound_iter[it]
    else:
        if idx >= nsamps:
            raise ValueError("The index requested goes beyond the "
                             "number of samples in the run.")
        try:
            samples_bound = results['samples_bound']
        except:
            raise ValueError("Cannot reconstruct the bound used to "
                             "compute the specified dead point since "
                             "sample bound indices were not saved "
                             "in the results.")
        # Grab relevant bound.
        pidx = samples_bound[idx]

    # Get desired bound.
    bound = bounds[pidx]

    # Do we want to show the live points at the specified iteration?
    # If so, we need to rewind our bound to check.
    # (We could also go forward; this is an arbitrary choice.)
    if show_live:
        try:
            # We can only reconstruct the run if the final set of live points
            # were added to the results. This is true by default for dynamic
            # nested sampling runs but not guaranteeed for standard runs.
            nlive = results['nlive']
            niter = results['niter']
            if nsamps - niter != nlive:
                raise ValueError("Cannot reconstruct bound because the "
                                 "final set of live points are not included "
                                 "in the results.")
            # Grab our final set of live points (with proper IDs).
            samples = results['samples_u']
            samples_id = results['samples_id']
            ndim = samples.shape[1]
            live_u = np.empty((nlive, ndim))
            live_u[samples_id[-nlive:]] = samples[-nlive:]
            # Find generating bound ID if necessary.
            if it is None:
                it = results['samples_it'][idx]
            # Run our sampling backwards.
            for i in range(1, niter - it + 1):
                r = -(nlive + i)
                uidx = samples_id[r]
                live_u[uidx] = samples[r]
        except:
            # In the dynamic sampling case, we will show the live points used
            # during the batch associated with a particular iteration/bound.
            if it is not None:
                batch = results['samples_batch'][it]  # select batch
            else:
                batch = results['samples_batch'][idx]
            nbatch = results['batch_nlive'][batch]  # nlive in the batch
            bsel = results['samples_batch'] == batch  # select batch
            niter_eff = sum(bsel) - nbatch  # "effective" iterations in batch
            # Grab our final set of live points (with proper IDs).
            samples = results['samples_u'][bsel]
            samples_id = results['samples_id'][bsel]
            samples_id -= min(samples_id)  # re-index to start at zero
            ndim = samples.shape[1]
            live_u = np.empty((nbatch, ndim))
            live_u[samples_id[-nbatch:]] = samples[-nbatch:]
            # Find generating bound ID if necessary.
            if it is None:
                it = results['samples_it'][idx]
            it_eff = sum(bsel[:it+1])  # effective iteration in batch
            # Run our sampling backwards.
            for i in range(1, niter_eff - it_eff + 1):
                r = -(nbatch + i)
                uidx = samples_id[r]
                live_u[uidx] = samples[r]

    # Draw samples from the bounding distribution.
    try:
        # If bound is "fixed", go ahead and draw samples from it.
        psamps = bound.samples(ndraws)
    except:
        # If bound is based on the distribution of live points at a
        # specific iteration, we need to reconstruct what those were.
        if not show_live:
            # Only reconstruct the run if we haven't done it already.
            nlive = results['nlive']
            niter = results['niter']
            if nsamps - niter != nlive:
                raise ValueError("Cannot reconstruct bound because the "
                                 "final set of live points are not included "
                                 "in the results.")
            # Grab our final set of live points (with proper IDs).
            samples = results['samples_u']
            samples_id = results['samples_id']
            ndim = samples.shape[1]
            live_u = np.empty((nlive, ndim))
            live_u[samples_id[-nlive:]] = samples[-nlive:]
            # Run our sampling backwards.
            if it is None:
                it = results['samples_it'][idx]
            for i in range(1, niter - it + 1):
                r = -(nlive + i)
                uidx = samples_id[r]
                live_u[uidx] = samples[r]
        # Construct a KDTree to speed up nearest-neighbor searches.
        kdtree = spatial.KDTree(live_u)
        # Draw samples.
        psamps = bound.samples(ndraws, live_u, kdtree=kdtree)

    # Projecting samples to input dimensions and possibly
    # the native model space.
    if prior_transform is None:
        psamps = psamps.T
        if show_live:
            lsamps = live_u.T
    else:
        # Remove points outside of the unit cube.
        sel = [unitcheck(point, nonbounded) for point in psamps]
        psamps = np.array(list(map(prior_transform, psamps[sel])))
        psamps = psamps.T
        if show_live:
            lsamps = np.array(list(map(prior_transform, live_u)))
            lsamps = lsamps.T

    # Subsample dimensions.
    if dims is not None:
        psamps = psamps[dims]
        if show_live:
            lsamps = lsamps[dims]
    ndim = psamps.shape[0]

    # Set labels.
    if labels is None:
        labels = [r"$x_{"+str(i+1)+"}$" for i in range(ndim)]

    # Setup axis layout (from `corner.py`).
    factor = 2.0  # size of side of one panel
    lbdim = 0.5 * factor  # size of left/bottom margin
    trdim = 0.2 * factor  # size of top/right margin
    whspace = 0.05  # size of width/height margin
    plotdim = factor * (ndim - 1.) + factor * (ndim - 2.) * whspace
    dim = lbdim + plotdim + trdim  # total size

    # Initialize figure.
    if fig is None:
        fig, axes = pl.subplots(ndim - 1, ndim - 1, figsize=(dim, dim))
    else:
        try:
            fig, axes = fig
            axes = np.array(axes).reshape((ndim - 1, ndim - 1))
        except:
            raise ValueError("Mismatch between axes and dimension.")

    # Format figure.
    lb = lbdim / dim
    tr = (lbdim + plotdim) / dim
    fig.subplots_adjust(left=lb, bottom=lb, right=tr, top=tr,
                        wspace=whspace, hspace=whspace)

    # Plot the 2-D projected samples.
    for i, x in enumerate(psamps[1:]):
        for j, y in enumerate(psamps[:-1]):
            try:
                ax = axes[i, j]
            except:
                ax = axes
            # Setup axes.
            if span is not None:
                ax.set_xlim(span[j])
                ax.set_ylim(span[i])
            if j > i:
                ax.set_frame_on(False)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            if max_n_ticks == 0:
                ax.xaxis.set_major_locator(NullLocator())
                ax.yaxis.set_major_locator(NullLocator())
            else:
                ax.xaxis.set_major_locator(MaxNLocator(max_n_ticks,
                                                       prune="lower"))
                ax.yaxis.set_major_locator(MaxNLocator(max_n_ticks,
                                                       prune="lower"))
            # Label axes.
            sf = ScalarFormatter(useMathText=use_math_text)
            ax.xaxis.set_major_formatter(sf)
            ax.yaxis.set_major_formatter(sf)
            if i < ndim - 2:
                ax.set_xticklabels([])
            else:
                [l.set_rotation(45) for l in ax.get_xticklabels()]
                ax.set_xlabel(labels[j], **label_kwargs)
                ax.xaxis.set_label_coords(0.5, -0.3)
            if j > 0:
                ax.set_yticklabels([])
            else:
                [l.set_rotation(45) for l in ax.get_yticklabels()]
                ax.set_ylabel(labels[i+1], **label_kwargs)
                ax.yaxis.set_label_coords(-0.3, 0.5)
            # Plot distribution.
            ax.plot(y, x, c=color, **plot_kwargs)
            # Add live points.
            if show_live:
                ax.plot(lsamps[j], lsamps[i+1], c=live_color, **live_kwargs)

    return (fig, axes)


def _hist2d(x, y, smooth=0.02, span=None, weights=None, levels=None,
            ax=None, color='gray', plot_datapoints=False, plot_density=True,
            plot_contours=True, no_fill_contours=False, fill_contours=True,
            contour_kwargs=None, contourf_kwargs=None, data_kwargs=None,
            **kwargs):
    """
    Internal function called by :meth:`cornerplot` used to generate a
    a 2-D histogram/contour of samples.

    Parameters
    ----------
    x : interable with shape (nsamps,)
       Sample positions in the first dimension.

    y : iterable with shape (nsamps,)
       Sample positions in the second dimension.

    span : iterable with shape (ndim,), optional
        A list where each element is either a length-2 tuple containing
        lower and upper bounds or a float from `(0., 1.]` giving the
        fraction of (weighted) samples to include. If a fraction is provided,
        the bounds are chosen to be equal-tailed. An example would be::

            span = [(0., 10.), 0.95, (5., 6.)]

        Default is `0.999999426697` (5-sigma credible interval).

    weights : iterable with shape (nsamps,)
        Weights associated with the samples. Default is `None` (no weights).

    levels : iterable, optional
        The contour levels to draw. Default are `[0.5, 1, 1.5, 2]`-sigma.

    ax : `~matplotlib.axes.Axes`, optional
        An `~matplotlib.axes.axes` instance on which to add the 2-D histogram.
        If not provided, a figure will be generated.

    color : str, optional
        The `~matplotlib`-style color used to draw lines and color cells
        and contours. Default is `'gray'`.

    plot_datapoints : bool, optional
        Whether to plot the individual data points. Default is `False`.

    plot_density : bool, optional
        Whether to draw the density colormap. Default is `True`.

    plot_contours : bool, optional
        Whether to draw the contours. Default is `True`.

    no_fill_contours : bool, optional
        Whether to add absolutely no filling to the contours. This differs
        from `fill_contours=False`, which still adds a white fill at the
        densest points. Default is `False`.

    fill_contours : bool, optional
        Whether to fill the contours. Default is `True`.

    contour_kwargs : dict
        Any additional keyword arguments to pass to the `contour` method.

    contourf_kwargs : dict
        Any additional keyword arguments to pass to the `contourf` method.

    data_kwargs : dict
        Any additional keyword arguments to pass to the `plot` method when
        adding the individual data points.

    """

    if ax is None:
        ax = pl.gca()

    # Determine plotting bounds.
    data = [x, y]
    if span is None:
        span = [0.999999426697 for i in range(2)]
    span = list(span)
    if len(span) != 2:
        raise ValueError("Dimension mismatch between samples and span.")
    for i, _ in enumerate(span):
        try:
            xmin, xmax = span[i]
        except:
            q = [0.5 - 0.5 * span[i], 0.5 + 0.5 * span[i]]
            span[i] = _quantile(data[i], q, weights=weights)

    # The default "sigma" contour levels.
    if levels is None:
        levels = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2)

    # Color map for the density plot, over-plotted to indicate the
    # density of the points near the center.
    density_cmap = LinearSegmentedColormap.from_list(
        "density_cmap", [color, (1, 1, 1, 0)])

    # Color map used to hide the points at the high density areas.
    white_cmap = LinearSegmentedColormap.from_list(
        "white_cmap", [(1, 1, 1), (1, 1, 1)], N=2)

    # This "color map" is the list of colors for the contour levels if the
    # contours are filled.
    rgba_color = colorConverter.to_rgba(color)
    contour_cmap = [list(rgba_color) for l in levels] + [rgba_color]
    for i, l in enumerate(levels):
        contour_cmap[i][-1] *= float(i) / (len(levels)+1)

    # Initialize smoothing.
    if (isinstance(smooth, int_type) or isinstance(smooth, float_type)):
        smooth = [smooth, smooth]
    bins = []
    svalues = []
    for s in smooth:
        if isinstance(s, int_type):
            # If `s` is an integer, the weighted histogram has
            # `s` bins within the provided bounds.
            bins.append(s)
            svalues.append(0.)
        else:
            # If `s` is a float, oversample the data relative to the
            # smoothing filter by a factor of 2, then use a Gaussian
            # filter to smooth the results.
            bins.append(int(round(2. / s)))
            svalues.append(2.)

    # We'll make the 2D histogram to directly estimate the density.
    try:
        H, X, Y = np.histogram2d(x.flatten(), y.flatten(), bins=bins,
                                 range=list(map(np.sort, span)),
                                 weights=weights)
    except ValueError:
        raise ValueError("It looks like at least one of your sample columns "
                         "have no dynamic range.")

    # Smooth the results.
    if not np.all(svalues == 0.):
        H = norm_kde(H, svalues)

    # Compute the density levels.
    Hflat = H.flatten()
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    sm = np.cumsum(Hflat)
    sm /= sm[-1]
    V = np.empty(len(levels))
    for i, v0 in enumerate(levels):
        try:
            V[i] = Hflat[sm <= v0][-1]
        except:
            V[i] = Hflat[0]
    V.sort()
    m = (np.diff(V) == 0)
    if np.any(m) and plot_contours:
        logging.warning("Too few points to create valid contours.")
    while np.any(m):
        V[np.where(m)[0][0]] *= 1.0 - 1e-4
        m = (np.diff(V) == 0)
    V.sort()

    # Compute the bin centers.
    X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])

    # Extend the array for the sake of the contours at the plot edges.
    H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
    H2[2:-2, 2:-2] = H
    H2[2:-2, 1] = H[:, 0]
    H2[2:-2, -2] = H[:, -1]
    H2[1, 2:-2] = H[0]
    H2[-2, 2:-2] = H[-1]
    H2[1, 1] = H[0, 0]
    H2[1, -2] = H[0, -1]
    H2[-2, 1] = H[-1, 0]
    H2[-2, -2] = H[-1, -1]
    X2 = np.concatenate([X1[0] + np.array([-2, -1]) * np.diff(X1[:2]), X1,
                         X1[-1] + np.array([1, 2]) * np.diff(X1[-2:])])
    Y2 = np.concatenate([Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]), Y1,
                         Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:])])

    # Plot the data points.
    if plot_datapoints:
        if data_kwargs is None:
            data_kwargs = dict()
        data_kwargs["color"] = data_kwargs.get("color", color)
        data_kwargs["ms"] = data_kwargs.get("ms", 2.0)
        data_kwargs["mec"] = data_kwargs.get("mec", "none")
        data_kwargs["alpha"] = data_kwargs.get("alpha", 0.1)
        ax.plot(x, y, "o", zorder=-1, rasterized=True, **data_kwargs)

    # Plot the base fill to hide the densest data points.
    if (plot_contours or plot_density) and not no_fill_contours:
        ax.contourf(X2, Y2, H2.T, [V.min(), H.max()],
                    cmap=white_cmap, antialiased=False)

    if plot_contours and fill_contours:
        if contourf_kwargs is None:
            contourf_kwargs = dict()
        contourf_kwargs["colors"] = contourf_kwargs.get("colors", contour_cmap)
        contourf_kwargs["antialiased"] = contourf_kwargs.get("antialiased",
                                                             False)
        ax.contourf(X2, Y2, H2.T, np.concatenate([[0], V, [H.max()*(1+1e-4)]]),
                    **contourf_kwargs)

    # Plot the density map. This can't be plotted at the same time as the
    # contour fills.
    elif plot_density:
        ax.pcolor(X, Y, H.max() - H.T, cmap=density_cmap)

    # Plot the contour edge colors.
    if plot_contours:
        if contour_kwargs is None:
            contour_kwargs = dict()
        contour_kwargs["colors"] = contour_kwargs.get("colors", color)
        ax.contour(X2, Y2, H2.T, V, **contour_kwargs)

    ax.set_xlim(span[0])
    ax.set_ylim(span[1])
