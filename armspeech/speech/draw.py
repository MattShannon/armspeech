"""Draw graphs of trajectories, etc."""

# Copyright 2011, 2012, 2013, 2014, 2015 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.

import math
import numpy as np

from matplotlib.figure import Figure
from matplotlib import cm
from matplotlib import transforms
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from codedep import codeDeps

import armspeech.modelling.dist as d
import armspeech.numpy_settings

# (FIXME : not sure that the matplotlib OO API is completely thread-safe.
#   It is conceivable that we should really be using explicit locks below.)

@codeDeps()
def partitionSeq(xs, numPartitions):
    out = [ [] for i in range(numPartitions) ]
    for i, x in enumerate(xs):
        out[i % numPartitions].append(x)
    return out

@codeDeps()
def drawLabelledSeq(dataSeqs, labelSeqs, outPdf,
                    figSizeRate = None, figHeight = 6.0,
                    fillBetween = [], xmin = None, xmax = None, ylims = None,
                    xlabel = None, ylabel = None, legend = None,
                    lineStyles = None,
                    labelColors = ['red', 'purple', 'orange', 'blue'],
                    tightLayout = False):
    if xmin is None:
        xmin = min([ dataSeq[0][0] for dataSeq in dataSeqs ] +
                   [ labelSeq[0][0] for labelSeq in labelSeqs ])
    if xmax is None:
        xmax = max([ dataSeq[0][-1] for dataSeq in dataSeqs ] +
                   [ labelSeq[-1][1] for labelSeq in labelSeqs ])

    if figSizeRate is None:
        fig = Figure()
    else:
        fig = Figure(
            figsize = ((xmax - xmin) * figSizeRate, figHeight),
            dpi = 300.0
        )
    ax = fig.add_subplot(1, 1, 1)

    for x, y1, y2 in fillBetween:
        ax.fill_between(x, y1, y2, color = 'blue', facecolor = 'blue',
                        linewidth = 0.2, alpha = 0.2)
    for dataSeqIndex, (x, y) in enumerate(dataSeqs):
        lineStyle = None if lineStyles is None else lineStyles[dataSeqIndex]
        ax.plot(x, y, '-' if lineStyle is None else lineStyle)

    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)

    for labelSeqIndex, labelSeq in enumerate(labelSeqs):
        color = labelColors[labelSeqIndex % len(labelColors)]
        height = labelSeqIndex * 0.04
        for start, end, label in labelSeq:
            pos = (start + end) / 2.0
            if xmin < pos < xmax:
                ax.text(pos, height + 0.02, label, transform = trans,
                        horizontalalignment = 'center',
                        verticalalignment = 'center',
                        fontsize = 8)
            ax.axvspan(xmin = start, xmax = end,
                       ymin = height + 0.0, ymax = height + 0.04,
                       facecolor = color, alpha = 0.5)
            ax.axvline(x = start, linewidth = 0.1, color = 'black')

    ax.set_xlim(xmin, xmax)
    if ylims is not None:
        ax.set_ylim(*ylims)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if legend is not None:
        ax.legend(legend)

    if tightLayout:
        # (FIXME : this seems to produce a warning about falling back to
        #   a different backend)
        fig.tight_layout()

    canvas = FigureCanvas(fig)
    canvas.print_figure(outPdf)

@codeDeps()
def drawWarping(transformList, outPdf, xlims, ylims = None, title = None):
    xmin, xmax = xlims

    fig = Figure()
    ax = fig.add_subplot(1, 1, 1)

    xs = np.linspace(xmin, xmax, 101)
    if len(transformList) == 0:
        print ('NOTE: no transforms being drawn for drawWarping with'
               ' outPdf = %s' % outPdf)
    for transform in transformList:
        ys = [ transform(x) for x in xs ]
        ax.plot(xs, ys, '-')
    if ylims is not None:
        ax.set_ylim(*ylims)
    if title is not None:
        ax.set_title(title)

    canvas = FigureCanvas(fig)
    canvas.print_figure(outPdf)

@codeDeps()
def drawLogPdf(outputs, bins, outPdf, fns = [], ylims = None, title = None):
    outputs = np.array(outputs)
    assert len(np.shape(outputs)) == 1
    counts, bins = np.histogram(outputs, bins = bins)
    avgPdfValues = counts * 1.0 / len(outputs) / np.diff(bins)
    binCentres = bins[:-1] + 0.5 * np.diff(bins)

    fig = Figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(binCentres, np.log(avgPdfValues))
    for f in fns:
        ax.plot(binCentres, [ f(x) for x in binCentres ])
    if title is not None:
        ax.set_title(title)
    ax.set_xlim(bins[0], bins[-1])
    if ylims is not None:
        ax.set_ylim(*ylims)

    canvas = FigureCanvas(fig)
    canvas.print_figure(outPdf)

@codeDeps(d.SynthMethod)
def drawFor1DInput(debugAcc, subDist, outPdf, xlims, ylims, title = None,
                   drawPlusMinusTwoStdev = False):
    """Draws plot showing several things of interest for the case of 1D input.

    For both debugAcc and subDist, input should be a 1D vector and output
    should be a scalar.
    """
    def subDrawScatter(ax, inputs, outputs):
        for input in inputs:
            if len(input) != 1:
                raise RuntimeError('input should be a vector of length 1,'
                                   ' but was %r' % input)
        for output in outputs:
            if np.shape(output) != ():
                raise RuntimeError('output should be a scalar,'
                                   ' but was %r' % output)
        if len(inputs) == 0:
            xs = np.zeros([0])
        else:
            xs = np.array(inputs)[:, 0]
        ys = np.array(outputs)
        ax.plot(xs, ys, '.', markersize = 0.2)

    def subDrawMeanish(ax, subDist, xlims, nx = 50,
                       drawPlusMinusTwoStdev = False,
                       numSamples = 200):
        xmin, xmax = xlims
        xs = np.linspace(xmin, xmax, nx + 1)
        ysMeanish = [
            subDist.synth(np.array([x]), method = d.SynthMethod.Meanish)
            for x in xs
        ]
        ax.plot(xs, ysMeanish, '-')
        if drawPlusMinusTwoStdev:
            ysSampleLow = []
            ysSampleMiddle = []
            ysSampleHigh = []
            for x in xs:
                samples = [
                    subDist.synth(np.array([x]), method = d.SynthMethod.Sample)
                    for i in range(numSamples)
                ]
                m = np.mean(samples)
                sd = np.std(samples)
                ysSampleLow.append(m - 2.0 * sd)
                ysSampleMiddle.append(m)
                ysSampleHigh.append(m + 2.0 * sd)
            ax.plot(xs, ysSampleMiddle, '-')
            ax.plot(xs, ysSampleLow, '-')
            ax.plot(xs, ysSampleHigh, '-')

    def subDrawPdfImage(ax, subDist, xlims, ylims, nx = 100, ny = 100):
        xmin, xmax = xlims
        ymin, ymax = ylims
        dx = (xmax - xmin) * 1.0 / nx
        dy = (ymax - ymin) * 1.0 / ny
        pdfValues = [ [ math.exp(subDist.logProb(np.array([x]), y))
                        for x in np.linspace(xmin, xmax, nx + 1) ]
                      for y in np.linspace(ymin, ymax, ny + 1) ]
        ax.imshow(
            pdfValues,
            extent = [xmin - 0.5 * dx, xmax - 0.5 * dx,
                      ymin - 0.5 * dy, ymax - 0.5 * dy],
            origin = 'lower',
            cmap = cm.gray,
            interpolation = 'nearest'
        )

    fig = Figure()
    ax = fig.add_subplot(1, 1, 1)

    subDrawScatter(ax, debugAcc.memo.inputs, debugAcc.memo.outputs)
    subDrawMeanish(ax, subDist, xlims,
                   drawPlusMinusTwoStdev = drawPlusMinusTwoStdev)
    subDrawPdfImage(ax, subDist, xlims, ylims)
    ax.set_xlim(*xlims)
    ax.set_ylim(*ylims)
    ax.set_xlabel('input')
    ax.set_ylabel('output')
    ax.grid(True)
    if title is not None:
        ax.set_title(title)

    canvas = FigureCanvas(fig)
    canvas.print_figure(outPdf)
