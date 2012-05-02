"""Draw graphs of trajectories, etc."""

# Copyright 2011, 2012 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

import armspeech.modelling.dist as d

import math
import numpy as np
import matplotlib
import matplotlib.transforms as transforms

def partitionSeq(xs, numPartitions):
    out = [ [] for i in range(numPartitions) ]
    for i, x in enumerate(xs):
        out[i % numPartitions].append(x)
    return out

def drawLabelledSeq(dataSeqs, labelSeqs, outPdf, figSizeRate = None, fillBetween = [], xmin = None, xmax = None, ylims = None, xlabel = None, ylabel = None, legend = None, colors = ['red', 'purple', 'orange', 'blue']):
    import matplotlib.pyplot as plt

    if xmin is None:
        xmin = min([ dataSeq[0][0] for dataSeq in dataSeqs ] + [ labelSeq[0][0] for labelSeq in labelSeqs ])
    if xmax is None:
        xmax = max([ dataSeq[0][-1] for dataSeq in dataSeqs ] + [ labelSeq[-1][1] for labelSeq in labelSeqs ])

    if figSizeRate is None:
        fig = plt.figure()
    else:
        fig = plt.figure(figsize = ((xmax - xmin) * figSizeRate, 6.0), dpi = 300.0)
    ax = fig.add_subplot(1, 1, 1)

    for x, y1, y2 in fillBetween:
        ax.fill_between(x, y1, y2, color = 'blue', facecolor = 'blue', linewidth = 0.2, alpha = 0.2)
    for x, y in dataSeqs:
        ax.plot(x, y, '-')

    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)

    for labelSeqIndex, labelSeq in enumerate(labelSeqs):
        color = colors[labelSeqIndex % len(colors)]
        height = labelSeqIndex * 0.04
        for start, end, label in labelSeq:
            pos = (start + end) / 2.0
            if xmin < pos < xmax:
                ax.text(pos, height + 0.02, label, transform = trans, horizontalalignment = 'center', verticalalignment = 'center', fontsize = 8)
            ax.axvspan(xmin = start, xmax = end, ymin = height + 0.0, ymax = height + 0.04, facecolor = color, alpha = 0.5)
            ax.axvline(x = start, linewidth = 0.1, color = 'black')

    ax.set_xlim(xmin, xmax)
    if ylims is not None:
        ax.set_ylim(*ylims)

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if legend is not None:
        ax.legend(legend)

    plt.savefig(outPdf)

def drawWarping(transformList, outPdf, xlims, ylims = None, title = None):
    import matplotlib.pyplot as plt

    xmin, xmax = xlims
    plt.figure()
    xs = np.linspace(xmin, xmax, 101)
    if len(transformList) == 0:
        print 'NOTE: no transforms being drawn for drawWarping with outPdf =', outPdf
    for transform in transformList:
        ys = [ transform(x) for x in xs ]
        plt.plot(xs, ys, '-')
    if ylims is not None:
        plt.ylim(*ylims)
    if title is not None:
        plt.title(title)

    plt.savefig(outPdf)

def drawLogPdf(outputs, bins, outPdf, fns = [], ylims = None, title = None):
    import matplotlib.pyplot as plt

    outputs = np.array(outputs)
    assert len(np.shape(outputs)) == 1
    counts, bins = np.histogram(outputs, bins = bins)
    avgPdfValues = counts * 1.0 / len(outputs) / np.diff(bins)
    binCentres = bins[:-1] + 0.5 * np.diff(bins)

    plt.figure()
    plt.plot(binCentres, np.log(avgPdfValues))
    for f in fns:
        plt.plot(binCentres, [ f(x) for x in binCentres ])
    if title is not None:
        plt.title(title)
    plt.xlim(bins[0], bins[-1])
    if ylims is not None:
        plt.ylim(*ylims)
    plt.savefig(outPdf)

def drawFor1DInput(debugAcc, subDist, outPdf, xlims, ylims, title = None, drawPlusMinusTwoStdev = False):
    """Draws plot showing several things of interest for the case of 1D input.

    For both debugAcc and subDist, input should be a 1D vector and output
    should be a scalar.
    """
    import matplotlib.pyplot as plt

    def subDrawScatter(inputs, outputs):
        for input in inputs:
            if len(input) != 1:
                raise RuntimeError('input should be a vector of length 1, but was '+repr(input))
        for output in outputs:
            if np.shape(output) != ():
                raise RuntimeError('output should be a scalar, but was '+repr(output))
        if len(inputs) == 0:
            xs = np.zeros([0])
        else:
            xs = np.array(inputs)[:, 0]
        ys = np.array(outputs)
        plt.plot(xs, ys, '.', markersize = 0.2)

    def subDrawMeanish(subDist, xlims, nx = 50, drawPlusMinusTwoStdev = False, numSamples = 200):
        xmin, xmax = xlims
        xs = np.linspace(xmin, xmax, nx + 1)
        ysMeanish = [ subDist.synth(np.array([x]), method = d.SynthMethod.Meanish) for x in xs ]
        plt.plot(xs, ysMeanish, '-')
        if drawPlusMinusTwoStdev:
            ysSampleLow = []
            ysSampleMiddle = []
            ysSampleHigh = []
            for x in xs:
                samples = [ subDist.synth(np.array([x]), method = d.SynthMethod.Sample) for i in range(numSamples) ]
                m = np.mean(samples)
                sd = np.std(samples)
                ysSampleLow.append(m - 2.0 * sd)
                ysSampleMiddle.append(m)
                ysSampleHigh.append(m + 2.0 * sd)
            plt.plot(xs, ysSampleMiddle, '-')
            plt.plot(xs, ysSampleLow, '-')
            plt.plot(xs, ysSampleHigh, '-')

    def subDrawPdfImage(subDist, xlims, ylims, nx = 100, ny = 100):
        xmin, xmax = xlims
        ymin, ymax = ylims
        dx = (xmax - xmin) / nx
        dy = (ymax - ymin) / ny
        pdfValues = [[ math.exp(subDist.logProb(np.array([x]), y)) for x in np.linspace(xmin, xmax, nx + 1) ] for y in np.linspace(ymin, ymax, ny + 1) ]
        plt.imshow(pdfValues, extent = [xmin - 0.5 * dx, xmax - 0.5 * dx, ymin - 0.5 * dy, ymax - 0.5 * dy], origin = 'lower', cmap = plt.cm.gray, interpolation = 'nearest')

    plt.figure()
    subDrawScatter(debugAcc.memo.inputs, debugAcc.memo.outputs)
    subDrawMeanish(subDist, xlims, drawPlusMinusTwoStdev = drawPlusMinusTwoStdev)
    subDrawPdfImage(subDist, xlims, ylims)
    plt.xlim(*xlims)
    plt.ylim(*ylims)
    plt.xlabel('input')
    plt.ylabel('output')
    plt.grid(True)
    if title is not None:
        plt.title(title)
    plt.savefig(outPdf)
