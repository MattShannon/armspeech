"""Helper functions for sequence types."""

# Copyright 2011, 2012 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

from collections import deque

def contextualizeIter(contextLength, iter, fillFrames = []):
    assert len(fillFrames) <= contextLength
    context = deque(fillFrames)
    for curr in iter:
        yield list(context), curr
        context.append(curr)
        if len(context) > contextLength:
            context.popleft()

def chunkList(xs, numChunks):
    """Splits list into roughly evenly-sized chunks."""
    n = len(xs)
    return [ xs[((n * chunkIndex) // numChunks):((n * (chunkIndex + 1)) // numChunks)] for chunkIndex in range(numChunks) ]
