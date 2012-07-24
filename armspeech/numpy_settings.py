"""Sets default global numpy options for armspeech.

This module is intended to be imported whenever numpy is (within armspeech).
The defaults specified below may be overridden in client applications by calling
np.seterr, etc as usual, but importing this module whenever numpy is ensures
consistency within armspeech when the calling script / test runner does not
explicitly call np.seterr, etc.
"""

# Copyright 2011, 2012 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.

from __future__ import division

import numpy as np

np.seterr(all = 'ignore')
np.set_printoptions(precision = 17)
