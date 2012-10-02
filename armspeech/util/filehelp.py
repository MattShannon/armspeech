"""Filesystem helper functions."""

# Copyright 2011, 2012 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

from codedep import codeDeps

import os
import logging
import shutil
import tempfile

@codeDeps()
class TempDir(object):
    def __init__(self):
        self.location = tempfile.mkdtemp(prefix = 'armspeech.')
    def remove(self):
        shutil.rmtree(self.location)
    def __del__(self):
        if os.path.isdir(self.location):
            logging.warning('temporary directory '+self.location+' not deleted. You probably want to do this manually after looking at its contents.')
