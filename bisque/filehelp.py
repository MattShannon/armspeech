"""Filesystem helper functions."""

# Copyright 2011, 2012, 2013 Matt Shannon

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
    def __init__(self, prefix = 'bisque.', removeOnException = False):
        self.prefix = prefix
        self.removeOnException = removeOnException

    def __enter__(self):
        self.location = tempfile.mkdtemp(prefix = self.prefix)
        return self

    def remove(self):
        shutil.rmtree(self.location)

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None or self.removeOnException:
            self.remove()
        if os.path.isdir(self.location):
            logging.warning('temporary directory %s not deleted. You probably'
                            ' want to do this manually after looking at its'
                            ' contents.' % self.location)
