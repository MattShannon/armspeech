#!/usr/bin/python
"""A distutils-based script for distributing and installing armspeech."""

# Copyright 2011, 2012, 2013, 2014, 2015 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from distutils.core import setup

with open('README.markdown') as readmeFile:
    long_description = readmeFile.read()

setup(
    name = 'armspeech',
    version = '0.6.dev1',
    description = 'Autoregressive probabilistic modelling for speech synthesis.',
    url = 'http://github.com/MattShannon/armspeech',
    author = 'Matt Shannon',
    author_email = 'matt.shannon@cantab.net',
    license = 'various open source licenses (see License file)',
    packages = ['bisque', 'armspeech', 'armspeech.modelling', 'armspeech.speech', 'armspeech.util'],
    # (there are some scripts in the bin directory, but none of them are
    #   suitable for system-wide installation)
    scripts = [],
    # (not actually used for anything with distutils!)
    requires = ['numpy', 'scipy', 'matplotlib', 'codedep (>=0.3)'],
    provides = ['bisque', 'armspeech'],
    long_description = long_description,
)
