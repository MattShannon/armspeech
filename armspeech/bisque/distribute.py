"""Basic definitions for specifying distributed computations."""

# Copyright 2011, 2012 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

from armspeech.util import persist
from codedep import codeDeps, ForwardRef

import os
import sys
import inspect
import modulefinder

# FIXME : the hash of an artifact currently only incorporates the source for the
#   class of the parent job (and everything that class definition could possibly
#   depend on). This has the undesirable consequence that two different
#   computations with different results may be stored in the same place (since
#   if job JA produces artifact A, and A is used as input for job JB producing
#   artifact B, then changing the code in JA may affect the hash of A and the
#   value of both A and B without affecting the hash of B. For example, if we
#   change a numerical constant in a method of JA, this does not affect anything
#   in the pickle of the artifact B which is used to compute the hash of B, but
#   may affect the value of A and B.)
#   This bug needs to be fixed.

@codeDeps()
def findDeps(srcFile):
    """Returns local dependencies for a given source file.

    Local means defined under PYTHONPATH (or in the same directory as the given
    source file if PYTHONPATH not defined), the idea being that local files are
    the ones subject to change and need to be hashed.
    It is assumed that modules that are on the system search path are fixed.
    """
    envPythonPath = os.environ['PYTHONPATH'] if 'PYTHONPATH' in os.environ else sys.path[0]
    finder = modulefinder.ModuleFinder(path = [envPythonPath])
    finder.run_script(srcFile)
    depFiles = [ mod.__file__ for modName, mod in finder.modules.items() if mod.__file__ is not None ]
    return sorted(depFiles)

@codeDeps(ForwardRef(lambda: ancestorArtifacts), persist.secHashObject)
class Artifact(object):
    def secHash(self):
        secHashAllExternals = [ secHash for art in ancestorArtifacts([self]) for secHash in art.secHashExternals() ]
        return persist.secHashObject((self, secHashAllExternals, self.secHashSources()))

@codeDeps(Artifact, persist.secHashFile)
class FixedArtifact(Artifact):
    def __init__(self, location):
        self.location = location
    def parentJobs(self):
        return []
    def parentArtifacts(self):
        return []
    def secHashExternals(self):
        return [persist.secHashFile(self.location)]
    def secHashSources(self):
        return []
    def loc(self, baseDir):
        return self.location

@codeDeps(Artifact, findDeps, persist.secHashFile)
class JobArtifact(Artifact):
    def __init__(self, parentJob):
        self.parentJob = parentJob
    def parentJobs(self):
        return [self.parentJob]
    def parentArtifacts(self):
        return self.parentJob.inputs
    def secHashExternals(self):
        return []
    def secHashSources(self):
        return [ persist.secHashFile(depFile) for depFile in findDeps(inspect.getsourcefile(self.parentJob.__class__)) ]
    def loc(self, baseDir):
        return os.path.join(baseDir, self.secHash())

@codeDeps()
def ancestorArtifacts(initialArts):
    ret = []
    agenda = list(initialArts)
    lookup = dict()
    while agenda:
        art = agenda.pop()
        ident = id(art)
        if not ident in lookup:
            lookup[ident] = True
            ret.append(art)
            agenda.extend(reversed(art.parentArtifacts()))
    return ret

@codeDeps(JobArtifact, ancestorArtifacts, persist.secHashObject)
class Job(object):
    # (N.B. some client code uses default hash routines)
    def parentJobs(self):
        return [ parentJob for input in self.inputs for parentJob in input.parentJobs() ]
    def newOutput(self):
        return JobArtifact(parentJob = self)
    def run(self, buildRepo):
        abstract
    def secHash(self):
        secHashAllExternals = [ secHash for art in ancestorArtifacts(self.inputs) for secHash in art.secHashExternals() ]
        return persist.secHashObject((self, secHashAllExternals))
