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

    # FIXME : this is broken in many ways. Firstly, an exception should
    #   probably be raised (or at least a warning printed) if there are any
    #   import errors. Currently all standard modules cause import errors due
    #   to the fact [envPythonPath] contains only the armspeech path. The path
    #   should probably be left blank instead, but this requires the returned
    #   list of module files to be pruned to only contain those that are below
    #   envPythonPath. Secondly, relative imports appear to fail in lots of
    #   circumstances (not sure exactly which). One fix to this problem would
    #   be to create a temporary file containing 'import armspeech.whatever'
    #   but this is a bit ugly and requires finding the fully-qualified module
    #   name (which is probably not hard). Alternatively it may be possible to
    #   use modulefinder in a better way (load_file is perhaps a bit better than
    #   run_script but still has the relative import problem). Finally, it would
    #   be nice for our use case to be able to use modulefinder on several files
    #   at once. Think we can just call run_script or load_file multiple times,
    #   but need to check.
    #
    #   All these issues will go away once we stop using findDeps (imminent).

    # FIXME : does sys.path[0] ever do anything here? Should just fail instead?
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
