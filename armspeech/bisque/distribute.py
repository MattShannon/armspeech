"""Basic definitions for specifying distributed computations."""

# Copyright 2011, 2012 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

import codedep
from armspeech.util import persist
from codedep import codeDeps, ForwardRef

import os
import sys
import inspect
import modulefinder

@codeDeps(ForwardRef(lambda: ancestorArtifacts), persist.secHashObject)
class Artifact(object):
    def secHash(self):
        secHashAllExternals = [ secHash for art in ancestorArtifacts([self]) for secHash in art.secHashExternals() ]
        secHashAllSources = [ secHash for art in ancestorArtifacts([self]) for secHash in art.secHashSources() ]
        return persist.secHashObject((self, secHashAllExternals, secHashAllSources))

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

@codeDeps(Artifact, codedep.getHash)
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
        return [codedep.getHash(self.parentJob.__class__)]
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

@codeDeps(JobArtifact, ancestorArtifacts, codedep.getHash,
    persist.secHashObject
)
class Job(object):
    """A job specifies a computation to be run on some input.

    This class is intended to be subclassed. The subclass should set
    self.inputs to the list of artifacts this job depends on during object
    initialization.

    Any changes to the source code generating the artifacts in self.inputs will
    be automatically detected and will be reflected in the hash value for this
    job and its output artifact. However the source code used to generate
    objects in the job's dictionary but not in self.inputs is *not*
    incorporated into the hash. This means that any objects used to construct
    this job which have even the remotest possibility of changing in the future
    should be included explicitly as artifacts in self.inputs. The recommended
    use is to make everything passed to the constructor except the job name
    part of self.inputs.

    Also note when subclassing that you should not break __hash__, since some
    of the queuer code calls this method.
    """
    def parentJobs(self):
        return [ parentJob for input in self.inputs for parentJob in input.parentJobs() ]
    def newOutput(self):
        return JobArtifact(parentJob = self)
    def run(self, buildRepo):
        abstract
    def secHash(self):
        secHashSource = codedep.getHash(self.__class__)
        secHashAllExternals = [ secHash for art in ancestorArtifacts(self.inputs) for secHash in art.secHashExternals() ]
        secHashAllSources = [ secHash for art in ancestorArtifacts(self.inputs) for secHash in art.secHashSources() ]
        return persist.secHashObject((self, secHashSource, secHashAllExternals, secHashAllSources))
