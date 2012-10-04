"""Basic definitions for specifying distributed computations."""

# Copyright 2011, 2012 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

import codedep
from armspeech.util import persist
from codedep import codeDeps, ForwardRef

import os

@codeDeps(ForwardRef(lambda: ancestors))
class DagNode(object):
    def secHash(self):
        if not hasattr(self, '_secHash'):
            self._secHash = self.computeSecHash()
        return self._secHash
    def checkAllSecHash(self):
        for node in ancestors([self]):
            node.checkSecHash()

@codeDeps()
def ancestors(initialNodes):
    ret = []
    agenda = list(initialNodes)
    lookup = dict()
    while agenda:
        node = agenda.pop()
        ident = id(node)
        if not ident in lookup:
            lookup[ident] = True
            ret.append(node)
            agenda.extend(reversed(node.parents()))
    return ret

@codeDeps(DagNode)
class Artifact(DagNode):
    def checkSecHash(self):
        if self.secHash() != self.computeSecHash():
            raise RuntimeError('secHash of artifact %s has changed' % self)

@codeDeps(Artifact, codedep.getHash, persist.secHashFile, persist.secHashObject)
class FixedArtifact(Artifact):
    def __init__(self, location):
        self.location = location
    def parents(self):
        return []
    def parentArtifacts(self):
        return []
    def computeSecHash(self):
        secHashSource = codedep.getHash(self.__class__)
        secHashFile = persist.secHashFile(self.location)
        return persist.secHashObject((secHashSource, secHashFile))
    def loc(self, buildRepo):
        return self.location
    def isDone(self, buildRepo):
        return os.path.exists(self.loc(buildRepo))

@codeDeps(Artifact, codedep.getHash, persist.loadPickle, persist.savePickle,
    persist.secHashObject
)
class JobArtifact(Artifact):
    def __init__(self, parentJob):
        self.parentJob = parentJob
    def parents(self):
        return [self.parentJob]
    def parentArtifacts(self):
        return self.parentJob.inputs
    def computeSecHash(self):
        secHashSource = codedep.getHash(self.__class__)
        return persist.secHashObject((secHashSource, self.parentJob.secHash()))
    def loc(self, buildRepo):
        return os.path.join(buildRepo.cacheDir(), self.secHash())
    def isDone(self, buildRepo):
        return os.path.exists(self.loc(buildRepo))
    def loadValue(self, buildRepo):
        return persist.loadPickle(self.loc(buildRepo))
    def saveValue(self, buildRepo, value):
        return persist.savePickle(self.loc(buildRepo), value)

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

@codeDeps(DagNode, JobArtifact, codedep.getHash, persist.secHashObject)
class Job(DagNode):
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
    def parents(self):
        return self.inputs
    def parentJobs(self):
        return [ parentJob for input in self.inputs for parentJob in input.parents() ]
    def newOutput(self):
        return JobArtifact(parentJob = self)
    def run(self, buildRepo):
        abstract
    def computeSecHash(self):
        secHashSource = codedep.getHash(self.__class__)
        return persist.secHashObject((secHashSource, [ input.secHash() for input in self.inputs ]))
    def checkSecHash(self):
        if self.secHash() != self.computeSecHash():
            raise RuntimeError('secHash of job %s has changed' % self.name)
