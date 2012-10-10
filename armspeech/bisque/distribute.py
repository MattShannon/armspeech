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

@codeDeps(Artifact, codedep.getHash, persist.secHashObject)
class ThunkArtifact(Artifact):
    """An artifact whose value is computed fresh each time it is needed.

    thunk should be a callable function or class. If shouldCache is True, the
    thunk is evaluated only once for each process and the value cached.
    """
    def __init__(self, thunk, shouldCache = True):
        self.thunk = thunk
        self.shouldCache = shouldCache
    def parents(self):
        return []
    def parentArtifacts(self):
        return []
    def computeSecHash(self):
        secHashSource = codedep.getHash(self.__class__)
        secHashThunk = codedep.getHash(self.thunk)
        return persist.secHashObject((secHashSource, secHashThunk, self.shouldCache))
    def isDone(self, buildRepo):
        return True
    def loadValue(self, buildRepo):
        if self.shouldCache:
            if not hasattr(self, '_value'):
                self._value = self.thunk()
            return self._value
        else:
            return self.thunk()
    def saveValue(self, buildRepo, value):
        raise RuntimeError('a ThunkArtifact cannot be saved')

@codeDeps(Artifact, codedep.getHash, persist.secHashObject)
class FuncAppliedArtifact(Artifact):
    def __init__(self, func, argArts, kwargArts, shouldCache = True):
        self.func = func
        self.argArts = argArts
        self.kwargArts = kwargArts
        self.shouldCache = shouldCache

        self.inputs = list(argArts) + kwargArts.values()
    def parents(self):
        return []
    def parentArtifacts(self):
        return []
    def computeSecHash(self):
        secHashSource = codedep.getHash(self.__class__)
        secHashFunc = codedep.getHash(self.func)
        secHashInputs = [ art.secHash() for art in self.inputs ]
        return persist.secHashObject((secHashSource, secHashFunc, secHashInputs, self.shouldCache))
    def isDone(self, buildRepo):
        return all([ art.isDone(buildRepo) for art in self.inputs ])
    def loadValue(self, buildRepo):
        if self.shouldCache:
            if not hasattr(self, '_value'):
                args = [ argArt.loadValue(buildRepo) for argArt in self.argArts ]
                kwargs = dict([ (argKey, argArt.loadValue(buildRepo)) for argKey, argArt in self.kwargArts.items() ])
                self._value = self.func(*args, **kwargs)
            return self._value
        else:
            args = [ argArt.loadValue(buildRepo) for argArt in self.argArts ]
            kwargs = dict([ (argKey, argArt.loadValue(buildRepo)) for argKey, argArt in self.kwargArts.items() ])
            return self.func(*args, **kwargs)
    def saveValue(self, buildRepo, value):
        raise RuntimeError('a FuncAppliedArtifact cannot be saved')

@codeDeps(FuncAppliedArtifact)
def liftLocal(func, shouldCache = True):
    def argumentsToArt(*argArts, **kwargArts):
        return FuncAppliedArtifact(func, argArts, kwargArts, shouldCache = shouldCache)
    return argumentsToArt

@codeDeps(Artifact, codedep.getHash, persist.secHashObject)
class FixedDirArtifact(Artifact):
    """An artifact which is a fixed directory (or file).

    The hash of the artifact is computed based on the location only, not the
    contents. The artifact value is the location.
    """
    def __init__(self, location):
        self.location = location
    def parents(self):
        return []
    def parentArtifacts(self):
        return []
    def computeSecHash(self):
        secHashSource = codedep.getHash(self.__class__)
        return persist.secHashObject((secHashSource, self.location))
    def loc(self, buildRepo):
        return self.location
    def isDone(self, buildRepo):
        return os.path.exists(self.location)
    def loadValue(self, buildRepo):
        return self.location
    def saveValue(self, buildRepo, value):
        raise RuntimeError('a FixedDirArtifact cannot be saved')

@codeDeps(Artifact, codedep.getHash, persist.secHashFile, persist.secHashObject)
class FixedFileArtifact(Artifact):
    """An artifact which is a fixed file.

    The hash of the artifact is computed based on the contents of the file,
    not its location.

    This class needs to be subclassed further adding loadValue and saveValue
    methods.
    """
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
        return os.path.exists(self.location)

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
        secHashInputs = [ art.secHash() for art in self.inputs ]
        return persist.secHashObject((secHashSource, secHashInputs))
    def checkSecHash(self):
        if self.secHash() != self.computeSecHash():
            raise RuntimeError('secHash of job %s has changed' % self.name)

@codeDeps(Job, codedep.getHash, persist.secHashObject)
class WrappedFuncJob(Job):
    def __init__(self, func, argArts, kwargArts, name = None):
        if name is None:
            try:
                name = func.func_name
            except AttributeError:
                name = '<noname>'
        self.func = func
        self.argArts = argArts
        self.kwargArts = kwargArts
        self.name = name

        self.inputs = list(argArts) + kwargArts.values()
        self.valueOut = self.newOutput()
    def computeSecHash(self):
        secHashSource = codedep.getHash(self.__class__)
        secHashFunc = codedep.getHash(self.func)
        secHashInputs = [ art.secHash() for art in self.inputs ]
        return persist.secHashObject((secHashSource, secHashFunc, secHashInputs))
    def run(self, buildRepo):
        args = [ argArt.loadValue(buildRepo) for argArt in self.argArts ]
        kwargs = dict([ (argKey, argArt.loadValue(buildRepo)) for argKey, argArt in self.kwargArts.items() ])

        valueOut = self.func(*args, **kwargs)

        self.valueOut.saveValue(buildRepo, valueOut)

@codeDeps(WrappedFuncJob)
def lift(func, name = None):
    """Lifts a function, allowing easy creation of jobs.

    func takes a collection of values and returns a value. lift(func) takes a
    corresponding set of artifacts and returns an artifact. A job is created
    which computes the value of the output artifact by applying func to the
    values of the input artifacts.

    This makes it easy to create distributable jobs from functions.

    For example:

        @codeDeps()
        def one():
            return 1

        @codeDeps()
        def add(x, y):
            return x + y

        oneArt = lift(one)()
        twoArt = lift(add)(oneArt, y = oneArt)

    Here twoArt is the output artifact of a job which has been created.
    """
    def argumentsToArt(*argArts, **kwargArts):
        job = WrappedFuncJob(func, argArts, kwargArts, name = name)
        return job.valueOut
    return argumentsToArt
