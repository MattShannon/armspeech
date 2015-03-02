"""Basic definitions for specifying distributed computations."""

# Copyright 2011, 2012, 2013, 2014, 2015 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

import codedep
from bisque import persist
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

# (FIXME : unused? Remove?)
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
    def __init__(self, func, argArts, kwargArts, indexOut = None,
                 shouldCache = True):
        self.func = func
        self.argArts = argArts
        self.kwargArts = kwargArts
        self.indexOut = indexOut
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
        return persist.secHashObject((secHashSource, secHashFunc,
                                      secHashInputs, self.indexOut,
                                      self.shouldCache))

    def isDone(self, buildRepo):
        return all([ art.isDone(buildRepo) for art in self.inputs ])

    def computeValue(self, buildRepo):
        args = [ argArt.loadValue(buildRepo) for argArt in self.argArts ]
        kwargs = dict([ (argKey, argArt.loadValue(buildRepo))
                        for argKey, argArt in self.kwargArts.items() ])
        if self.indexOut is None:
            return self.func(*args, **kwargs)
        else:
            return self.func(*args, **kwargs)[self.indexOut]

    def loadValue(self, buildRepo):
        if self.shouldCache:
            if not hasattr(self, '_value'):
                self._value = self.computeValue(buildRepo)
            return self._value
        else:
            return self.computeValue(buildRepo)

    def saveValue(self, buildRepo, value):
        raise RuntimeError('a FuncAppliedArtifact cannot be saved')

@codeDeps(FuncAppliedArtifact)
def liftLocal(func, numOut = None, shouldCache = True):
    def argumentsToArt(*argArts, **kwargArts):
        if numOut is None:
            return FuncAppliedArtifact(func, argArts, kwargArts,
                                       shouldCache = shouldCache)
        else:
            return [ FuncAppliedArtifact(func, argArts, kwargArts,
                                         indexOut = indexOut,
                                         shouldCache = shouldCache)
                     for indexOut in range(numOut) ]
    return argumentsToArt

@codeDeps(Artifact, codedep.getHash, persist.secHashObject)
class LiteralArtifact(Artifact):
    """An artifact which is a literal value.

    In general the preferred approach to encoding constant values as artifacts
    is to explicitly define a module-level function which returns the constant
    value and wrap this using liftLocal to provide an artifact. However doing
    this for all literal constants is tedious. This class provides a less
    verbose alternative when the value of the artifact is a literal which is
    picklable (and for which the pickle will not change across different runs).

    Example usage:

        LiteralValue(2)
        LiteralValue('the')

    but not:

        LiteralValue(<call to some function which returns the value 2>)
        LiteralValue(<an instance of a class>)

    (In some cases the bad examples above will be fine, but haven't worked out
    full details so for now using this form is not recommended.)
    """
    def __init__(self, litValue):
        self.litValue = litValue

    def parents(self):
        return []

    def parentArtifacts(self):
        return []

    def computeSecHash(self):
        secHashSource = codedep.getHash(self.__class__)
        return persist.secHashObject((secHashSource, self.litValue))

    def isDone(self, buildRepo):
        return True

    def loadValue(self, buildRepo):
        return self.litValue

    def saveValue(self, buildRepo, value):
        raise RuntimeError('a LiteralArtifact cannot be saved')

@codeDeps(LiteralArtifact)
def lit(value):
    """Constructs LiteralArtifacts (and has a short name).

    For example:

        theArt = lit('the')
    """
    return LiteralArtifact(value)

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

@codeDeps(Artifact)
class LazySeqArtifact(Artifact):
    """A wrapper for sequence-valued artifacts allowing lazy loading.

    This (slightly hacky) class provides a wrapper around an underlying
    sequence-valued artifact.
    The artifact value for this class is an iterator representing the same
    sequence as the underlying value.
    When loadValue is called for this artifact an iterator is returned, but the
    underlying value is only computed when the first element of the iterator is
    accessed.
    The underlying value is then stored in memory to allow it to be used for
    the remainder of the lifetime of the iterator.
    This class therefore provides a way for a given sequence-valued artifact to
    be loaded (e.g. unpickled) only when it is needed.
    """
    def __init__(self, seqArt):
        self.seqArt = seqArt
    def parents(self):
        return self.seqArt.parents()
    def parentArtifacts(self):
        return self.seqArt.parentArtifacts()
    def computeSecHash(self):
        return self.seqArt.computeSecHash()
    def loc(self, buildRepo):
        return self.seqArt.loc(buildRepo)
    def isDone(self, buildRepo):
        return self.seqArt.isDone(buildRepo)
    def loadValue(self, buildRepo):
        def getValueNew():
            for elem in self.seqArt.loadValue(buildRepo):
                yield elem
        return getValueNew()
    def saveValue(self, buildRepo, value):
        return self.seqArt.saveValue(buildRepo, value)

@codeDeps(LazySeqArtifact)
def lazySeq(seqArt):
    return LazySeqArtifact(seqArt)

@codeDeps(Artifact, codedep.getHash, persist.loadPickle, persist.savePickle,
    persist.secHashObject
)
class JobArtifact(Artifact):
    """An artifact which is the output of a Job.

    The secHash of this artifact depends on indexOut.
    """
    def __init__(self, parentJob, indexOut = None):
        self.parentJob = parentJob
        self.indexOut = indexOut
    def parents(self):
        return [self.parentJob]
    def parentArtifacts(self):
        return self.parentJob.inputs
    def computeSecHash(self):
        secHashSource = codedep.getHash(self.__class__)
        return persist.secHashObject((secHashSource, self.parentJob.secHash(),
                                      self.indexOut))
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

@codeDeps(DagNode, codedep.getHash, persist.secHashObject)
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

    Note that if a job has multiple outputs it is important to ensure that each
    output JobArtifact has a distinct indexOut. This is to ensure that the
    secHashes of the outputs are distinct.

    Also note when subclassing that you should not break __hash__, since some
    of the queuer code calls this method.
    """
    def parents(self):
        return self.inputs
    def parentJobs(self):
        return [ parentJob for input in self.inputs for parentJob in input.parents() ]
    def run(self, buildRepo):
        abstract
    def computeSecHash(self):
        secHashSource = codedep.getHash(self.__class__)
        secHashInputs = [ art.secHash() for art in self.inputs ]
        return persist.secHashObject((secHashSource, secHashInputs))
    def checkSecHash(self):
        if self.secHash() != self.computeSecHash():
            raise RuntimeError('secHash of job %s has changed' % self.name)

@codeDeps(Job, JobArtifact, codedep.getHash, persist.secHashObject)
class WrappedFuncJob(Job):
    def __init__(self, func, argArts, kwargArts, numOut = None, name = None):
        if name is None:
            try:
                name = func.func_name
            except AttributeError:
                name = '<noname>'
        self.func = func
        self.argArts = argArts
        self.kwargArts = kwargArts
        self.numOut = numOut
        self.name = name

        self.inputs = list(argArts) + kwargArts.values()
        if self.numOut is None:
            self.valueOut = JobArtifact(parentJob = self)
        else:
            self.valuesOut = [ JobArtifact(parentJob = self,
                                           indexOut = indexOut)
                               for indexOut in range(self.numOut) ]

    def computeSecHash(self):
        secHashSource = codedep.getHash(self.__class__)
        secHashFunc = codedep.getHash(self.func)
        secHashInputs = [ art.secHash() for art in self.inputs ]
        return persist.secHashObject((secHashSource, secHashFunc,
                                      secHashInputs, self.numOut))

    def run(self, buildRepo):
        args = [ argArt.loadValue(buildRepo) for argArt in self.argArts ]
        kwargs = dict([ (argKey, argArt.loadValue(buildRepo))
                        for argKey, argArt in self.kwargArts.items() ])

        if self.numOut is None:
            valueOut = self.func(*args, **kwargs)
            self.valueOut.saveValue(buildRepo, valueOut)
        else:
            valuesOut = self.func(*args, **kwargs)
            assert len(valuesOut) == self.numOut
            for valueOut, selfValueOut in zip(valuesOut, self.valuesOut):
                selfValueOut.saveValue(buildRepo, valueOut)

@codeDeps(WrappedFuncJob)
def lift(func, numOut = None, name = None):
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

    If numOut is not None then the output of func should be a sequence (e.g. a
    list or a tuple) of length numOut, and the lifted function produces a
    corresponding sequence of artifacts. This makes it easy to create
    distributable jobs with multiple outputs. For example:

        @codeDeps()
        def outputTuple():
            return 'a', 'b'

        aArt, bArt = lift(outputTuple, numOut = 2)()

    Using lift inside a for comprehension makes it easy to create a set of jobs
    effecting a "map" operation, where each value in a sequence of inputs is
    acted on by a common function, producing a corresponding sequence of
    outputs. For example:

        @codeDeps()
        def one():
            return 1

        @codeDeps()
        def getSeq():
            return [5, 6]

        @codeDeps()
        def add(x, y):
            return x + y

        oneArt = lift(one)()
        xArts = lift(getSeq, numOut = 2)()
        zArts = [ lift(add)(xArt, oneArt) for xArt in xArts ]
    """
    def argumentsToArt(*argArts, **kwargArts):
        job = WrappedFuncJob(func, argArts, kwargArts, numOut = numOut, name = name)
        if numOut is None:
            return job.valueOut
        else:
            return job.valuesOut
    return argumentsToArt
