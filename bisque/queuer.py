"""Basic definitions for executing distributed computations."""

# Copyright 2011, 2012, 2013 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

from bisque import persist
from codedep import codeDeps, ForwardRef

import os
import random
import tempfile
import time

# (FIXME : presumably there are lots of race issues w.r.t. artifact and job
#   writing and updating status on disk, both below and perhaps elsewhere in
#   bisque. Not yet sure how I want to tackle this general issue, so for now
#   just leave something that usually works.)

@codeDeps()
def createDir(path):
    try:
        os.mkdir(path)
    except OSError, e:
        # (FIXME : is error number cross-platform or specific to posix?)
        if e.errno != 17:
            raise

@codeDeps(ForwardRef(lambda: LiveJob), createDir, persist.loadPickle,
    persist.savePickle, persist.secHashObject
)
class BuildRepo(object):
    def __init__(self, base, createDirsIfNece = True):
        self.base = base
        if createDirsIfNece:
            createDir(self.base)
            createDir(os.path.join(self.base, 'cache'))
            createDir(os.path.join(self.base, 'liveJobs'))
    def cacheDir(self):
        return os.path.join(self.base, 'cache')
    def getJobQueuerDir(self, job, queuer):
        jobQueuerId = persist.secHashObject((job.secHash(), queuer.secHash()))
        return os.path.join(self.base, 'liveJobs', jobQueuerId)
    def liveJobs(self, job, queuer):
        jobQueuerDir = self.getJobQueuerDir(job, queuer)
        liveJobIds = os.listdir(jobQueuerDir) if os.path.isdir(jobQueuerDir) else []
        return [ LiveJob(os.path.join(jobQueuerDir, liveJobId)) for liveJobId in liveJobIds ]
    def createLiveJob(self, job, queuer):
        jobSecHash = job.secHash()
        jobQueuerDir = self.getJobQueuerDir(job, queuer)
        createDir(jobQueuerDir)
        liveJobDir = tempfile.mkdtemp(prefix = 'liveJob.', dir = jobQueuerDir)
        liveJob = LiveJob(liveJobDir)
        persist.savePickle(os.path.join(liveJob.dir, 'job.pickle'), job)
        liveJob.setExtra('secHash', jobSecHash)
        persist.savePickle(os.path.join(liveJob.dir, 'buildRepo.pickle'), self)
        liveJob.setStored()

        job.checkAllSecHash()
        assert self.getJobQueuerDir(job, queuer) == jobQueuerDir
        jobAgain = persist.loadPickle(os.path.join(liveJob.dir, 'job.pickle'))
        job.checkAllSecHash()

        return liveJob

@codeDeps()
class LiveJob(object):
    def __init__(self, dir):
        self.dir = dir
    def status(self):
        statusFile = os.path.join(self.dir, 'status')
        if not os.path.exists(statusFile):
            return -1
        else:
            return int(file(os.path.join(self.dir, 'status')).read().strip())
    def setStatus(self, status):
        statusFile = os.path.join(self.dir, 'status')
        statusFileNew = os.path.join(
            self.dir,
            '.status.%s' % random.randint(0, 1000000)
        )
        with open(statusFileNew, 'w') as f:
            f.write(str(status)+'\n')
        os.rename(statusFileNew, statusFile)
    def setStored(self):
        self.setStatus(0)
    def setSubmitted(self):
        assert self.status() == 0
        self.setStatus(1)
    def setRunning(self):
        assert self.status() == 1
        self.setStatus(2)
    def setCompleted(self):
        assert self.status() == 2
        self.setStatus(3)
    def setError(self):
        assert self.status() == 2
        self.setStatus(10)
    def hasCompleted(self):
        status = self.status()
        return status == 3
    def hasError(self):
        status = self.status()
        return status == 10
    def hasEnded(self):
        status = self.status()
        return status == 3 or status == 10
    def extra(self, key):
        slurped = file(os.path.join(self.dir, 'extra_'+key), 'rb').read()
        assert slurped[-1] == '\n'
        return slurped[:-1]
    def setExtra(self, key, value):
        with open(os.path.join(self.dir, 'extra_'+key), 'wb') as f:
            f.write(value+'\n')

@codeDeps()
class Queuer(object):
    def secHash(self):
        return self.secHashUid

    def submitAll(self, job, live, verbosity):
        if job.secHash() not in live:
            for inputArt in job.inputs:
                self.generateArtifact(inputArt, live, verbosity)
            runningLiveJobs = [ liveJob for liveJob in self.buildRepo.liveJobs(job, queuer = self) if not liveJob.hasEnded() ]
            if runningLiveJobs:
                live[job.secHash()] = runningLiveJobs[0]
                if verbosity >= 1:
                    print 'queuer: not submitting job', job.secHash(), 'since already submitted as', runningLiveJobs[0].dir
            else:
                live[job.secHash()] = self.submitOne(job, live, verbosity)

    def generateArtifact(self, art, live, verbosity):
        if not art.isDone(self.buildRepo):
            for job in art.parents():
                self.submitAll(job, live, verbosity)

    def generateArtifacts(self, finalArtifacts, verbosity = 1):
        live = dict()
        for art in finalArtifacts:
            self.generateArtifact(art, live, verbosity)
        if verbosity >= 1:
            print 'queuer: final artifacts will be at:'
            for art in finalArtifacts:
                print 'queuer:     '+art.loc(self.buildRepo)
        return live

@codeDeps(Queuer, persist.secHashObject)
class LocalQueuer(Queuer):
    def __init__(self, buildRepo):
        self.buildRepo = buildRepo

        self.secHashUid = persist.secHashObject(id(self))

    def submitOne(self, job, live, verbosity):
        job.run(self.buildRepo)
