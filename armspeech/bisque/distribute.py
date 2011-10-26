"""Simple distributed computation framework."""

# Copyright 2011 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

import persist

import os
import sys
import traceback
from datetime import datetime

class ExecContext(object):
    def generateArtifacts(self, finalArtifacts, verbosity = 1):
        abstract

class LocalExecContext(ExecContext):
    def __init__(self, repo):
        self.repo = repo

    def submitAll(self, job, submitted):
        if job not in submitted:
            for parentJob in job.parentJobsToRun():
                self.submitAll(parentJob, submitted)
            self.submitOne(job)
            submitted.add(job)

    def generateArtifacts(self, finalArtifacts, verbosity = 1):
        submitted = set()
        for art in finalArtifacts:
            for parentJob in getParentJobs(art):
                self.submitAll(parentJob, submitted)
        if verbosity >= 1:
            print 'distribute: final artifacts will be at:'
            for art in finalArtifacts:
                print 'distribute:     '+art.location

    def submitOne(self, job):
        job.run()

class Job(object):
    # (N.B. some client code uses default hash routines)
    def parentJobs(self):
        return [ parentJob for input in self.inputs for parentJob in getParentJobs(input) ]
    def parentJobsToRun(self):
        return [ parentJob for input in self.inputs for parentJob in getParentJobs(input) if not os.path.exists(input.location) ]
    def newOutput(self):
        return JobArtifact(self.repo.newLocation(), self)
    def run(self):
        abstract
    def ancestorJobs(self):
        return [ job for parentJob in self.parentJobs() for job in parentJob.ancestorJobs() ] + [self]
    def ancestorJobsToRun(self):
        return [ job for parentJob in self.parentJobsToRun() for job in parentJob.ancestorJobsToRun() ] + [self]
    def ancestorArtifacts(self):
        return self.inputs + [ art for parentJob in self.parentJobs() for art in parentJob.ancestorArtifacts() ]

class JobArtifact(persist.Artifact):
    def __init__(self, location, parentJob):
        self.location = location
        self.parentJob = parentJob
    def ancestorJobs(self):
        return self.parentJob.ancestorJobs()
    def ancestorArtifacts(self):
        return self.parentJob.ancestorArtifacts() + [self]

def getParentJobs(art):
    try:
        parentJob = art.parentJob
    except AttributeError:
        ret = []
    else:
        ret = [parentJob]
    return ret

def main(args):
    assert len(args) == 2
    jobLocation = args[1]
    job = persist.loadPickle(jobLocation)
    print 'distribute: job', job.name, '(', jobLocation, ') started at', datetime.now(), 'on', os.environ['HOSTNAME']
    print 'distribute: inputs =', [ input.location for input in job.inputs ]
    job.run()
    print 'distribute: job', job.name, '(', jobLocation, ') finished at', datetime.now(), 'on', os.environ['HOSTNAME']

if __name__ == '__main__':
    try:
        main(sys.argv)
    except:
        traceback.print_exc()
        # exit code 100 so Sun Grid Engine treats specially
        # (FIXME : SGE-specific)
        sys.exit(100)
