"""Distributed computation stuff specific to the Sun Grid Engine.

Requires Sun Grid Engine `qsub` command work on the local machine.
"""

# Copyright 2011 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division
from __future__ import with_statement

import persist
import distribute

import re
import os
import sys
import subprocess
from subprocess import PIPE
import subprocesshelp
import threading
import time

class SgeContext(distribute.ExecContext):
    def submitAll(self, job, submittedJid, verbosity):
        if job in submittedJid:
            return submittedJid[job]
        else:
            parentJids = []
            for parentJob in job.parentJobsToRun():
                parentJid = self.submitAll(parentJob, submittedJid, verbosity)
                parentJids.append(parentJid)
            jid = self.submitOne(job, parentJids, verbosity)
            submittedJid[job] = jid
            return jid

    def generateArtifacts(self, finalArtifacts, verbosity = 1):
        submittedJid = dict()
        for art in finalArtifacts:
            for parentJob in distribute.getParentJobs(art):
                self.submitAll(parentJob, submittedJid, verbosity)
        if verbosity >= 1:
            print 'sge: final artifacts will be at:'
            for art in finalArtifacts:
                print 'sge:     '+art.location

class MockSgeContext(SgeContext):
    def __init__(self, repo, logDir = '.', pythonExec = '/usr/bin/python'):
        self.repo = repo
        self.logDir = logDir
        self.pythonExec = pythonExec

        self.counter = 0
        self.done = dict()

    def submitOne(self, job, parentJids, verbosity):
        jobLocation = self.repo.newLocation()
        persist.savePickle(jobLocation, job)

        args = [
            'distribute.py',
            jobLocation
        ]

        if verbosity >= 2:
            print 'sge: command:'
            print '\t', ' '.join(args)

        jid = str(self.counter)
        self.counter += 1
        self.done[jid] = False

        mockEnv = dict()
        mockEnv['JOB_ID'] = jid
        mockEnv['HOSTNAME'] = 'anotherhost'
        mockEnv['SGE_TASK_ID'] = 'undefined'
        mockEnv['JOB_NAME'] = job.name

        def run():
            while not all([ self.done[parentJid] for parentJid in parentJids ]):
                time.sleep(0.5)
            p = subprocess.Popen([self.pythonExec] + args, stdout = PIPE, stderr = PIPE, env = mockEnv)
            stdoutdata, stderrdata = p.communicate()
            with open(os.path.join(self.logDir, job.name+'.o'+jid), 'w') as outFile:
                outFile.write(stdoutdata)
            with open(os.path.join(self.logDir, job.name+'.e'+jid), 'w') as errFile:
                errFile.write(stderrdata)
            self.done[jid] = True
        t = threading.Timer(2.0, run)
        t.start()

        if verbosity >= 1:
            print 'sge: job', job.name, '(', jobLocation, ') submitted with',
            print 'jid', jid

        return jid

qsubRe = re.compile(r'Your job.* ([0-9]+) \("(.*)"\) has been submitted\n$')

class QsubSgeContext(SgeContext):
    def __init__(self, repo, project, email = '', emailOpts = 'n', logDir = '.', jointLog = False, pythonExec = '/usr/bin/python'):
        self.repo = repo
        self.project = project
        self.email = email
        self.emailOpts = emailOpts
        self.logDir = logDir
        self.jointLog = jointLog
        self.pythonExec = pythonExec

    def submitOne(self, job, parentJids, verbosity):
        jobLocation = self.repo.newLocation()
        persist.savePickle(jobLocation, job)

        args = [
            'qsub',
            '-N', job.name,
            '-P', self.project,
            '-m', self.emailOpts,
            '-M', self.email,
            '-S', self.pythonExec,
            # N.B. below is necessary since SGE copies distribute.py to a spool
            #   directory, which ruins the import search path sys.path
            '-v', 'PYTHONPATH='+sys.path[0],
            '-cwd'
        ] + (
            ['-hold_jid', ','.join(parentJids)] if parentJids else []
        ) + [
            '-j', 'y' if self.jointLog else 'n',
            '-o', self.logDir,
            '-e', self.logDir,
            'distribute.py',
            jobLocation
        ]

        if verbosity >= 2:
            print 'sge: command:'
            print '\t', ' '.join(args)

        stdoutdata = subprocesshelp.check_output(args)

        if verbosity >= 1:
            print 'sge: job', job.name, '(', jobLocation, ') submitted with',
        match = qsubRe.match(stdoutdata)
        if not match:
            raise RuntimeError('qsub output was not of expected format: '+stdoutdata)
        jid, nameAgain = match.groups()
        assert job.name == nameAgain
        if verbosity >= 1:
            print 'jid', jid

        return jid
