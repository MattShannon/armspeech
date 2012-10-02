"""Sun Grid Engine-specific stuff for executing distributed computations.

Requires Sun Grid Engine `qsub` command to work on the local machine.
"""

# Copyright 2011, 2012 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

import queuer as qr
import sge_runner
from armspeech.util import persist
from codedep import codeDeps

import re
import os
import sys
import logging
import subprocess
from subprocess import PIPE
from armspeech.util import subprocesshelp
import inspect
import threading
import socket
from collections import defaultdict

@codeDeps(qr.Queuer)
class SgeQueuer(qr.Queuer):
    def submitOne(self, job, live, verbosity):
        liveJob = self.buildRepo.createLiveJob(job, queuer = self)

        parentJids = [ live[parentJob.secHash()].extra('jid') for parentJob in job.parentJobs() if parentJob.secHash() in live ]

        jidAssigned = self.qsub(job.name, liveJob.dir, parentJids, verbosity)
        liveJob.setExtra('jid', jidAssigned)

        if verbosity >= 1:
            print 'queuer: job', job.secHash(), '(', job.name, ')', '(', liveJob.dir, ') submitted with jid', jidAssigned

        liveJob.setSubmitted()

        return liveJob

@codeDeps()
class MockQueueState(object):
    def __init__(self):
        self.counter = 0
        self.mockStatus = dict()
        self.children = defaultdict(list)
        self.parents = dict()
        self.jobSpec = dict()
    def newJid(self):
        self.counter += 1
        return str(self.counter)
    def submit(self, parentJids, jobName, args, logDir):
        jid = self.newJid()
        self.parents[jid] = parentJids
        for parentJid in parentJids:
            self.children[parentJid].append(jid)
        self.jobSpec[jid] = jobName, args, logDir
        return jid

# (FIXME : use multiprocessing instead of threading, and then move this into
#   a separate file? Would require changes to the way code shares state, but
#   would allow true multithreaded single-machine job running. At the moment
#   we presumably only get a speed-up from using threads if jobs are I/O-bound.)
@codeDeps(MockQueueState, SgeQueuer, persist.secHashObject, sge_runner)
class MockSgeQueuer(SgeQueuer):
    def __init__(self, buildRepo, pythonExec = '/usr/bin/python'):
        self.buildRepo = buildRepo
        self.pythonExec = pythonExec

        self.queueState = MockQueueState()
        self.secHashUid = persist.secHashObject(id(self))

    def parentsDone(self, jid):
        return all([ self.queueState.mockStatus[parentJid] in [3, 10] for parentJid in self.queueState.parents[jid] ])

    def runIfEligible(self, jid, verbosity):
        if self.parentsDone(jid) and self.queueState.mockStatus[jid] == 1:
            def run():
                jobName, args, logDir = self.queueState.jobSpec[jid]
                mockEnv = dict()
                if 'PYTHONPATH' in os.environ:
                    mockEnv['PYTHONPATH'] = os.environ['PYTHONPATH']
                mockEnv['JOB_ID'] = jid
                if 'HOSTNAME' in os.environ:
                    mockEnv['HOSTNAME'] = os.environ['HOSTNAME']
                else:
                    mockEnv['HOSTNAME'] = socket.gethostname()
                mockEnv['SGE_TASK_ID'] = 'undefined'
                mockEnv['JOB_NAME'] = jobName

                self.queueState.mockStatus[jid] = 2
                if verbosity >= 1:
                    print 'mock_sge_queuer: job', jid, 'started'
                try:
                    p = subprocess.Popen([self.pythonExec] + args, stdout = PIPE, stderr = PIPE, env = mockEnv)
                    stdoutdata, stderrdata = p.communicate()
                    with open(os.path.join(logDir, jobName+'.o'+jid), 'w') as outFile:
                        outFile.write(stdoutdata)
                    with open(os.path.join(logDir, jobName+'.e'+jid), 'w') as errFile:
                        errFile.write(stderrdata)
                    if p.returncode == 0:
                        self.queueState.mockStatus[jid] = 3
                        if verbosity >= 1:
                            print 'mock_sge_queuer: job', jid, 'finished'
                    elif p.returncode == 100:
                        self.queueState.mockStatus[jid] = 11
                        logging.warning('mock_sge_queuer: error in job '+str(jid)+' (holding successors)')
                    else:
                        self.queueState.mockStatus[jid] = 10
                        logging.warning('mock_sge_queuer: error in job '+str(jid)+'')
                except:
                    self.queueState.mockStatus[jid] = 10
                    logging.warning('mock_sge_queuer: error in job '+str(jid)+'')
                    raise
                finally:
                    self.runNext(jid, verbosity)
            t = threading.Thread(target = run)
            t.start()

    def runNext(self, jidJustFinished, verbosity):
        jids = self.queueState.children[jidJustFinished]
        for jid in jids:
            self.runIfEligible(jid, verbosity)

    def qsub(self, jobName, liveJobDir, parentJids, verbosity):
        args = [
            inspect.getsourcefile(sge_runner),
            liveJobDir
        ]

        if verbosity >= 2:
            print 'queuer: command:'
            print '\t', ' '.join(args)

        jidAssigned = self.queueState.submit(parentJids, jobName, args, logDir = liveJobDir)
        self.queueState.mockStatus[jidAssigned] = 1

        self.runIfEligible(jidAssigned, verbosity)

        return jidAssigned

qsubRe = re.compile(r'Your job.* ([0-9]+) \("(.*)"\) has been submitted\n$')

@codeDeps(SgeQueuer, persist.secHashObject, qsubRe, sge_runner,
    subprocesshelp.check_output
)
class QsubSgeQueuer(SgeQueuer):
    def __init__(self, buildRepo, project, email = None, emailOpts = 'n', jointLog = False, pythonExec = '/usr/bin/python'):
        self.buildRepo = buildRepo
        self.project = project
        self.email = email
        self.emailOpts = emailOpts
        self.jointLog = jointLog
        self.pythonExec = pythonExec

        # attempt to find name of cell as a way of identifying this grid
        self.cell = os.environ.get('GE_CELL', os.environ.get('SGE_CELL', 'default'))
        self.secHashUid = persist.secHashObject((self.buildRepo, self.cell, self.project))

        # check there is a grid available for us to use
        try:
            subprocesshelp.check_output(['qstat'])
        except OSError:
            raise RuntimeError('grid not found (running qstat failed)')

    def qsub(self, jobName, liveJobDir, parentJids, verbosity):
        # N.B. SGE copies sge_runner.py to a spool directory. This means
        #   that if we somehow manage to call this module without PYTHONPATH
        #   being set (e.g. if we call it from a module defined in the parent
        #   directory of 'armspeech') then we still need to set PYTHONPATH when
        #   running the job (and sys.path[0] is just a decent heuristic guess).
        envPythonPath = os.environ['PYTHONPATH'] if 'PYTHONPATH' in os.environ else sys.path[0]
        args = [
            'qsub',
            '-N', jobName,
            '-P', self.project,
            '-m', self.emailOpts,
        ] + (
            [] if self.email is None else ['-M', self.email]
        ) + [
            '-S', self.pythonExec,
            '-v', 'PYTHONPATH='+envPythonPath,
            '-cwd'
        ] + (
            ['-hold_jid', ','.join(parentJids)] if parentJids else []
        ) + [
            '-j', 'y' if self.jointLog else 'n',
            '-o', liveJobDir,
            '-e', liveJobDir,
            inspect.getsourcefile(sge_runner),
            liveJobDir
        ]

        if verbosity >= 2:
            print 'queuer: command:'
            print '\t', ' '.join(args)

        stdoutdata = subprocesshelp.check_output(args)

        match = qsubRe.match(stdoutdata)
        if not match:
            raise RuntimeError('qsub output was not of expected format: '+stdoutdata)
        jidAssigned, nameAgain = match.groups()
        assert jobName == nameAgain

        return jidAssigned
