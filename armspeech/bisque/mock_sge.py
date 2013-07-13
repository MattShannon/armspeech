"""A mock Sun Grid Engine-style server which runs locally.

Uses multiple cores if available.
"""

# Copyright 2011, 2012, 2013 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

from codedep import codeDeps

import os
import logging
import subprocess
import multiprocessing
import socket
from collections import defaultdict

@codeDeps()
class RunServer(object):
    def __init__(self, submitConn, pollInterval = 0.1, verbosity = 1):
        self.submitConn = submitConn
        self.pollInterval = pollInterval
        self.verbosity = verbosity

        self.children = defaultdict(list)
        self.parents = dict()
        self.jobSpec = dict()
        self.status = dict()
        self.exitWhenDone = False

    def parentsDone(self, jid):
        return all([ self.status[parentJid] in [3, 10]
                     for parentJid in self.parents[jid] ])

    def startJob(self, jid):
        jobName, env, execFile, args, logDir, jointLog = self.jobSpec[jid]

        env = dict(env)
        env['JOB_ID'] = jid
        if 'HOSTNAME' in os.environ:
            env['HOSTNAME'] = os.environ['HOSTNAME']
        else:
            env['HOSTNAME'] = socket.gethostname()
        env['SGE_TASK_ID'] = 'undefined'
        env['JOB_NAME'] = jobName

        assert self.status[jid] == 1
        self.status[jid] = 2
        if self.verbosity >= 1:
            print 'mock_sge: job %s starting' % jid
        try:
            if not os.path.isdir(logDir):
                raise RuntimeError('log directory %s does not exist' % logDir)
            outFilename = os.path.join(logDir, jobName+'.o'+jid)
            outFile = open(outFilename, 'w')
            if jointLog:
                errFile = subprocess.STDOUT
            else:
                errFilename = os.path.join(logDir, jobName+'.e'+jid)
                errFile = open(errFilename, 'w')
            process = subprocess.Popen([execFile] + args, stdout = outFile,
                                       stderr = errFile, env = env)
            return process
        except Exception, e:
            logging.warning(
                'mock_sge: error while starting job %s: %s' %
                (jid, e)
            )
            assert self.status[jid] == 2
            self.status[jid] = 10
            return None

    def processFinishedJob(self, jid, process):
        if process.returncode == 0:
            assert self.status[jid] == 2
            self.status[jid] = 3
            if self.verbosity >= 1:
                print 'mock_sge: job %s finished' % jid
        elif process.returncode == 100:
            logging.warning(
                'mock_sge: error in job %s (holding successors)' % jid
            )
            assert self.status[jid] == 2
            self.status[jid] = 11
        else:
            logging.warning('mock_sge: error in job %s' % jid)
            assert self.status[jid] == 2
            self.status[jid] = 10

    def loop(self):
        heldJobs = set()
        activeJobs = dict()

        def startIfEligible(jid):
            if self.status[jid] == 1 and self.parentsDone(jid):
                heldJobs.remove(jid)
                process = self.startJob(jid)
                activeJobs[jid] = process

        while True:
            if self.submitConn.poll(self.pollInterval):
                msg = self.submitConn.recv()
                if len(msg) == 1:
                    self.exitWhenDone, = msg
                else:
                    jid, parentJids, jobSpec = msg

                    self.parents[jid] = parentJids
                    for parentJid in parentJids:
                        self.children[parentJid].append(jid)
                    self.jobSpec[jid] = jobSpec
                    assert jid not in self.status
                    self.status[jid] = 1

                    heldJobs.add(jid)
                    startIfEligible(jid)
            elif self.exitWhenDone and not activeJobs:
                return

            for jid, process in activeJobs.items():
                # process may be None, indicating a job that failed to start
                if process is None or process.poll() is not None:
                    del activeJobs[jid]
                    if process is not None:
                        self.processFinishedJob(jid, process)
                    for childJid in self.children[jid]:
                        startIfEligible(childJid)

@codeDeps()
class SubmitServer(object):
    def __init__(self, submitConn):
        self.submitConn = submitConn

        self.counter = 0

    def _newJid(self):
        self.counter += 1
        return str(self.counter)

    def submit(self, parentJids, jobName, env, execFile, args, logDir,
               jointLog):
        jid = self._newJid()
        jobSpec = jobName, env, execFile, args, logDir, jointLog
        self.submitConn.send((jid, parentJids, jobSpec))
        return jid

    def requestExitWhenDone(self, exitWhenDone = True):
        self.submitConn.send((exitWhenDone,))

@codeDeps(RunServer, SubmitServer)
def getMockSge(pollInterval = 0.1, verbosity = 0):
    runServerSubmitConn, submitServerSubmitConn = multiprocessing.Pipe(False)
    runServer = RunServer(runServerSubmitConn, pollInterval = pollInterval,
                          verbosity = verbosity)
    submitServer = SubmitServer(submitServerSubmitConn)
    runServerProcess = multiprocessing.Process(target = runServer.loop)
    runServerProcess.start()
    return submitServer, runServerProcess
