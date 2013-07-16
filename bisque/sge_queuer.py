"""Sun Grid Engine-specific stuff for executing distributed computations.

Requires Sun Grid Engine `qsub` command to work on the local machine.
"""

# Copyright 2011, 2012, 2013 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

import queuer as qr
import sge_runner
import mock_sge
from bisque import persist
from codedep import codeDeps

import re
import os
import sys
from bisque import subprocesshelp
import inspect

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

@codeDeps(SgeQueuer, mock_sge.getMockSge, persist.secHashObject, sge_runner)
class MockSgeQueuer(SgeQueuer):
    def __init__(self, buildRepo, jointLog = False, pythonExec = '/usr/bin/python'):
        self.buildRepo = buildRepo
        self.jointLog = jointLog
        self.pythonExec = pythonExec

    def __enter__(self):
        self.submitServer, self.runServerProcess = mock_sge.getMockSge()
        self.secHashUid = persist.secHashObject((
            self.buildRepo, self.runServerProcess.pid
        ))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.submitServer.requestExitWhenDone()
        self.runServerProcess.join()

    def qsub(self, jobName, liveJobDir, parentJids, verbosity):
        args = [
            inspect.getsourcefile(sge_runner),
            liveJobDir
        ]

        if verbosity >= 2:
            print 'queuer: command:'
            print '\t', ' '.join(args)

        env = dict()
        if 'PYTHONPATH' in os.environ:
            env['PYTHONPATH'] = os.environ['PYTHONPATH']
        env['PYTHONUNBUFFERED'] = 'yes'

        jidAssigned = self.submitServer.submit(
            parentJids,
            jobName,
            env,
            self.pythonExec,
            args,
            logDir = liveJobDir,
            jointLog = self.jointLog
        )
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
        # FIXME : does sys.path[0] ever do anything here? Should just fail instead?
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
            '-v', 'PYTHONUNBUFFERED=yes',
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
