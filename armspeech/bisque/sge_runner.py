"""Wrapper to run a job on Sun Grid Engine."""

# Copyright 2011, 2012 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

from armspeech.util.timing import timed

import os
import sys
import traceback
from datetime import datetime

def main(args):
    import armspeech.bisque.queuer as qr
    from armspeech.util import persist

    assert len(args) == 2
    liveJobDir = args[1]
    liveJob = qr.LiveJob(liveJobDir)
    liveJob.setRunning()

    try:
        job = persist.loadPickle(os.path.join(liveJob.dir, 'job.pickle'))
        jobSecHash = liveJob.extra('secHash')
        # (FIXME : this should be trivially true now. Remove secHash extra entirely?)
        assert jobSecHash == job.secHash()
        buildRepo = persist.loadPickle(os.path.join(liveJob.dir, 'buildRepo.pickle'))
        print 'sge_runner: job', job.secHash(), '(', job.name, ')', '(', liveJob.dir, ') started at', datetime.now(), 'on', os.environ['HOSTNAME']
        print 'sge_runner: build dir =', buildRepo.base
        print 'sge_runner: inputs =', [ inputArt.secHash() for inputArt in job.inputs ]
        job.checkAllSecHash()
        timed(job.run)(buildRepo)
        job.checkAllSecHash()
        print 'sge_runner: job', job.secHash(), '(', job.name, ')', '(', liveJob.dir, ') finished at', datetime.now(), 'on', os.environ['HOSTNAME']
    except:
        liveJob.setError()
        raise

    liveJob.setCompleted()

if __name__ == '__main__':
    try:
        main(sys.argv)
    except:
        traceback.print_exc()
        # using exit code 100 (treated specially by SGE) ensures child jobs don't run if this job fails
        sys.exit(100)
