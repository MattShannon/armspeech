"""Unit tests for distributed computation execution."""

# Copyright 2011, 2012 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

# N.B. need to use absolute imports below
#   (This is because otherwise when this module is run as a script, the pickled
#   jobs don't contain the fully-qualified names for any modules imported below
#   using implicit relative imports, which leads to problems when these jobs
#   are unpickled to be run.)
import armspeech.bisque.queuer as qr
from armspeech.bisque import sge_queuer
import armspeech.bisque.test_queuer_jobs as jobs
from armspeech.util.filehelp import TempDir

import unittest
import logging
import time

def simpleTestDag():
    oneJob1 = jobs.OneJob(name = 'oneJob1')
    oneJob2 = jobs.OneJob(name = 'oneJob2')
    addJobA = jobs.AddJob(oneJob1.valueOut, oneJob2.valueOut, name = 'addJobA')
    addJobB = jobs.AddJob(oneJob2.valueOut, addJobA.valueOut, name = 'addJobB')
    addJobC = jobs.AddJob(addJobA.valueOut, addJobB.valueOut, name = 'addJobC')
    addJobD = jobs.AddJob(oneJob1.valueOut, addJobB.valueOut, name = 'addJobD')
    return [(addJobC.valueOut, 5), (addJobD.valueOut, 4)], 6, 2

class TestDistribute(unittest.TestCase):
    def test_LocalQueuer(self):
        tempDir = TempDir()
        buildRepo = qr.BuildRepo(base = tempDir.location)
        queuer = qr.LocalQueuer(buildRepo = buildRepo)
        for testDag, totJobs, finalJobs in [simpleTestDag()]:
            live = queuer.generateArtifacts([ art for art, expectedValue in testDag ], verbosity = 0)
            for art, expectedValue in testDag:
                assert buildRepo.loadFromArt(art) == expectedValue
        tempDir.remove()

    def test_MockSgeQueuer_one_big_submission(self):
        tempDir = TempDir()
        buildRepo = qr.BuildRepo(base = tempDir.location)
        queuer = sge_queuer.MockSgeQueuer(buildRepo = buildRepo)
        for testDag, totJobs, finalJobs in [simpleTestDag()]:
            finalArtifacts = [ art for art, expectedValue in testDag ]
            live = queuer.generateArtifacts(finalArtifacts, verbosity = 0)
            assert len(live) == totJobs
            finalLiveJobs = [ live[job.secHash()] for art in finalArtifacts for job in art.parentJobs() if job.secHash() in live ]
            assert len(finalLiveJobs) == finalJobs
            while not all([ liveJob.hasEnded() for liveJob in finalLiveJobs ]):
                time.sleep(0.1)
            for art, expectedValue in testDag:
                assert buildRepo.loadFromArt(art) == expectedValue

            # check no jobs submitted if we already have the desired artifacts
            live = queuer.generateArtifacts(finalArtifacts, verbosity = 0)
            assert len(live) == 0
        tempDir.remove()

    def test_MockSgeQueuer_several_little_submissions(self):
        tempDir = TempDir()
        buildRepo = qr.BuildRepo(base = tempDir.location)
        queuer = sge_queuer.MockSgeQueuer(buildRepo = buildRepo)
        for testDag, totJobs, finalJobs in [simpleTestDag()]:
            finalLiveJobs = []
            liveJobDirs = set()
            totSubmitted = 0
            for art, expectedValue in testDag:
                live = queuer.generateArtifacts([art], verbosity = 0)
                finalLiveJobs.extend([ live[job.secHash()] for job in art.parentJobs() if job.secHash() in live ])
                liveJobDirs.update([ liveJob.dir for liveJob in live.values() ])
                totSubmitted += len(live)
            assert len(liveJobDirs) == totJobs
            assert len(finalLiveJobs) == finalJobs
            if totSubmitted == totJobs:
                logging.warning('re-use of submitted jobs for MockSgeQueuer not properly tested, since jobs completed too fast')
            while not all([ liveJob.hasEnded() for liveJob in finalLiveJobs ]):
                time.sleep(0.1)
            for art, expectedValue in testDag:
                assert buildRepo.loadFromArt(art) == expectedValue

            # check no jobs submitted if we already have the desired artifacts
            for art, expectedValue in testDag:
                live = queuer.generateArtifacts([art], verbosity = 0)
                assert len(live) == 0
        tempDir.remove()

def suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestDistribute)

if __name__ == '__main__':
    unittest.main()
