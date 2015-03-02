"""Unit tests for distributed computation execution."""

# Copyright 2011, 2012, 2013, 2014, 2015 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

# N.B. need to use absolute imports below
#   (This is because otherwise when this module is run as a script, the pickled
#   jobs don't contain the fully-qualified names for any modules imported below
#   using implicit relative imports, which leads to problems when these jobs
#   are unpickled to be run.)
from bisque import distribute
import bisque.queuer as qr
from bisque import sge_queuer
from bisque.distribute import lift
import bisque.test_queuer_jobs as jobs
from bisque.filehelp import TempDir
from codedep import codeDeps

import unittest
import logging
import time

@codeDeps(distribute.ThunkArtifact, jobs.AddJob, jobs.OneJob, jobs.getOne)
def simpleTestDag():
    oneJob1 = jobs.OneJob(name = 'oneJob1')
    oneArt = distribute.ThunkArtifact(jobs.getOne)
    addJobA = jobs.AddJob(oneJob1.valueOut, oneArt, name = 'addJobA')
    addJobB = jobs.AddJob(oneArt, addJobA.valueOut, name = 'addJobB')
    addJobC = jobs.AddJob(addJobA.valueOut, addJobB.valueOut, name = 'addJobC')
    addJobD = jobs.AddJob(oneJob1.valueOut, addJobB.valueOut, name = 'addJobD')
    return [(addJobC.valueOut, 5), (addJobD.valueOut, 4)], 5, 2

@codeDeps(distribute.ThunkArtifact, jobs.add, jobs.getOne, lift)
def simpleTestDagFunctionalSugar():
    outArt1 = lift(jobs.getOne, name = 'oneJob1')()
    outArt2 = distribute.ThunkArtifact(jobs.getOne)
    outArtA = lift(jobs.add, name = 'addJobA')(outArt1, outArt2)
    outArtB = lift(jobs.add, name = 'addJobB')(outArt2, outArtA)
    outArtC = lift(jobs.add, name = 'addJobC')(outArtA, outArtB)
    outArtD = lift(jobs.add, name = 'addJobD')(outArt1, outArtB)
    return [(outArtC, 5), (outArtD, 4)], 5, 2

@codeDeps(jobs.add, jobs.getOne, lift)
def liftExampleDag():
    oneArt = lift(jobs.getOne)()
    twoArt = lift(jobs.add)(oneArt, y = oneArt)
    return [(twoArt, 2), (oneArt, 1)], 2, 2

@codeDeps(TempDir, liftExampleDag, qr.BuildRepo, qr.LocalQueuer,
    sge_queuer.MockSgeQueuer, simpleTestDag, simpleTestDagFunctionalSugar
)
class TestDistribute(unittest.TestCase):
    def test_LocalQueuer(self):
        with TempDir() as tempDir:
            buildRepo = qr.BuildRepo(base = tempDir.location)
            queuer = qr.LocalQueuer(buildRepo = buildRepo)
            for testDag, totJobs, finalJobs in [simpleTestDag(),
                                                simpleTestDagFunctionalSugar(),
                                                liftExampleDag()]:
                live = queuer.generateArtifacts(
                    [ art for art, expectedValue in testDag ],
                    verbosity = 0
                )
                for art, expectedValue in testDag:
                    assert art.loadValue(buildRepo) == expectedValue

    def test_MockSgeQueuer_one_big_submission(self):
        for testDag, totJobs, finalJobs in [simpleTestDag(),
                                            simpleTestDagFunctionalSugar(),
                                            liftExampleDag()]:
            with TempDir() as tempDir:
                buildRepo = qr.BuildRepo(base = tempDir.location)
                with sge_queuer.MockSgeQueuer(buildRepo = buildRepo) as queuer:
                    finalArtifacts = [ art for art, expectedValue in testDag ]
                    live = queuer.generateArtifacts(finalArtifacts,
                                                    verbosity = 0)
                    assert len(live) == totJobs
                    finalLiveJobs = [
                        live[job.secHash()]
                        for art in finalArtifacts for job in art.parents()
                        if job.secHash() in live
                    ]
                    assert len(finalLiveJobs) == finalJobs
                    while not all([ liveJob.hasEnded()
                                    for liveJob in finalLiveJobs ]):
                        time.sleep(0.1)
                    assert all([ liveJob.hasCompleted()
                                 for liveJob in live.values() ])
                    for art, expectedValue in testDag:
                        assert art.loadValue(buildRepo) == expectedValue

                    # check no jobs submitted if we already have the desired
                    #   artifacts
                    live = queuer.generateArtifacts(finalArtifacts,
                                                    verbosity = 0)
                    assert len(live) == 0

    def test_MockSgeQueuer_several_little_submissions(self):
        for testDag, totJobs, finalJobs in [simpleTestDag(),
                                            simpleTestDagFunctionalSugar(),
                                            liftExampleDag()]:
            with TempDir() as tempDir:
                buildRepo = qr.BuildRepo(base = tempDir.location)
                with sge_queuer.MockSgeQueuer(buildRepo = buildRepo) as queuer:
                    finalLiveJobs = []
                    liveJobDirs = set()
                    totSubmitted = 0
                    for art, expectedValue in testDag:
                        live = queuer.generateArtifacts([art], verbosity = 0)
                        finalLiveJobs.extend([
                            live[job.secHash()]
                            for job in art.parents() if job.secHash() in live
                        ])
                        liveJobDirs.update([ liveJob.dir
                                             for liveJob in live.values() ])
                        totSubmitted += len(live)
                    assert len(liveJobDirs) == totJobs
                    assert len(finalLiveJobs) == finalJobs
                    if totSubmitted == totJobs:
                        logging.warning(
                            're-use of submitted jobs for MockSgeQueuer not'
                            ' properly tested, since jobs completed too fast'
                        )
                    while not all([ liveJob.hasEnded()
                                    for liveJob in finalLiveJobs ]):
                        time.sleep(0.1)
                    assert all([ liveJob.hasCompleted()
                                 for liveJob in live.values() ])
                    for art, expectedValue in testDag:
                        assert art.loadValue(buildRepo) == expectedValue

                    # check no jobs submitted if we already have the desired
                    #   artifacts
                    for art, expectedValue in testDag:
                        live = queuer.generateArtifacts([art], verbosity = 0)
                        assert len(live) == 0

@codeDeps(TestDistribute)
def suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestDistribute)

if __name__ == '__main__':
    unittest.main()
