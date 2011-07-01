"""Unit tests for distributed computation framework."""

# Copyright 2011 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

from dist import LinearGaussian
import persist
import distribute
import test_persist
import test_distribute_jobs as jobs

import unittest
import numpy as np

def assert_allclose(actual, desired, rtol = 1e-7, atol = 1e-14, msg = 'items not almost equal'):
    if np.shape(actual) != np.shape(desired) or not np.allclose(actual, desired, rtol, atol):
        raise AssertionError(msg+'\n ACTUAL:  '+repr(actual)+'\n DESIRED: '+repr(desired))

class TestDistribute(unittest.TestCase):
    def test_full_generation_simple_addition(self):
        repo = test_persist.TempSimpleRepo()

        initJob = jobs.ZeroJob(repo, name = 'initJob')
        job1 = jobs.AddOneJob(repo, initJob.valueOut, name = 'job1')
        job2 = jobs.AddOneJob(repo, job1.valueOut, name = 'job2')
        job3 = jobs.AddOneJob(repo, job2.valueOut, name = 'job3')
        valueOut = job3.valueOut

        sgeContext = distribute.LocalExecContext(repo = repo)

        sgeContext.generateArtifacts([valueOut], verbosity = 0)

        assert persist.loadPickle(valueOut.location) == 3
        assert persist.loadPickle(initJob.valueOut.location) == 0
        assert persist.loadPickle(job1.valueOut.location) == 1
        assert persist.loadPickle(job2.valueOut.location) == 2
        assert persist.loadPickle(job3.valueOut.location) == 3

        repo.remove()

    def test_full_generation_LinearGaussian_with_split(self):
        repo = test_persist.TempSimpleRepo()

        simpleCorpus = [
            (np.array([1.0]), 1.0),
            (np.array([2.0]), 2.0),
            (np.array([3.0]), 3.1),
            (np.array([2.5]), 2.4)
        ]
        corpusList = [simpleCorpus[:3], simpleCorpus[3:]]

        initAccJob = jobs.InitAccJob(repo, simpleCorpus, name = 'initAccJob')
        estimateJob1 = jobs.EstimateJob(repo, accIn = initAccJob.accOut, name = 'estimateJob1')
        accJobs2 = jobs.createSplitAccJobs(repo, corpusList, distIn = estimateJob1.distOut, name = 'accJob2')
        estimateJob2 = jobs.SplitEstimateJob(repo, accsIn = [ accJob.accOut for accJob in accJobs2 ], name = 'estimateJob2')
        distOut = estimateJob2.distOut

        sgeContext = distribute.LocalExecContext(repo = repo)

        sgeContext.generateArtifacts([distOut], verbosity = 0)

        distExpected = LinearGaussian(np.array([ 1.00246914]), 0.0049691358024697152)
        dist = persist.loadPickle(distOut.location)
        assert_allclose(dist.coeff, distExpected.coeff)
        assert_allclose(dist.variance, distExpected.variance)
        dist = persist.loadPickle(estimateJob1.distOut.location)
        assert_allclose(dist.coeff, distExpected.coeff)
        assert_allclose(dist.variance, distExpected.variance)

        repo.remove()

if __name__ == '__main__':
    unittest.main()
