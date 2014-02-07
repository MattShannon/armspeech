"""Unit tests for weighted network stuff."""

# Copyright 2011, 2012, 2013, 2014 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

import wnet
import semiring
from armspeech.util.memoize import memoize
from codedep import codeDeps

import unittest
import random
from numpy.random import randn, randint
import armspeech.numpy_settings

@codeDeps()
def randBool():
    return randint(0, 2) == 0

@codeDeps(wnet.ConcreteNet, wnet.checkConsistent, wnet.concretizeNet,
    wnet.nodeSetCompute
)
def gen_simple_ConcreteNet(genLabel, deltaTime, sortable = True, pathMustExist = False, maxNodes = 12, maxEdgesPerNode = 3):
    numNodes = randint(2, maxNodes + 1)

    edgesForwards = dict()
    for node in range(0, numNodes - 1):
        edgesForwards[node] = []
        for edge in range(randint(0, maxEdgesPerNode + 1)):
            label = genLabel()
            minNode = (node + 1) if sortable and deltaTime(label) == 0 else 1
            nextNode = randint(minNode, numNodes)
            edgesForwards[node].append((label, nextNode))
    edgesForwards[numNodes - 1] = []

    elems = [ None for node in range(numNodes) ]
    net = wnet.ConcreteNet(startNode = 0, endNode = numNodes - 1, elems = elems, edgesForwards = edgesForwards)

    if pathMustExist and not wnet.nodeSetCompute(net, accessibleOnly = True):
        return gen_simple_ConcreteNet(genLabel = genLabel, deltaTime = deltaTime, sortable = sortable, pathMustExist = pathMustExist, maxNodes = maxNodes, maxEdgesPerNode = maxEdgesPerNode)
    else:
        perm = list(range(numNodes))
        random.shuffle(perm)
        netPerm = wnet.concretizeNet(net, perm)

        wnet.checkConsistent(netPerm, nodeSet = set(range(numNodes)))
        return netPerm

@codeDeps(wnet.nodeSetCompute)
def is_valid_topSort(net, sortedNodes, deltaTime):
    nodeLookup = dict()
    for pos, node in enumerate(sortedNodes):
        assert node not in nodeLookup
        nodeLookup[node] = pos
    for node in wnet.nodeSetCompute(net, accessibleOnly = True):
        assert node in nodeLookup
    for node in nodeLookup:
        for forwards in [True, False]:
            for label, nextNode in net.next(node, forwards):
                if nextNode in nodeLookup and deltaTime(label) == 0:
                    leftNode = node if forwards else nextNode
                    rightNode = nextNode if forwards else node
                    if nodeLookup[leftNode] >= nodeLookup[rightNode]:
                        return False
    return True

@codeDeps()
def defaultGenLabel():
    return None if randint(0, 3) != 0 else randint(0, 3)
@codeDeps()
def defaultDeltaTime(label):
    return 0 if label is None else 1

@codeDeps(defaultDeltaTime, defaultGenLabel, gen_simple_ConcreteNet,
    is_valid_topSort, memoize, randBool, semiring.LogRealsField,
    wnet.ConcreteNet, wnet.HasCycleError, wnet.SimpleSumAgenda, wnet.TrivialNet,
    wnet.concretizeNetSimple, wnet.concretizeNetTopSort, wnet.isConsistent,
    wnet.netIsTopSorted, wnet.nodeSetCompute, wnet.sum, wnet.topSort
)
class TestWnet(unittest.TestCase):
    def test_TrivialNet_one_parameter_construction(self):
        net = wnet.TrivialNet('a')
    def test_isConsistent(self):
        def checkFor(forwardsLoops, backwardsLoops, shouldBeConsistent):
            edgesForwards = {
                0: [(None, 1)],
                1: [(None, 2)] + forwardsLoops,
                2: []
            }
            edgesBackwards = {
                0: [],
                1: [(None, 0)] + backwardsLoops,
                2: [(None, 1)]
            }
            elems = [ None for node in range(3) ]
            net = wnet.ConcreteNet(0, 2, elems, edgesForwards, edgesBackwards)
            assert wnet.isConsistent(net, set(range(net.numNodes))) == shouldBeConsistent
        checkFor([(None, 1)], [(None, 1)], True)
        checkFor([(None, 1), (None, 1)], [(None, 1)], False)
        checkFor([(None, 1), (None, 1)], [(None, 1), (None, 1)], True)
        # next two lines are an important check of multiple edge processing in isConsistent
        checkFor([('a', 1), ('b', 1)], [('a', 1), ('b', 1)], True)
        checkFor([('a', 1), ('b', 1)], [('b', 1), ('a', 1)], True)
        checkFor([(None, 1)], [('a', 1)], False)
        checkFor([('a', 1), ('b', 1)], [('a', 1), ('a', 1)], False)
    def test_topSort(self, its = 1000):
        for it in range(its):
            net = gen_simple_ConcreteNet(defaultGenLabel, defaultDeltaTime, sortable = True, pathMustExist = True)
            if randBool():
                net = wnet.concretizeNetSimple(net, accessibleOnly = True)
            sortedNodes = wnet.topSort(net, defaultDeltaTime, forwards = randBool(), detectCycles = randBool(), debugCheckInvariants = randBool())
            assert is_valid_topSort(net, sortedNodes, defaultDeltaTime)
    def test_topSort_net_not_necessarily_sortable(self, its = 500):
        for it in range(its):
            net = gen_simple_ConcreteNet(defaultGenLabel, defaultDeltaTime, sortable = False, pathMustExist = True)
            if randBool():
                net = wnet.concretizeNetSimple(net, accessibleOnly = True)
            sortedNodes = wnet.topSort(net, defaultDeltaTime, forwards = randBool(), detectCycles = False, debugCheckInvariants = randBool())
            # perform basic sanity checks (don't check return value, since we
            #   already know the net is probably not sortable)
            is_valid_topSort(net, sortedNodes, defaultDeltaTime)
    def test_topSort_non_emitting_self_cycle_detection(self):
        edges = {
            0: [(None, 1)],
            1: [(None, 2), (None, 1)],
            2: []
        }
        elems = [ None for node in range(3) ]
        net = wnet.ConcreteNet(0, 2, elems, edges)
        for forwards in [False, True]:
            for debugCheckInvariants in [False, True]:
                self.assertRaises(wnet.HasCycleError, wnet.topSort, net, defaultDeltaTime, forwards = forwards, detectCycles = True, debugCheckInvariants = debugCheckInvariants)
    def test_topSort_emitting_self_cycle_ok(self):
        edges = {
            0: [(None, 1)],
            1: [(None, 2), ('a', 1)],
            2: []
        }
        elems = [ None for node in range(3) ]
        net = wnet.ConcreteNet(0, 2, elems, edges)
        for forwards in [False, True]:
            for debugCheckInvariants in [False, True]:
                wnet.topSort(net, defaultDeltaTime, forwards = forwards, detectCycles = True, debugCheckInvariants = debugCheckInvariants)
    def test_topSort_cycle_detection(self):
        edges = {
            0: [(None, 1)],
            1: [(None, 2)],
            2: [(None, 3)],
            3: [(None, 4), (None, 1)],
            4: []
        }
        elems = [ None for node in range(5) ]
        net = wnet.ConcreteNet(0, 4, elems, edges)
        for forwards in [False, True]:
            for debugCheckInvariants in [False, True]:
                self.assertRaises(wnet.HasCycleError, wnet.topSort, net, defaultDeltaTime, forwards = forwards, detectCycles = True, debugCheckInvariants = debugCheckInvariants)
    def test_concretizeNetTopSort(self, its = 1000):
        for it in range(its):
            net = gen_simple_ConcreteNet(defaultGenLabel, defaultDeltaTime, sortable = True, pathMustExist = True)
            if randBool():
                net = wnet.concretizeNetSimple(net, accessibleOnly = True)
            sortedNodes = wnet.topSort(net, defaultDeltaTime)
            sortedNet = wnet.concretizeNetTopSort(net, defaultDeltaTime)
            sortedNetNodeSet = wnet.nodeSetCompute(sortedNet, accessibleOnly = False)
            assert wnet.netIsTopSorted(sortedNet, sortedNetNodeSet, defaultDeltaTime)
            assert len(sortedNetNodeSet) == len(sortedNodes)
    def test_sum_forwards_equals_backwards(self, its = 200):
        for it in range(its):
            def deltaTime(label):
                return 0
            net = gen_simple_ConcreteNet(defaultGenLabel, deltaTime, sortable = True, pathMustExist = True)

            ring = semiring.LogRealsField()
            labelToWeight = memoize(lambda label: ring.one if label is None else randn())
            def getAgenda(forwards):
                return wnet.SimpleSumAgenda(ring)
            totalWeightForwards = wnet.sum(net, labelToWeight, ring, getAgenda = getAgenda, forwards = True)
            totalWeightBackwards = wnet.sum(net, labelToWeight, ring, getAgenda = getAgenda, forwards = True)
            assert ring.isClose(totalWeightForwards, totalWeightBackwards)
    # FIXME : add tests for other stuff in wnet

@codeDeps(TestWnet)
def suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestWnet)

if __name__ == '__main__':
    unittest.main()
