"""Weighted network definitions and algorithms."""

# Copyright 2011, 2012, 2013, 2014, 2015 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.

import logging
import math
import heapq
from collections import deque, defaultdict

from codedep import codeDeps, ForwardRef

@codeDeps()
class Net(object):
    def start(self, forwards):
        abstract
    def end(self, forwards):
        return self.start(not forwards)
    def elem(self, node):
        abstract
    def next(self, node, forwards):
        abstract

@codeDeps()
def basicChecks(net):
    """Basic checks that all nets should satisfy."""
    assert net.start(True) != net.start(False)
    for forwards in [True, False]:
        assert net.start(forwards) == net.end(not forwards)
    for forwards in [True, False]:
        assert len(net.next(net.end(forwards), forwards)) == 0
    assert net.elem(net.start(True)) is None
    assert net.elem(net.end(True)) is None

@codeDeps(Net, basicChecks)
class MappedElemNet(Net):
    def __init__(self, fn, net):
        self.fn = fn
        self.net = net

        basicChecks(self)
    def start(self, forwards):
        return self.net.start(forwards)
    def elem(self, node):
        orig = self.net.elem(node)
        return None if orig is None else self.fn(orig)
    def next(self, node, forwards):
        return self.net.next(node, forwards)

@codeDeps(Net, basicChecks)
class MappedLabelNet(Net):
    """Applies a function to each non-None label in a net."""
    def __init__(self, fn, net):
        self.fn = fn
        self.net = net

        basicChecks(self)
    def start(self, forwards):
        return self.net.start(forwards)
    def elem(self, node):
        return self.net.elem(node)
    def next(self, node, forwards):
        return [ (None if label is None else self.fn(label), nextNode) for label, nextNode in self.net.next(node, forwards) ]

@codeDeps(Net, basicChecks)
class FlatMappedNet(Net):
    """Lazy flat-mapped net.

    Each emitting node in the given net is replaced by the sub-net
    f(<elem of the emitting node>).

    N.B. f is called frequently, so should probably be memoized if computing it
    is an expensive operation.
    """
    def __init__(self, fn, netTop):
        self.fn = fn
        self.netTop = netTop

        basicChecks(self)
    def start(self, forwards):
        return (self.netTop.start(forwards), None)
    def elem(self, node):
        nodeTop, nodeBtm = node
        elemTop = self.netTop.elem(nodeTop)
        return None if elemTop is None else self.fn(elemTop).elem(nodeBtm)
    def next(self, node, forwards):
        nodeTop, nodeBtm = node
        elemTop = self.netTop.elem(nodeTop)
        netBtm = self.fn(elemTop) if elemTop is not None else None
        if elemTop is None or nodeBtm == netBtm.end(forwards):
            ret = []
            for label, nextNodeTop in self.netTop.next(nodeTop, forwards):
                nextElemTop = self.netTop.elem(nextNodeTop)
                if nextElemTop is None:
                    ret.append((label, (nextNodeTop, None)))
                else:
                    netBtm = self.fn(nextElemTop)
                    ret.append((label, (nextNodeTop, netBtm.start(forwards))))
            return ret
        else:
            return [ (label, (nodeTop, nextNodeBtm)) for label, nextNodeBtm in netBtm.next(nodeBtm, forwards) ]

@codeDeps(Net, basicChecks)
class TrivialNet(Net):
    def __init__(self, label):
        self.label = label

        basicChecks(self)
    def start(self, forwards):
        return 0 if forwards else 1
    def elem(self, node):
        return None
    def next(self, node, forwards):
        if node == 0 and forwards or node == 1 and not forwards:
            return [(self.label, 1 - node)]
        else:
            return []

@codeDeps(Net, basicChecks)
class SequenceNet(Net):
    def __init__(self, elems, label):
        self.count = len(elems)
        self.elems = elems
        self.label = label

        basicChecks(self)
    def start(self, forwards):
        return -1 if forwards else self.count
    def elem(self, node):
        if node == -1 or node == self.count:
            return None
        else:
            return self.elems[node]
    def next(self, node, forwards):
        if node < -1 or node > self.count:
            raise RuntimeError('bad node id ('+str(node)+') for net '+repr(self))
        nextNode = node + (1 if forwards else -1)
        rightNode = node + (1 if forwards else 0)
        if node == self.end(forwards):
            return []
        else:
            return [(self.label, nextNode)]

@codeDeps(Net, basicChecks)
class ProbLeftToRightNet(Net):
    def __init__(self, numStates, elemFor, transLabelFor):
        self.count = numStates
        self.elemFor = elemFor
        self.transLabelFor = transLabelFor

        basicChecks(self)
    def start(self, forwards):
        return -1 if forwards else self.count
    def elem(self, node):
        if node == -1 or node == self.count:
            return None
        else:
            return self.elemFor(node)
    def next(self, node, forwards):
        if node < -1 or node > self.count:
            raise RuntimeError('bad node id ('+str(node)+') for net '+repr(self))
        nextNode = node + (1 if forwards else -1)
        leftNode = node + (0 if forwards else -1)
        if node == self.end(forwards):
            return []
        elif node == self.start(forwards):
            neighTransLabel = self.transLabelFor(leftNode, 1) if leftNode != -1 else None
            return [
                (neighTransLabel, nextNode)
            ]
        else:
            selfTransLabel = self.transLabelFor(node, 0)
            neighTransLabel = self.transLabelFor(leftNode, 1) if leftNode != -1 else None
            return [
                (selfTransLabel, node),
                (neighTransLabel, nextNode)
            ]

@codeDeps(ProbLeftToRightNet)
def probLeftToRightNet(elems, transLabels):
    return ProbLeftToRightNet(len(elems), elemFor = lambda state: elems[state], transLabelFor = lambda state, adv: transLabels[state][adv])

@codeDeps(ProbLeftToRightNet)
def sumGeometricDurNet(elem, transLabels):
    numDurStates = len(transLabels)
    return ProbLeftToRightNet(numDurStates, elemFor = lambda durState: elem, transLabelFor = lambda durState, adv: transLabels[durState][adv])

@codeDeps(Net, basicChecks)
class ProbLeftToRightZeroNet(Net):
    def __init__(self, numStates, elemFor, transLabelFor):
        self.count = numStates
        self.elemFor = elemFor
        self.transLabelFor = transLabelFor

        basicChecks(self)
    def start(self, forwards):
        return -1 if forwards else self.count * 2 + 1
    def elem(self, node):
        if node < -1 or node == self.count * 2 or node > self.count * 2 + 1:
            raise RuntimeError('bad node id ('+str(node)+') for net '+repr(self))
        if node % 2 == 1:
            return None
        else:
            return self.elemFor(node // 2)
    def next(self, node, forwards):
        if node < -1 or node == self.count * 2 or node > self.count * 2 + 1:
            raise RuntimeError('bad node id ('+str(node)+') for net '+repr(self))
        if node % 2 == 0:
            selfTransLabel = None if forwards else self.transLabelFor(node // 2, 0)
            return [(selfTransLabel, node + 1)]
        else:
            nextNode = node + (2 if forwards else -2)
            leftNode = node + (0 if forwards else -2)
            if node == self.end(forwards):
                return []
            elif node == self.start(forwards):
                neighTransLabel = self.transLabelFor(leftNode // 2, 1) if leftNode != -1 else None
                return [
                    (neighTransLabel, nextNode)
                ]
            else:
                selfTransLabel = self.transLabelFor(node // 2, 0) if forwards else None
                neighTransLabel = self.transLabelFor(leftNode // 2, 1) if leftNode != -1 else None
                return [
                    (selfTransLabel, node - 1),
                    (neighTransLabel, nextNode)
                ]

@codeDeps(ProbLeftToRightZeroNet)
def probLeftToRightZeroNet(elems, transLabels):
    return ProbLeftToRightZeroNet(len(elems), elemFor = lambda state: elems[state], transLabelFor = lambda state, adv: transLabels[state][adv])

@codeDeps(ProbLeftToRightZeroNet)
def sumGeometricZeroDurNet(elem, transLabels):
    numDurStates = len(transLabels)
    return ProbLeftToRightZeroNet(numDurStates, elemFor = lambda durState: elem, transLabelFor = lambda durState, adv: transLabels[durState][adv])

@codeDeps(Net, basicChecks)
class SemiMarkovDurNet(Net):
    def __init__(self, elemm, minDur, maxDur, transLabelFor):
        self.elemm = elemm
        self.minDur = minDur
        self.maxDur = maxDur
        self.transLabelFor = transLabelFor

        basicChecks(self)
    def start(self, forwards):
        return -1 if forwards else 0
    def elem(self, node):
        if node == -1 or node == 0:
            return None
        else:
            return self.elemm
    def next(self, node, forwards):
        if node < -1 or node > self.maxDur:
            raise RuntimeError('bad node id ('+str(node)+') for net '+repr(self))
        if forwards:
            if node == -1:
                ret = []
                for dur in range(self.minDur, self.maxDur + 1):
                    label = self.transLabelFor(dur)
                    ret.append((label, dur))
                return ret
            elif node == 0:
                return []
            else:
                return [(None, node - 1)]
        else:
            if node == -1:
                return []
            else:
                ret = []
                if node >= self.minDur:
                    label = self.transLabelFor(node)
                    ret.append((label, -1))
                if node < self.maxDur:
                    ret.append((None, node + 1))
                return ret

@codeDeps(Net, basicChecks)
class ConcreteNet(Net):
    """A net where nodes are 0, 1, ..., (N - 1)."""
    def __init__(self, startNode, endNode, elems, edgesForwards, edgesBackwards = None):
        self.startNode = startNode
        self.endNode = endNode
        self.elems = elems
        self.edgesForwards = edgesForwards

        self.numNodes = len(elems)
        assert 0 <= startNode < self.numNodes
        assert 0 <= endNode < self.numNodes
        assert len(edgesForwards) == self.numNodes

        if edgesBackwards is None:
            self.edgesBackwards = [ [] for node in range(self.numNodes) ]
            for node in range(self.numNodes):
                for label, nextNode in self.edgesForwards[node]:
                    assert 0 <= nextNode < self.numNodes
                    self.edgesBackwards[nextNode].append((label, node))
        else:
            self.edgesBackwards = edgesBackwards

        assert len(self.edgesBackwards) == self.numNodes
        basicChecks(self)
    def __repr__(self):
        return 'ConcreteNet('+', '.join(map(repr, [self.startNode, self.endNode, self.elems, self.edgesForwards, self.edgesBackwards]))+')'
    def start(self, forwards):
        return self.startNode if forwards else self.endNode
    def elem(self, node):
        return self.elems[node]
    def next(self, node, forwards):
        return self.edgesForwards[node] if forwards else self.edgesBackwards[node]

@codeDeps(ConcreteNet)
def concretizeNet(net, nodes):
    """Concretizes an existing net.

    nodes should be an iterable of nodes in net.

    Returns a ConcreteNet with the same structure as the sub-net defined by net
    and nodes. The order of nodes in the iterable is taken into account, so the
    first node becomes node 0, the second node becomes node 1, etc.
    """
    indexToNode = list(nodes)
    nodeToIndex = dict([ (node, index) for index, node in enumerate(indexToNode) ])

    startIndex = nodeToIndex[net.start(True)]
    endIndex = nodeToIndex[net.start(False)]
    elems = [ net.elem(node) for node in indexToNode ]

    edgesForwards = [ [ (label, nodeToIndex[nextNode]) for label, nextNode in net.next(node, forwards = True) if nextNode in nodeToIndex ] for node in indexToNode ]
    edgesBackwards = [ [ (label, nodeToIndex[nextNode]) for label, nextNode in net.next(node, forwards = False) if nextNode in nodeToIndex ] for node in indexToNode ]

    return ConcreteNet(startNode = startIndex, endNode = endIndex, elems = elems, edgesForwards = edgesForwards, edgesBackwards = edgesBackwards)

@codeDeps(concretizeNet, ForwardRef(lambda: nodeSetCompute))
def concretizeNetSimple(net, accessibleOnly = True):
    nodeSet = nodeSetCompute(net, accessibleOnly = accessibleOnly)
    if not nodeSet:
        raise RuntimeError('no path from start to end for net '+repr(net))
    return concretizeNet(net, nodeSet)

@codeDeps(concretizeNet, ForwardRef(lambda: topSort))
def concretizeNetTopSort(net, deltaTime):
    sortedNodes = topSort(net, deltaTime)
    return concretizeNet(net, sortedNodes)
@codeDeps()
def netIsTopSorted(net, nodeSet, deltaTime):
    for node in nodeSet:
        for forwards in [True, False]:
            for label, nextNode in net.next(node, forwards):
                if nextNode in nodeSet and deltaTime(label) == 0:
                    leftNode = node if forwards else nextNode
                    rightNode = nextNode if forwards else node
                    if leftNode >= rightNode:
                        return False
    return True

@codeDeps()
class HasCycleError(Exception):
    pass

@codeDeps(HasCycleError)
def topSort(net, deltaTime, forwards = False, detectCycles = True, debugCheckInvariants = False):
    """Topologically sorts the nodes in a net.

    Only non-emitting edges (those with deltaTime(label) == 0) count for the
    topological sort. If detectCycles is True and a non-emitting cycle is
    present then a HasCycleError will be raised.
    """
    rootsRemaining = deque([net.start(forwards)])
    seen = set()
    agenda = []
    if detectCycles:
        agendaSet = set()
    sortedNodes = []

    def splitParents(node):
        noDeltaParents = []
        deltaParents = []
        for label, nextNode in net.next(node, forwards):
            if deltaTime(label) == 0:
                noDeltaParents.append(nextNode)
            else:
                deltaParents.append(nextNode)
        return noDeltaParents, deltaParents

    def agendaAdd(node):
        revNoDeltaParents, deltaParents = splitParents(node)
        revNoDeltaParents.reverse()
        agenda.append((node, revNoDeltaParents))
        for deltaParent in deltaParents:
            # below seen check just for efficiency (still correct without it)
            if deltaParent not in seen:
                rootsRemaining.append(deltaParent)

    # invariants:
    #  - sortedNodes + agenda (as lists) contains no dups
    #  - sortedNodes + agenda == seen (as sets)
    #  - agenda is a path through net from a root to some node (backwards
    #    in default forwards == False case)
    # complexity is O(#nodes + #edges) since for each node we have (uniformly)
    #   O(number-of-edges-coming-from-that-node) instructions
    # (FIXME : is this complexity still true now that we have delta edges?
    #   Are there some pathological cases where delta edges give a rootsRemaining
    #   that grows really really big with lots of duplicates??)
    while rootsRemaining:
        root = rootsRemaining.popleft()
        if root not in seen:
            seen.add(root)
            assert not agenda
            agendaAdd(root)
            if detectCycles:
                assert not agendaSet
                agendaSet.add(root)
            while agenda:
                if debugCheckInvariants:
                    agendaNodes = [ node for node, parentsRemaining in agenda ]
                    seenAgain = set(sortedNodes + agendaNodes)
                    assert len(seenAgain) == len(sortedNodes) + len(agendaNodes)
                    assert seenAgain == seen
                    assert agendaNodes[0] == root
                    for fromNode, toNode in zip(agendaNodes, agendaNodes[1:]):
                        assert toNode in [ nextNode for label, nextNode in net.next(fromNode, forwards) if deltaTime(label) == 0 ]
                node, parentsRemaining = agenda[-1]
                if parentsRemaining:
                    nextNode = parentsRemaining.pop()
                    if detectCycles and nextNode in agendaSet:
                        path = [ node for node, parentsRemaining in agenda ]
                        cycle = path[path.index(nextNode):]
                        if not forwards:
                            cycle.reverse()
                        raise HasCycleError('cycle detected: '+repr(cycle))
                    if nextNode not in seen:
                        seen.add(nextNode)
                        agendaAdd(nextNode)
                        if detectCycles:
                            agendaSet.add(nextNode)
                else:
                    # if all parents are seen, then each parent is either on the agenda,
                    #   in which case we have a cycle, or in sortedNodes. Assuming
                    #   no cycles this means it is safe to add this node to sortedNodes.
                    node, parentsRemaining = agenda.pop()
                    if detectCycles:
                        agendaSet.remove(node)
                    sortedNodes.append(node)

    if forwards:
        sortedNodes.reverse()
    return sortedNodes

@codeDeps(ForwardRef(lambda: nodeSetComputeSub))
def nodeSetCompute(net, accessibleOnly = True):
    """Traverses a net to compute the set of nodes.

    If accessibleOnly is True, returns the set of nodes accessible both going
    forwards from start and going backwards from end. If accessibleOnly is
    False, returns the set of nodes connected to either start or end,
    traversing the net in both directions at every stage.
    """
    if accessibleOnly:
        nodeSet = nodeSetComputeSub(net, [True])
        nodeSet.intersection_update(nodeSetComputeSub(net, [False]))
        return nodeSet
    else:
        return nodeSetComputeSub(net, [True, False])
@codeDeps()
def nodeSetComputeSub(net, dirs):
    nodeSet = set()
    agenda = deque()

    def add(node):
        if node not in nodeSet:
            agenda.append(node)
            nodeSet.add(node)

    for forwards in dirs:
        add(net.start(forwards))
    while agenda:
        node = agenda.popleft()

        for forwards in dirs:
            for label, nextNode in net.next(node, forwards):
                add(nextNode)

    return nodeSet

@codeDeps(ForwardRef(lambda: isConsistent))
def checkConsistent(net, nodeSet):
    if not isConsistent(net, nodeSet):
        raise RuntimeError('net inconsistent (looks different viewed forwards and backwards)')
@codeDeps()
def isConsistent(net, nodeSet, verbose = False):
    """Checks if net looks the same viewed forwards and backwards.

    More precisely, checks if the sub-net defined by net and nodes looks the
    same viewed forwards and backwards.

    Copes fine with the case where there may be multiple edges between a given
    pair of nodes.
    """
    arrowsForwards = defaultdict(list)
    arrowsBackwards = defaultdict(list)

    def arrows(forwards):
        return arrowsForwards if forwards else arrowsBackwards

    for forwards in [True, False]:
        for node in nodeSet:
            for label, nextNode in net.next(node, forwards):
                if nextNode in nodeSet:
                    leftNode = node if forwards else nextNode
                    rightNode = nextNode if forwards else node
                    arrows(forwards)[(leftNode, rightNode)].append(label)

    if len(arrowsForwards) != len(arrowsBackwards):
        return False
    for nodePair in arrowsForwards:
        if nodePair not in arrowsBackwards:
            return False
        if sorted(arrowsForwards[nodePair]) != sorted(arrowsBackwards[nodePair]):
            if verbose:
                print 'NOTE: net inconsistent:', sorted(arrowsForwards[nodePair]), 'forwards vs', sorted(arrowsBackwards[nodePair]), 'backwards for node pair', nodePair
            return False
    return True

@codeDeps(nodeSetCompute, ForwardRef(lambda: toDotGeneral))
def toDot(net, accessibleOnly = True, highlighted = lambda node: False):
    nodeSet = nodeSetCompute(net, accessibleOnly = accessibleOnly)
    if not nodeSet:
        raise RuntimeError('no path from start to end for net '+repr(net))

    return toDotGeneral(net, nodeSet, highlighted = highlighted)
@codeDeps(concretizeNet, isConsistent)
def toDotGeneral(net, nodeSet, highlighted = lambda node: False):
    assert isConsistent(net, nodeSet)
    net = concretizeNet(net, nodeSet)

    sb = []

    def nodeLine(node, label):
        return '\tnode [ label="'+label+'",'+styleString(node)+' ]; '+str(node)+';'
    def edgeLine(leftNode, rightNode, label):
        return '\t'+str(leftNode)+' -> '+str(rightNode)+' [ label="'+label+'" ];'
    def styleString(node):
        if net.elem(node) is None:
            return ','.join([
                'shape=circle' if node != net.start(True) and node != net.start(False) else 'shape=doublecircle',
                'style=filled,fillcolor=black' if not highlighted(node) else 'style=filled,fillcolor=red',
                'height=0.1',
                'fixedsize=true'
            ])
        else:
            return ','.join([
                'shape=ellipse',
                'style=solid' if not highlighted(node) else 'style=filled,fillcolor=red',
                'height=0.0',
                'fixedsize=false'
            ])

    sb += [ 'digraph Net {', '\trankdir=LR;' ]
    for node in range(0, net.numNodes):
        elem = net.elem(node)
        sb.append(nodeLine(node, str(elem) if elem is not None else ''))
    for node in range(0, net.numNodes):
        for label, nextNode in net.next(node, forwards = True):
            sb.append(edgeLine(node, nextNode, str(label) if label is not None else ''))
    sb.append('}')

    return '\n'.join(sb)

@codeDeps(Net, basicChecks)
class UnrolledNet(Net):
    """Unrolls given net over time.

    The time change in traversing an edge forwards is given by deltaTime
    (the range of this function should be the non-negative integers). The
    labels of the resulting net are (label, labelStartTime, labelEndTime)
    triples.

    Note that there is no necessary connection between emitting nodes and time
    changes. In fact, this function is intended to be used with nets that
    consist of non-emitting nodes.
    """
    def __init__(self, netTop, startTime, endTime, deltaTime):
        self.netTop = netTop
        self.startTime = startTime
        self.endTime = endTime
        self.deltaTime = deltaTime

        assert endTime >= startTime
        basicChecks(self)
    def start(self, forwards):
        return (self.startTime if forwards else self.endTime, self.netTop.start(forwards))
    def elem(self, (time, nodeTop)):
        return self.netTop.elem(nodeTop)
    def next(self, (time, nodeTop), forwards):
        ret = []
        for label, nextNodeTop in self.netTop.next(nodeTop, forwards):
            nextTime = time + self.deltaTime(label) * (1 if forwards else -1)
            if self.startTime <= nextTime <= self.endTime:
                labelStartTime, labelEndTime = (time, nextTime) if forwards else (nextTime, time)
                ret.append(((label, labelStartTime, labelEndTime), (nextTime, nextNodeTop)))
        return ret

@codeDeps(Net, basicChecks)
class ReweightedNet(Net):
    def __init__(self, net, labelToWeight, divisionRing, beta):
        self.net = net
        self.labelToWeight = labelToWeight
        self.divisionRing = divisionRing
        self.beta = beta

        basicChecks(self)
    def start(self, forwards):
        return self.net.start(forwards)
    def elem(self, node):
        return self.net.elem(node)
    def next(self, node, forwards):
        ring = self.divisionRing
        ret = []
        for label, nextNode in self.net.next(node, forwards):
            weight = self.labelToWeight(label)
            leftWeight = self.beta[node if forwards else nextNode]
            rightWeight = self.beta[nextNode if forwards else node]
            newWeight = ring.ldivide(leftWeight, ring.times(weight, rightWeight)) if leftWeight != ring.zero else ring.zero
            if newWeight != ring.zero:
                ret.append(((label, newWeight), nextNode))
        return ret
@codeDeps(ReweightedNet, ForwardRef(lambda: sumGetAlpha))
def reweight(net, labelToWeight, divisionRing, getAgenda):
    totalWeight, beta = sumGetAlpha(net, labelToWeight, ring = divisionRing, getAgenda = getAgenda, forwards = False)
    rnet = ReweightedNet(net, labelToWeight, divisionRing = divisionRing, beta = beta)
    return totalWeight, rnet

@codeDeps()
class SumAgenda(object):
    def __nonzero__(self):
        abstract
    def add(self, node, weight):
        abstract
    def pop(self):
        abstract
    # (FIXME : add transform method (signature (node, weight) => weight)?)
    def printStats(self):
        pass
@codeDeps(SumAgenda)
class SimpleSumAgenda(SumAgenda):
    def __init__(self, ring, useQueue = True):
        self.ring = ring
        self.useQueue = useQueue

        self.active = dict()
        self.queue = deque()
    def __nonzero__(self):
        return bool(self.active)
    def add(self, node, weight):
        if node in self.active:
            self.active[node] = self.ring.plus(self.active[node], weight)
        else:
            self.queue.append(node) if self.useQueue else self.queue.appendleft(node)
            self.active[node] = weight
    def pop(self):
        node = self.queue.popleft()
        weight = self.active[node]
        del self.active[node]
        return node, weight
@codeDeps(SumAgenda)
class PriorityQueueSumAgenda(SumAgenda):
    """SumAgenda backed by a priority queue.

    When using a net that is topologically sorted there are guaranteed to be no
    repeat pops.

    negMap should take a node and "make it negative", in the sense that
    node1 < node2 iff negMap(node1) > negMap(node2). negMap is used when
    forwards == False as a very ugly hack to turn a min heap into a max heap.
    """
    # (FIXME : find a nicer solution to the max heap problem!)
    def __init__(self, ring, forwards, negMap, pruneThresh = None, pruneTrigger = None):
        self.ring = ring
        self.forwards = forwards
        self.negMap = negMap
        self.pruneThresh = pruneThresh
        self.pruneTrigger = pruneTrigger

        self.active = dict()
        self.heap = []
        self.nodePrevPop = None
    def __nonzero__(self):
        return bool(self.active)
    def add(self, node, weight):
        if node in self.active:
            self.active[node] = self.ring.plus(self.active[node], weight)
        else:
            heapq.heappush(self.heap, node if self.forwards else self.negMap(node))
            self.active[node] = weight
    def pop(self):
        node = heapq.heappop(self.heap)
        if not self.forwards:
            node = self.negMap(node)
        # it's ok to pop node from heap before doing pruning since pruning doesn't touch heap
        if self.nodePrevPop is not None and self.pruneThresh is not None and self.pruneTrigger is not None and self.pruneTrigger(self.nodePrevPop, node):
            self.prune()
        weight = self.active[node]
        del self.active[node]
        self.nodePrevPop = node
        return node, weight
    def prune(self):
        thresh = self.pruneThresh
        best = self.ring.max(self.active.itervalues())
        for node in self.active:
            weight = self.active[node]
            if self.ring.lt(self.ring.times(weight, thresh) if self.forwards else self.ring.times(thresh, weight), best):
                self.active[node] = self.ring.zero
@codeDeps(SumAgenda)
class TimeSyncSumAgenda(SumAgenda):
    # (FIXME : does stack (rather than queue) really work well enough for TimeSyncSumAgenda that we should have it as a default?)
    #   (note that stack definitely does have an advantage in avoiding repeat pops when dealing with ~flatmapped zero-duration stuff~)
    def __init__(self, ring, forwards, startTime, endTime, nodeToTime, useQueue = False, pruneThresh = None):
        self.ring = ring
        self.forwards = forwards
        self.startTime = startTime
        self.endTime = endTime
        self.nodeToTime = nodeToTime
        self.useQueue = useQueue
        self.pruneThresh = pruneThresh

        self.active = dict()
        self.queue = defaultdict(deque)
        self.time = self.startTime if self.forwards else self.endTime
    def __nonzero__(self):
        return bool(self.active)
    def add(self, node, weight):
        if node in self.active:
            self.active[node] = self.ring.plus(self.active[node], weight)
        else:
            time = self.nodeToTime(node)
            if time < self.time and self.forwards or time > self.time and not self.forwards:
                raise RuntimeError('cannot insert agenda item into past')
            self.queue[time].append(node) if self.useQueue else self.queue[time].appendleft(node)
            self.active[node] = weight
    def pop(self):
        if not self.active:
            raise IndexError('pop from an empty agenda')
        while not self.queue[self.time]:
            self.time += (1 if self.forwards else -1)
            assert self.startTime <= self.time <= self.endTime
            if self.pruneThresh is not None:
                self.prune()
        node = self.queue[self.time].popleft()
        weight = self.active[node]
        del self.active[node]
        return node, weight
    def prune(self):
        thresh = self.pruneThresh
        queue = self.queue[self.time]
        if not queue:
            return
        best = self.ring.max([ self.active[node] for node in queue ])

        queueNew = []
        for node in queue:
            weight = self.active[node]
            if self.ring.lt(self.ring.times(weight, thresh) if self.forwards else self.ring.times(thresh, weight), best):
                del self.active[node]
            else:
                queueNew.append(node)
        self.queue[self.time] = deque(queueNew)
@codeDeps(SumAgenda)
class TrackRepeatPopsSumAgenda(SumAgenda):
    def __init__(self, sumAgenda, verbose = True):
        self.sa = sumAgenda
        self.verbose = verbose

        self.popped = defaultdict(int)
        self.pops = 0
    # (FIXME : doesn't seem quite right to access active directly (and wouldn't need to if we had a transform method))
    @property
    def active(self):
        return self.sa.active
    def __nonzero__(self):
        return bool(self.sa)
    def add(self, node, weight):
        self.sa.add(node, weight)
    def pop(self):
        node, weight = self.sa.pop()
        self.popped[node] += 1
        self.pops += 1
        return node, weight
    def summary(self):
        nodesPopped = len(self.popped)
        return nodesPopped, self.pops
    def topOffenders(self, n = 10):
        # (FIXME : come up with faster way to implement this if it's useful)
        return reversed(sorted([ (count, node) for node, count in self.popped.items() if count > 1 ])[-n:])
    def printStats(self):
        poppedNodes, pops = self.summary()
        if self.verbose or pops != poppedNodes:
            print 'SumAgenda: repeat pop stats:', pops, 'pops for', poppedNodes, 'unique nodes'
        if self.verbose and pops != poppedNodes:
            print 'SumAgenda: repeat pop stats: top offenders:'
            for count, node in self.topOffenders():
                print 'SumAgenda:\t', node, '->', count, 'times'

@codeDeps()
def sum(net, labelToWeight, ring, getAgenda, forwards = True):
    """Sums over all paths in the given net.

    forwards is direction in which to compute the sum.

    Assumes net doesn't have any cycles.

    labelToWeight function should be memoized if it is expensive to compute.
    """
    agenda = getAgenda(forwards)

    agenda.add(net.start(forwards), ring.one)
    totalWeight = ring.zero
    endNode = net.end(forwards)

    while agenda:
        node, weight = agenda.pop()
        if weight != ring.zero:
            if node == endNode:
                totalWeight = ring.plus(totalWeight, weight)
            for label, nextNode in net.next(node, forwards):
                linkWeight = labelToWeight(label)
                newWeight = ring.times(weight, linkWeight) if forwards else ring.times(linkWeight, weight)
                if newWeight != ring.zero:
                    agenda.add(nextNode, newWeight)

    agenda.printStats()
    return totalWeight
@codeDeps()
def sumGetAlpha(net, labelToWeight, ring, getAgenda, forwards = True):
    """Sums over all paths in the given net, storing alpha for all nodes.

    alpha value of a node is sum over all partial paths ending with that node.

    forwards is direction in which to compute the sum.

    Assumes net doesn't have any cycles.
    """
    agenda = getAgenda(forwards)

    alpha = defaultdict(lambda: ring.zero)
    agenda.add(net.start(forwards), ring.one)

    while agenda:
        node, weight = agenda.pop()
        if weight != ring.zero:
            for label, nextNode in net.next(node, forwards):
                linkWeight = labelToWeight(label)
                newWeight = ring.times(weight, linkWeight) if forwards else ring.times(linkWeight, weight)
                if newWeight != ring.zero:
                    agenda.add(nextNode, newWeight)
            if node in alpha:
                alpha[node] = ring.plus(alpha[node], weight)
            else:
                alpha[node] = weight

    agenda.printStats()
    return alpha[net.end(forwards)], alpha
@codeDeps()
def sumYieldGamma(net, labelToWeight, divisionRing, totalWeight, beta, getAgenda, forwards = True):
    """Computes gamma values for labelled edges in the given net.

    Returns an iterator over (label, gamma) pairs.

    totalWeight is the sum over all paths.
    beta should be a function which returns the beta value for any node
    (for a reweighted net the beta value will always be divisionRing.one).
    forwards is direction in which to do search.

    The gamma value of an edge is the "occupancy" of that edge -- the sum over
    all paths that pass through that edge divided by the sum over all paths
    (e.g. as used during expectation-maximization). (For a non-commutative
    division ring the division is left-division, so the occupancy is the inverse
    of (the sum over all paths) times (the sum over all paths that pass through
    the given edge) ). For the probabilistic semiring gamma is the probability
    that a random path passes through that edge.

    Due to the implementation of the algorithm multiple (label, gamma) pairs
    may be produced for the same labelled edge, in which case the gamma value
    for that edge is the sum of the given gamma values produced for that edge.

    Assumes net doesn't have any cycles.
    """
    ring = divisionRing
    agenda = getAgenda(forwards)

    agenda.add(net.start(forwards), ring.one)
    totalWeightAgain = ring.zero
    endNode = net.end(forwards)

    while agenda:
        node, weight = agenda.pop()
        if weight != ring.zero:
            if node == endNode:
                totalWeightAgain = ring.plus(totalWeightAgain, weight)
            for label, nextNode in net.next(node, forwards):
                linkWeight = labelToWeight(label)
                newWeight = ring.times(weight, linkWeight) if forwards else ring.times(linkWeight, weight)
                betaWeight = beta(nextNode)
                if newWeight != ring.zero and betaWeight != ring.zero:
                    agenda.add(nextNode, newWeight)
                    edgeTotalWeight = ring.times(newWeight, betaWeight) if forwards else ring.times(betaWeight, newWeight)
                    gamma = ring.ldivide(totalWeight, edgeTotalWeight)
                    yield label, gamma

    if not ring.isClose(totalWeight, totalWeightAgain):
        logging.warning('recomputed total weight ('+str(totalWeightAgain)+') differs from given value ('+str(totalWeight)+')')
    agenda.printStats()

@codeDeps(sumGetAlpha, sumYieldGamma)
def forwardBackward(net, labelToWeight, divisionRing, getAgenda, forwardsFirst = False):
    """Performs Forward-Backward algorithm on net.

    Returns total weight, and an iterator over (label, gamma) pairs.

    See notes for sumYieldGamma for iterator details.
    """
    totalWeight, beta = sumGetAlpha(net, labelToWeight = labelToWeight, ring = divisionRing, getAgenda = getAgenda, forwards = forwardsFirst)
    edgeGen = sumYieldGamma(net, labelToWeight = labelToWeight, divisionRing = divisionRing, totalWeight = totalWeight, beta = lambda node: beta[node], getAgenda = getAgenda, forwards = not forwardsFirst)
    return totalWeight, edgeGen
@codeDeps(reweight, sumYieldGamma)
def forwardBackwardAlt(net, labelToWeight, divisionRing, getAgenda):
    totalWeight, rnet = reweight(net, labelToWeight = labelToWeight, divisionRing = divisionRing, getAgenda = getAgenda)
    edgeGen = sumYieldGamma(rnet, labelToWeight = lambda (label, weight): weight, divisionRing = divisionRing, totalWeight = divisionRing.one, beta = lambda node: divisionRing.one, getAgenda = getAgenda, forwards = True)
    # remove extra part of label added by reweighting
    edgeGen = ( (label, logOcc) for (label, _), logOcc in edgeGen )
    return totalWeight, edgeGen
