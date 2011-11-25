"""Functions for querying and manipulating DAGs of nodes.

Each node in DAG is represented by an object with a `children` and a
`mapChildren` method.
Node identity is (by default) represented by python object identity, which is
based on memory location.
"""

# Copyright 2011 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

# (N.B. lookup code (for sharing) is not concurrent-safe, and doesn't detect loops)

def nodeList(
    parentNode,
    idValue = lambda node: id(node),
    includeNode = lambda node: True,
    includeDescendants = lambda node: True
):
    ret = []
    agenda = [parentNode]
    lookup = dict()
    while agenda:
        node = agenda.pop()
        ident = idValue(node)
        if ident in lookup:
            #print 'DEBUG: nodeList: re-using shared value', ident, '->', lookup[ident]
            pass
        if not ident in lookup:
            lookup[ident] = True
            if includeNode(node):
                ret.append(node)
            if includeDescendants(node):
                agenda.extend(reversed(node.children()))
    return ret

def findTaggedNodes(parentNode, f):
    def includeNode(node):
        try:
            tag = node.tag
        except AttributeError:
            return False
        else:
            return f(tag)
    return nodeList(parentNode, includeNode = includeNode)
def findTaggedNode(parentNode, f):
    nodes = findTaggedNodes(parentNode, f)
    if len(nodes) == 0:
        raise RuntimeError('node not found')
    elif len(nodes) > 1:
        raise RuntimeError('more than one node found ('+str(len(nodes))+' nodes, with tags '+repr([ node.tag for node in nodes ])+')')
    else:
        return nodes[0]

# (FIXME : could make this more concrete and live with code duplication if that improves clarity)
def getDagMap(
    partialMaps,
    idValue = lambda args: id(args[0]),
    storeValue = lambda ret, args: ret,
    restoreValue = lambda stored, args: stored
):
    """Map an object DAG by recursively applying a function to each sub-DAG.

    This function provides a flexible mechanism to traverse an object DAG,
    possibly creating a new object DAG with similar structure.
    The basic idea is that each sub-DAG should be pretty much arbitrarily
    mappable to a new sub-DAG, but that it should be easy to cope with the
    common case where a sub-DAG is mapped by recursively mapping its
    descendants.

    Control is passed between this function, which copes with the fact a DAG
    can have shared nodes, and a function defined by partialMaps that
    essentially uses pattern matching to map each sub-tree.
    Typically the function defined by partialMaps calls a callback (defined
    in this function) to map each child node of a given node, and so the
    overall DAG is traversed recursively with control passing between this
    function and the partialMap for each node.
    However each partialMap is free not to use the callback, and so can in fact
    map sub-trees arbitrarily.

    Specifically each partialMap is a partial function which takes as input
    a node and a closure (callback).
    By convention the partialMap returns None where (and only where) it is
    undefined.
    To apply partialMaps to a node, each partialMap is tried in turn, and
    the first matching partial function is used.
    Typically each partialMap calls the given closure on each child node, then
    uses the values returned and the node input to return a new value, often a
    new DAG.

    The closure defined by this function tracks which nodes have already been
    mapped, and uses this to map shared nodes appropriately (as defined by
    idValue, storeValue and restoreValue).

    The envisioned form of partialMaps is a sequence of partial functions for
    special cases followed by a general fall-through function which calls
    class-specific helper functions that implement sensible default behaviour.

    N.B. care must be taken that a partialMap does not return None where it is
    defined (this is most likely to happen when the partialMap makes some
    mutable change and does not have a natural return value).
    """
    def dagMap(*parentArgs):
        lookup = dict()
        def mapChild(*args):
            ident = idValue(args)
            if ident in lookup:
                #print 'DEBUG: dagMap: re-using shared value', ident, '->', lookup[ident]
                return restoreValue(lookup[ident], args)
            else:
                ret = None
                for partialMap in partialMaps:
                    ret = partialMap(*(args + (mapChild,)))
                    if ret is not None:
                        break
                if ret is None:
                    raise RuntimeError('none of the given partial functions was defined at input '+repr(args))
                lookup[ident] = storeValue(ret, args)
                return ret
        return mapChild(*parentArgs)
    return dagMap

def defaultMapPartial(node, mapChild):
    return node.mapChildren(mapChild)
defaultMap = getDagMap([defaultMapPartial])
