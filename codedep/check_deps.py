#!/usr/bin/python -u
"""Checks whether code-level dependencies are correctly declared."""

# Copyright 2011, 2012, 2013 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

import os
import sys
import inspect
import _ast
import ast
import symtable
import importlib

def peekIter(it):
    it = iter(it)
    elem = it.next()
    done = False
    while not done:
        try:
            elemNext = it.next()
        except StopIteration:
            elemNext = None
            done = True
        yield elem, elemNext
        elem = elemNext

def attachAstAndSymtab(nodes, symtab, depth = 0):
    """Walks an AST and a symtable at the same time, linking them.

    A scope_depth attribute is added to each symtab giving the number of levels
    of nested scope above the current symtab.
    A symtab_current_scope attribute is added to each AST node giving the symtab
    for the scope that is current at that node.
    """
    symtab.scope_depth = depth
    symtabChildrenLeft = list(reversed(symtab.get_children()))
    for node in nodes:
        attachAstAndSymtabSub(node, symtab, symtabChildrenLeft, depth = depth)
    assert not symtabChildrenLeft

def attachAstAndSymtabSub(node, symtab, symtabChildrenLeft, depth):
    node.symtab_current_scope = symtab
    if isinstance(node, (_ast.FunctionDef, _ast.ClassDef, _ast.Lambda, _ast.GeneratorExp)):
        # new scope introduced, and used in some of the children

        # (FIXME : order of descent into new scopes may not be correct (that is,
        #   may disagree with ordering used by symtable) for complicated nested
        #   scope cases. Need to think about.)
        if isinstance(node, _ast.FunctionDef):
            subNodesOldScope = node.args.defaults + node.decorator_list
            subNodesNewScope = node.args.args + node.body
            # node.args itself would otherwise be missed out
            node.args.symtab_current_scope = symtab
        elif isinstance(node, _ast.ClassDef):
            subNodesOldScope = node.bases + node.decorator_list
            subNodesNewScope = node.body
        elif isinstance(node, _ast.Lambda):
            subNodesOldScope = node.args.defaults
            subNodesNewScope = node.args.args + [node.body]
            # node.args itself would otherwise be missed out
            node.args.symtab_current_scope = symtab
        elif isinstance(node, (_ast.SetComp, _ast.DictComp, _ast.GeneratorExp)):
            subNodesOldScope = [ subNode.iter for subNode in node.generators ]
            subNodesNewScope = (
                ([node.key, node.value] if isinstance(node, _ast.DictComp) else [node.elt]) +
                [ subNode.target for subNode in node.generators ] +
                [ subSubNode for subNode in node.generators for subSubNode in subNode.ifs ]
            )
            # each node in node.generators would otherwise be missed out
            for subNode in node.generators:
                subNode.symtab_current_scope = symtab

        for subNode in subNodesOldScope:
            attachAstAndSymtabSub(subNode, symtab, symtabChildrenLeft, depth = depth)

        symtabChild = symtabChildrenLeft.pop()
        if isinstance(node, (_ast.FunctionDef, _ast.ClassDef)):
            assert symtabChild.get_name() == node.name
        attachAstAndSymtab(subNodesNewScope, symtabChild, depth = depth + 1)
    else:
        for subNode in ast.iter_child_nodes(node):
            attachAstAndSymtabSub(subNode, symtab, symtabChildrenLeft, depth = depth)

def isGlobal(symtab, symName):
    """Returns True if symbol referred to by symName in symtab is global.

    This is provided to work around the fact that Symbol.is_global() is False
    for module-level variables accessed from module-level.
    """
    sym = symtab.lookup(symName)
    return symtab.get_type() == 'module' or sym.is_global()

def findGlobalUses(node, onLoadGlobalFound):
    curr_symtab = node.symtab_current_scope
    if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Name) and node.func.id == 'codeDeps':
        # ignore names present in arguments to codeDeps
        for subNode in ast.iter_child_nodes(node):
            findGlobalUses(subNode, lambda name: ())
    elif isinstance(node, _ast.Name) and isGlobal(curr_symtab, node.id):
        if isinstance(node.ctx, (_ast.Load, _ast.AugLoad)):
            onLoadGlobalFound(node.id)
        # no children
    elif isinstance(node, _ast.Attribute) and isinstance(node.value, _ast.Name) and isGlobal(curr_symtab, node.value.id):
        if isinstance(node.ctx, (_ast.Load, _ast.AugLoad)):
            onLoadGlobalFound(node.value.id+'.'+node.attr)
        else:
            onLoadGlobalFound(node.value.id)
        # all children have already been dealt with
    else:
        for subNode in ast.iter_child_nodes(node):
            findGlobalUses(subNode, onLoadGlobalFound)

def assignsNames(node):
    if isinstance(node, (_ast.FunctionDef, _ast.ClassDef)):
        return [node.name]
    elif isinstance(node, _ast.Assign):
        ret = []
        for subNode in node.targets:
            if isinstance(subNode, _ast.Name):
                assert isinstance(subNode.ctx, _ast.Store)
                ret.append(subNode.id)
        return ret
    else:
        return []

def simpleAssignToName(node):
    """If a simple assignment to a name, return name, otherwise None."""
    if isinstance(node, _ast.Assign) and len(node.targets) == 1:
        targetNode = node.targets[0]
        if isinstance(targetNode, _ast.Name):
            return targetNode.id
        else:
            return None
    else:
        return None

def prettyPrintBisqueDepsStanza(deps, init = '@', maxLineLength = 80):
    if not deps:
        return init+'codeDeps()'
    else:
        ret = init+'codeDeps('+(', '.join(deps))+')'
        if len(ret) <= maxLineLength:
            return ret
        else:
            ret = ''
            currLine = init+'codeDeps('+deps[0]+','
            for dep in deps[1:]:
                if len(currLine) + len(dep) + 2 <= maxLineLength:
                    currLine += (' '+dep+',')
                else:
                    ret += (currLine+'\n')
                    currLine = '    '+dep+','
            ret += currLine[:-1]+'\n)'
            return ret

def main(args):
    srcRootDir = os.path.abspath(args[1])
    moduleName = args[2]

    sys.stderr.write('(using srcRootDir = %s)\n' % srcRootDir)

    module = importlib.import_module(moduleName)

    moduleFile = os.path.abspath(inspect.getsourcefile(module))
    moduleFileContents = file(moduleFile).read()
    moduleFileLines = moduleFileContents.split('\n')
    assert moduleFileLines[-1] == ''
    moduleFileLines = moduleFileLines[:-1]

    sys.stderr.write('(module %s from %s)\n' % (moduleName, moduleFile))

    nodeModule = ast.parse(moduleFileContents, moduleFile, 'exec')
    symtab = symtable.symtable(moduleFileContents, moduleFile, 'exec')

    attachAstAndSymtab(nodeModule.body, symtab)

    sys.stderr.write('\n')
    sys.stderr.write('FINDING GLOBALS:\n')

    loadGlobalss = []
    for node in nodeModule.body:
        loadGlobals = []
        def onLoadGlobalFound(name):
            if '.' in name:
                nameLeft, nameRight = name.split('.', 1)
                nameLeftObj = eval(nameLeft, vars(module))
                if inspect.ismodule(nameLeftObj):
                    loadGlobals.append(name)
                else:
                    loadGlobals.append(nameLeft)
            else:
                loadGlobals.append(name)
        findGlobalUses(node, onLoadGlobalFound)
        loadGlobals = sorted(set(loadGlobals))
        loadGlobalss.append(loadGlobals)

    sys.stderr.write('\n')
    sys.stderr.write('REWIRING DEPS FOR PRIVATE VARIABLES:\n')

    privateDeps = dict()
    for nodeIndex, node in enumerate(nodeModule.body):
        # expand any private variables which are in loadGlobals for this node
        loadGlobals = loadGlobalss[nodeIndex]
        newLoadGlobals = set()
        for name in loadGlobals:
            if name.startswith('_') and not name.startswith('__'):
                if name in privateDeps:
                    newLoadGlobals.update(privateDeps[name])
                else:
                    newLoadGlobals.add(name)
                    sys.stderr.write('NOTE: treating %s as non-private\n' % name)
            else:
                newLoadGlobals.add(name)
        loadGlobalss[nodeIndex] = sorted(newLoadGlobals)

        # add current node to privateDeps if appropriate
        nameAssignedTo = simpleAssignToName(node)
        if nameAssignedTo is not None and nameAssignedTo.startswith('_'):
            # statement is simple assignment to a private variable, i.e. of the form '_bla = ...'
            privateDeps[nameAssignedTo] = loadGlobals
            sys.stderr.write('will expand %s to %s\n' % (nameAssignedTo, loadGlobals))

    sys.stderr.write('\n')
    sys.stderr.write('RESOLVING LOCATIONS:\n')
    namesDefinedInModule = set([ subNode for node in nodeModule.body for subNode in assignsNames(node) ])

    names = set()
    for node, loadGlobals in zip(nodeModule.body, loadGlobalss):
        names.update(loadGlobals)
    namesWithinRoot = set()
    for name in names:
        if name in namesDefinedInModule:
            namesWithinRoot.add(name)
        elif name in ('True', 'False'):
            pass
        else:
            if '.' in name:
                nameLeft, nameRight = name.split('.', 1)
                nameLeftObj = eval(nameLeft, vars(module))
                assert inspect.ismodule(nameLeftObj)
                if hasattr(nameLeftObj, '__file__'):
                    sourceFileRel = inspect.getsourcefile(nameLeftObj)
                else:
                    # built-in module (according to code in inspect.py)
                    sourceFileRel = None
            else:
                try:
                    nameObj = eval(name, vars(module))
                except NameError:
                    sys.stderr.write('NOTE: %s refers to nothing (ignoring)\n' % name)
                    nameObj = None
                if nameObj is None:
                    sourceFileRel = None
                elif inspect.isbuiltin(nameObj) or getattr(nameObj, '__module__', None) == '__builtin__':
                    sourceFileRel = None
                elif inspect.ismodule(nameObj) and not hasattr(nameObj, '__file__'):
                    # built-in module (according to code in inspect.py)
                    sourceFileRel = None
                elif inspect.isclass(nameObj) and not hasattr(sys.modules.get(nameObj.__module__), '__file__'):
                    # built-in class (according to code in inspect.py)
                    sourceFileRel = None
                else:
                    try:
                        sourceFileRel = inspect.getsourcefile(nameObj)
                    except TypeError:
                        sourceFileRel = None
                        sys.stderr.write('NOTE: %s had no source file\n' % name)
            if sourceFileRel is not None:
                sourceFile = os.path.abspath(sourceFileRel)
                if os.path.commonprefix([srcRootDir, sourceFile]) == srcRootDir:
                    namesWithinRoot.add(name)

    sys.stderr.write('\n')
    sys.stderr.write('RESULTS:\n')
    sys.stderr.write('\n')

    namesNotYetDefined = set(namesDefinedInModule)

    def nameToString(name):
        if name in namesNotYetDefined:
            return 'ForwardRef(lambda: '+name+')'
        else:
            return name

    # print stuff which occurs before the first node
    bodyStartLine = (nodeModule.body[0].lineno - 1) if nodeModule.body else len(moduleFileLines)
    for line in moduleFileLines[:bodyStartLine]:
        print line

    for (node, nextNode), loadGlobals in zip(peekIter(nodeModule.body), loadGlobalss):
        # (module docstrings which last more than one line seem to have col_offset -1)
        assert node.col_offset == 0 or node.col_offset == -1
        if nextNode is not None:
            assert nextNode.col_offset == 0
        startLine = node.lineno - 1
        endLine = (nextNode.lineno - 1) if nextNode is not None else len(moduleFileLines)
        assert 0 <= startLine < endLine <= len(moduleFileLines)
        sourceLines = moduleFileLines[startLine:endLine]

        if isinstance(node, (_ast.FunctionDef, _ast.ClassDef)):
            sortedDeps = [ nameToString(name) for name in loadGlobals if name in namesWithinRoot and name != node.name ]

            # work around the fact there is no completely reliable way to get the
            #   line number of the first non-decorator line from the AST alone
            defLineOffset = None
            for lineOffset, line in enumerate(sourceLines):
                if line.startswith('def ') or line.startswith('class '):
                    defLineOffset = lineOffset
                    break
            assert defLineOffset is not None
            assert all([ (dec.lineno - 1) < startLine + defLineOffset for dec in node.decorator_list ])

            codeDepsPrinted = False

            # print decorator lines
            for subNode, nextSubNode in peekIter(node.decorator_list):
                if isinstance(subNode, _ast.Call) and isinstance(subNode.func, _ast.Name) and subNode.func.id == 'codeDeps':
                    # print new codeDeps stanza in place of old one
                    print prettyPrintBisqueDepsStanza(sortedDeps)
                    codeDepsPrinted = True
                else:
                    startSubOffset = subNode.lineno - 1 - startLine
                    endSubOffset = (nextSubNode.lineno - 1 - startLine) if nextSubNode is not None else defLineOffset
                    assert 0 <= startSubOffset < endSubOffset <= len(sourceLines)
                    for line in sourceLines[startSubOffset:endSubOffset]:
                        print line
            if not codeDepsPrinted:
                # print codeDeps stanza just before def
                print prettyPrintBisqueDepsStanza(sortedDeps)
                codeDepsPrinted = True

            # print rest of function / class
            for line in sourceLines[defLineOffset:]:
                print line
        else:
            sortedDeps = [ nameToString(name) for name in loadGlobals if name in namesWithinRoot ]
            if sortedDeps:
                nameAssignedTo = simpleAssignToName(node)
                if nameAssignedTo != None and nameAssignedTo.startswith('_'):
                    # node is a simple assignment to a private variable, so don't need a codeDeps line
                    for line in sourceLines:
                        print line
                else:
                    if nameAssignedTo != None and node:
                        nodeValue = node.value

                        hadToRemoveExisting = False
                        if isinstance(nodeValue, _ast.Call) and isinstance(nodeValue.func, _ast.Call) and isinstance(nodeValue.func.func, _ast.Name) and nodeValue.func.func.id == 'codeDeps':
                            # remove existing codeDeps stanza
                            # (FIXME : assumes there is an arg (and that it comes before kwargs, etc, but I think that's safe))
                            nodeValue = nodeValue.args[0]
                            hadToRemoveExisting = True

                        currLineOffset = nodeValue.lineno - 1 - startLine
                        restOfCurrLine = sourceLines[currLineOffset][nodeValue.col_offset:]
                        restOfLines = sourceLines[(currLineOffset + 1):]
                        if not hadToRemoveExisting:
                            # if codeDeps is already present, then user must
                            #   have added, so no need to warn
                            print '# FIXME : to examine manually (assumes original RHS is function or class)'
                        print prettyPrintBisqueDepsStanza(sortedDeps, init = nameAssignedTo+' = ')+'('
                        print '    '+restOfCurrLine
                        for line in restOfLines:
                            print line
                        if not hadToRemoveExisting:
                            print ')'
                    else:
                        print '# FIXME : to examine manually -- codeDeps(%s)' % (', '.join(sortedDeps))
                        for line in sourceLines:
                            print line
            else:
                for line in sourceLines:
                    print line

        namesNotYetDefined.difference_update(assignsNames(node))


if __name__ == '__main__':
    main(sys.argv)
