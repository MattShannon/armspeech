"""Implementation of lazy properties."""

# This file is part of armspeech.
# See `License` for details of license and warranty.

from codedep import codeDeps

@codeDeps()
class lazyproperty(object):
    """(based on http://code.activestate.com/recipes/363602-lazy-property-evaluation/)"""
    def __init__(self, calculate_function):
        self._calculate = calculate_function

    def __get__(self, obj, _ = None):
        if obj is None:
            return self
        value = self._calculate(obj)
        setattr(obj, self._calculate.func_name, value)
        return value
