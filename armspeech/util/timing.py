"""Time and timing helper functions."""

# This file is part of armspeech.
# See `License` for details of license and warranty.

from datetime import datetime
import time

from codedep import codeDeps

@codeDeps()
def printTime(msg):
    print 'TIMING:', datetime.now(), msg

@codeDeps()
def timed(func, msg = None):
    """(probably based on http://www.daniweb.com/software-development/python/code/216610)"""
    if msg is None:
        msg = func.func_name+' took'
    def ret(*args, **kwargs):
        t1 = time.time()
        res = func(*args, **kwargs)
        t2 = time.time()
        print '%s %0.3f ms' % (msg, (t2 - t1) * 1000.0)
        return res
    return ret
