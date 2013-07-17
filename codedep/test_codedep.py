"""Unit tests for code-level dependency tracking."""

# Copyright 2011, 2012, 2013 Matt Shannon

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

from codedep.hash import hashString

import unittest

class TestCodedep(unittest.TestCase):
    def test_hashString_characterize(self):
        """Tests hashString function has not changed."""
        assert hashString('') == 'e69de29bb2d1d6434b8b29ae775ad8c2e48c5391'
        assert hashString('"') == '9d68933c44f13985b9eb19159da6eb3ff0e574bf'
        assert hashString('abc') == 'f2ba8f84ab5c1bce84a7b441cb1959cfc7093b7f'
        assert (hashString('aRvo;ui') ==
                'af7aa2bc77f03b961a67b633f93674331eab36c8')

def suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestCodedep)

if __name__ == '__main__':
    unittest.main()
