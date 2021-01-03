"""
Author: Phil Dreizen
Utility functions go here.
"""

import sys

def eprint(*args, **kwargs):
    """print to stderr"""
    print(*args, file=sys.stderr, **kwargs)

