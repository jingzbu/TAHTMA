#!/usr/bin/env python
"""
This file contains all the simulation experiments
"""

from __future__ import absolute_import, division

__author__ = "Jing Zhang"
__email__ = "jingzbu@gmail.com"
__status__ = "Development"


import sys
import os

ROOT = os.environ.get('tahtma_ROOT')
if ROOT is None:
    print('Please set <tahtma_ROOT> variable in bash.')
    sys.exit()
if not ROOT.endswith('TAHTMA'):
    print('Please set <tahtma_ROOT> path variable correctly. '
          'Please change the name of the <ROOT> folder to TAHTMA. Other name(s) '
          'will cause import problem.')
    sys.exit()
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT.rstrip('TAHTMA'))


from TAHTMA.util.util import visualization

N, beta= map(float, sys.argv[1:3])
visualization(int(N), beta)