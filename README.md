TAHTMA
======

TAHTMA is short for Threshold Approximation for Hoeffding Test under Markovian Assumption.


Usage
=====
Assuming having set `tahtma_ROOT` path variable correctly, then type `./tahtma -h` to get the following help message:
```
usage: tahtma [-h] [--beta BETA] [--N N] [--fig_dir FIG_DIR]
              [--show_pic SHOW_PIC]

optional arguments:
  -h, --help           show this help message and exit
  --beta BETA          false alarm rate for Hoeffding's rule; default=0.001
  --N N                number of states in the original Markov chain;
                       default=4
  --fig_dir FIG_DIR    folder for saving the output plot; default='./Results/'
  --show_pic SHOW_PIC  whether or not to show the output plot; indicated by
                       'T' or 'F'; default='F'
```

Examples:

 `$ ./tahtma --beta 0.1 --N 3`

 `$ ./tahtma --beta 0.01 --N 4 --show_pic T`

 `$ ./tahtma --beta 0.0001 --N 5 --show_pic T`



Author
=============
Jing Zhang

Jing Zhang currently is a PhD student in Division of Systems Engineering at Boston University, working with Professor [Yannis Paschalidis](http://ionia.bu.edu/).


Email: jzh AT bu.edu

Homepage: http://people.bu.edu/jzh/


Copyright 2014-2015 Jing Zhang. All rights reserved. TAHTMA is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software Foundation.

