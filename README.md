TAHTMA
======

TAHTMA is short for Threshold Approximation for Hoeffding Test under Markovian Assumption.


Usage
=====
Assuming having set `tahtma_ROOT` path variable correctly, then type `./tahtma -h` to get the following help message:
```
usage: tahtma [-h] [-e E] [-beta BETA] [-N N] [-fig_dir FIG_DIR] [-show_pic]

optional arguments:
  -h, --help        show this help message and exit
  -e E              experiment type; indicated by 'eta' (threshold calculation
                    and visualization) or 'cdf' (empirical CDF calculation and
                    visualization); default='eta'
  -beta BETA        false alarm rate for Hoeffding's rule; default=0.001
  -N N              number of states in the original Markov chain; default=4
  -fig_dir FIG_DIR  folder for saving the output plot; default='./Results/'
  -show_pic         whether or not to show the output plot; default=False
```

Examples:

 `$ ./tahtma -beta 0.1 -N 3`
 
 `$ ./tahtma -e cdf -beta 0.01 -N 3 -show_pic`

 `$ ./tahtma -beta 0.01 -N 4 -show_pic`

 `$ ./tahtma -beta 0.0001 -N 5 -show_pic`



Author
=============
Jing Zhang

Jing Zhang currently is a PhD student in Division of Systems Engineering at Boston University, working with Professor [Yannis Paschalidis](http://ionia.bu.edu/).


Email: `jzh@bu.edu`

Homepage: http://people.bu.edu/jzh/


Copyright 2014-2015 Jing Zhang. All rights reserved. TAHTMA is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software Foundation.