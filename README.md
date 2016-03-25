TAHTMA
======

TAHTMA is short for Threshold Approximation for Hoeffding's Test under Markovian Assumption.

If you are interested in our recent publication (see below) on network anomaly detection and want to use it as reference, please cite the repository [TAHTMA](https://github.com/jingzbu/TAHTMA) together with:

Jing Zhang and I. Ch. Paschalidis, "***An Improved Composite Hypothesis Test for Markov Models with Applications in Network Anomaly Detection***," Proceedings of the 54th IEEE Conference on Decision and Control, pp. 3810-3815, December 15-18, 2015, Osaka, Japan.



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

Jing Zhang currently is a PhD student in the Division of Systems Engineering at Boston University, advised by Professor [Yannis Paschalidis](http://sites.bu.edu/paschalidis/).


Email: `jzh@bu.edu`

Homepage: http://people.bu.edu/jzh/


Copyright 2014-2015 Jing Zhang. All rights reserved. TAHTMA is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software Foundation.
