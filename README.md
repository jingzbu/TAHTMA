TAHTMA
======

TAHTMA is short for Threshold Approximation for Hoeffding Test under Markovian Assumption.


Usage
=====
Please type `./tahtma -h` to get the following help message:
```
usage: tahtma [-h] [--beta BETA] [--N N] [--fig_dir FIG_DIR]
              [--show_pic SHOW_PIC]

optional arguments:
  -h, --help           show this help message and exit
  --beta BETA          false alarm rate for Hoeffding's rule; default=0.001
  --N N                number of states in the original Markov chain;
                       default=4
  --fig_dir FIG_DIR    folder for saving the output plot; default='./Results/'
  --show_pic SHOW_PIC  whether or not to show the output plot; use 'T' or 'F'
                       to indicate; default='F'
```

Examples:

 `jzh@jzh:~/Research/Anomaly_Detection/TAHTMA$ ./tahtma --beta 0.1 --N 3`

 `jzh@jzh:~/Research/Anomaly_Detection/TAHTMA$ ./tahtma --beta 0.01 --N 4 --show_pic T`

 `jzh@jzh:~/Research/Anomaly_Detection/TAHTMA$ ./tahtma --beta 0.0001 --N 5 --show_pic T`



Author
=============
Jing Zhang

I am now a PhD student in Division of Systems Engineering at Boston University, working with Professor [Yannis Paschalidis](http://ionia.bu.edu/). My research lies in two-fold. One is using applied probability, stochastic processes, and optimization theory to deal with modelling and algorithms development. In particular, currently we are working on a problem regarding an anomaly detection algorithm, which aims to improve the accuracy of the threshold needed by the generalized Hoeffding test. To outperform the existing methods (typically using the large deviations theory), we are trying to establish weak convergence results for certain quantities of interest. The other is using machine learning approaches to tackle health care problems, making predictions of the necessity of hospitalization for people in the US.

EMail: jzh@bu.edu

Personal Webpage: http://people.bu.edu/jzh/


Copyright 2014-2015 Jing Zhang. All rights reserved. TAHTMA is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation.

