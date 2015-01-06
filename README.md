TAHTMA
======

Threshold Approximation for Hoeffding Test under Markovian Assumption


Usage
=====
Please type ./tahtma -h to get the following help message:
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

