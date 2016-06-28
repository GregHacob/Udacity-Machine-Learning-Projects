# Train a Smartcab How to Drive

## Synopsis
Smartcab is a self-driving car from the not-so-distant future that transfer customers from one arbitrary location to another. This application demonstrates how to use **reinforcement learning** Q-learning to train a smartcab through trial and error.

## Installation

This project requires Python 2.7x with the [Pygame](https://www.pygame.org/wiki/GettingStarted) library installed.

Notes on installation of Pygame:

- If using [anaconda](https://www.continuum.io/downloads): `conda install -c https://conda.anaconda.org/quasiben pygame`; or search the conda repo: `anaconda search -t conda pygame`.
- [This tutorial](http://kidscancode.org/blog/2015/09/pygame_install/) is very helpful.

[Pandas](http://pandas.pydata.org/) library is also required -- make sure you have it properly installed.

## Run

Make sure you are in the **lowest-level** project directory `./smartcab/smartcab` (that contains `agent.py`). Then run:

```python smartcab/agent.py```

There are **options** to be passed as follows:
* `-b`: Run Basic Agent. The Basic Agent is an agent with completely random moves. 
* `-s`: Run Smart Agent. The smart Agent is an agent that learns the environment, traffic laws, and reached the destination on time with enough trials.
* `-t`: The number of trials to be run (each trial is a new "game" with new deadline and destination). Q-function is accumulated and averaged over the trials so that later trials are expected to perform better than initial ones. Default is trials=100 if ignored.
* `-d`: Control the simulation speed, lower delay for higher speed. Default is 1s.
* `-p:` Preserve Q_table data in a csv file
* `-r:` Read from preserved file csv file
* `-h`: Read from previously saved history file to speed up the learning process.
* `-o`: Write out the **"Smartcab Performance Report.csv"** file



## Demos

The following scenarios are considered for demonstrations with parameters:

* Run Basic Agent : `python smartcab/agent.py -b`
* Run Smart Agent with default values: `python smartcab/agent.py -s`

    **defailt values:** `init_value = 0, epsilon=0.2, alpha=0.9, gamma=0.4, simulator speed = 1, number of trials = 100`
* Run Smart Agent with 50 trials and .05s delay: `python smartcab/agent.py -s -t50 -d0.05`
* Run Smart Agent with 50 trials, .05s delay, hyper parameters, and the output file: `python smartcab/agent.py -s -t50 -d0.05 -h -o`
    **hyper parameetrs:**
    hyper_params = {"init_values":[0],
                        "epsilons":[.1,.2,.4],
                        "alphas":[.3,.7,.9],
                        "gammas":[.4,.7,.9]}

    **Note:** This command generates a file called **Smartcab Perfrmance Report.csv** that lists performance of all parameter combinations:
* Run Smart Agent with 50 trials, .05s delay, and preserve utility value data `python smartcab/agent.py -s -t50 -d0.05 -p`
* Run Smart Agent and read from a preserved utility values data `python smartcab/agent.py -r'q_table_1467072690.csv`


