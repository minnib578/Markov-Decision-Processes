

# crawler.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu) and Ioannis Karamouzas (ioannis@g.clemson.edu)

"""
In this file, you should test your Q-learning implementation on the robot crawler environment 
that we saw in class. It is suggested to test your code in the grid world environments before this one.

The package `matplotlib` is needed for the program to run.


The Crawler environment has discrete state and action spaces
and provides both of model-based and model-free access.

It has the following properties:
    env.observation_space.n     # the number of states
    env.action_space.n          # the number of actions
  

Once a terminal state is reached the environment should be (re)initialized by
    s = env.reset()
where s is the initial state.
An experience (sample) can be collected from s by taking an action a as follows:
    s_, r, terminal, info = env.step(a)
where s_ is the resulted state by taking the action a,
      r is the reward achieved by taking the action a,
      terminal is a boolean flag to indicate if s_ is a terminal state, and
      info is just used to keep compatible with openAI gym library.


A Logger instance is provided for each function, through which you can
visualize the process of the algorithm.
You can visualize the value, v, and policy, pi, for the i-th iteration by
    logger.log(i, v, pi)
"""



import random
import numpy as np



def q_learning(env, logger):
    """
    Implement Q-learning to return a deterministic policy for all states.

    Parameters
    ----------
    env: CrawlerEnv
        the environment
    logger: app.grid_world.App.Logger
        a logger instance to perform test and record the iteration process.
    
    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """

    NUM_STATES = env.observation_space.n                                           # number of states
    NUM_ACTIONS = env.action_space.n                                               # number of actions


    v = [0] * NUM_STATES                                                            # initialization v
    pi = [0] * NUM_STATES                                                           # initialize optimal action
    logger.log(0,v,pi)
    gamma = 0.95   

    #########################
    # Adjust superparameters as you see fit
    # parameter for the epsilon-greedy method to trade off exploration and exploitation
    eps = 1
    # learning rate for updating q values based on sample estimates
    alpha = 0.1
    # maximum number of training iterations
    max_iterations = 5000
    #########################

    ### Please finish the code below ##############################################

    q_table = np.zeros((NUM_STATES, NUM_ACTIONS))                              # q_table for storing Q[s,a]

    s = env.reset()                                                            # initialize state

    converge=False

    sample=0

    # notice:there are no terminal state as gridworld.py and crawler can continue move forward

    while not converge:
        sample +=1
        # epsilon greedy to trade off exploration and exploitation in here, we use exponential decay to linear decay ε
        eps=np.power(0.9992,sample-1)                        #  exponent of eps should between [0.9991-0.9999]  if exponent<0.9991, exploration will be reduce. if exponent>0.9999 exploration will be increase and crawler not move forward efficiently

        if np.random.uniform(0, 1) < eps:                            # act randomly

            a = int(np.random.randint(0, NUM_ACTIONS))

        else:
            a = np.argmax(q_table[s])                                # act best action

        new_state, r, terminal, info = env.step(a)

        target = r + gamma*np.max(q_table[new_state])                # new Q[s,a]

        q_table[s,a] = (1 - alpha)* q_table[s,a] + alpha * target    # update q_table

        s = new_state                                               # update state

        v[s] = np.max(q_table[s])                                   # update v value

        pi[s] = np.argmax(q_table[s])                               # update pi

        if sample==max_iterations:                                 # when sample >max_iterations ,stop

            logger.log(sample,v,pi)

            return pi

        logger.log(sample,v,pi)



if __name__ == "__main__":
    from app.crawler import App
    import tkinter as tk
    
    algs = {
        "Q Learning": q_learning,
    }

    root = tk.Tk()
    App(algs, root)
    root.mainloop()
