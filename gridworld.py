
# Minni Bin and Ziming Yan


# grid_world.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu) and Ioannis Karamouzas (ioannis@g.clemson.edu)

"""
In this assignment, you will implement three classic algorithm for 
solving Markov Decision Processes either offline or online. 
These algorithms include: value_iteration, policy_iteration and q_learning.
You will test your implementation on three grid world environments. 
You will also have the opportunity to use Q-learning to control a simulated robot 
in crawler.py

The package `matplotlib` is needed for the program to run.

The Grid World environment has discrete state and action spaces
and allows for both model-based and model-free access.

It has the following properties:
    env.observation_space.n     # the number of states
    env.action_space.n          # the number of actions
    env.trans_model             # the transition/dynamics model

In value_iteration and policy_iteration, you can access the transition model 
at a given state s and action by calling
    t = env.trans_model[s][a]
where s is an integer in the range [0, env.observation_space.n),
      a is an integer in the range [0, env.action_space.n), and
      t is a list of four-element tuples in the form of
        (p, s_, r, terminal)
where s_ is a new state reachable from the state s by taking the action a,
      p is the probability to reach s_ from s by a, i.e. p(s_|s, a),
      r is the reward of reaching s_ from s by a, and
      terminal is a boolean flag to indicate if s_ is a terminal state.

In q_learning, once a terminal state is reached the environment should be (re)initialized by
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
You can also only update the visualization of the v values by
    logger.log(i, v)
"""


import random
import numpy as np

def value_iteration(env, gamma, max_iterations, logger):
    """
    Implement value iteration to return a deterministic policy for all states.
    See lines 20-40 for details.  

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the reward discount factor
    max_iterations: integer
        the maximum number of value iterations that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
        Hint: The value iteration may converge before reaching max_iterations.  
        In this case, you may want to exit the algorithm earlier. A way to check 
        if value iteration has already converged is to check whether 
        the max over (or sum of) L1 or L2 norms between the values before and
        after an iteration is small enough. For the Grid World environment, 1e-4
        is an acceptable tolerance.
    logger: app.grid_world.App.Logger
        a logger instance to perform test and record the iteration process
    
    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """


    NUM_STATES = env.observation_space.n                                   # number states=12
    NUM_ACTIONS = env.action_space.n                                       # number of action=4
    TRANSITION_MODEL = env.trans_model                                     # transition model

    # initialization
    v = [0] * NUM_STATES                                                   # initialize v value for all states
    pi = [0] * NUM_STATES                                                  # initialize optimal action for alll states
    logger.log(0, v, pi)                                                   # visualize the v and pi

    ### Please finish the code below ##############################################
    ####################################################################################################################

    for k in range(max_iterations):

        v_old=v.copy()                                                     # old v value used to calculate stop threshold

        for s in range(NUM_STATES):

            t=TRANSITION_MODEL[s][0]

            if len(t)==1:                                                   #  see if state is terminal state, if so, v[s] and pi[s]=0

                v[s]=0

                pi[s]=0

            else:
                actions=[]                                                   # store actions for each state and later pick up the optimal one

                V=[]                                                         # store Q vule for each state and later pick up the maximum one

                for a in range(NUM_ACTIONS):

                    t=TRANSITION_MODEL[s][a]                                 # (p, s_, r, terminal)

                    v_s=0

                    for n in t:

                        v_n=n[0]*(n[2]+gamma*v_old[n[1]])

                        v_s +=v_n

                    V.append(v_s)

                    actions.append(a)

                v_star=max(np.asarray(V))                                    # pick up max v value

                pi[s]=actions[np.argmax(V)]                                  # pick up optimal action for current state

                v[s]=v_star                                                  # update v value

        logger.log(k+1,v,pi)                                                 # visualize v and pi

        error=np.asarray(v)-np.asarray(v_old)                                # converge criterion: sum(v_old-v_new)<1e-4

        stop=sum(abs(error))

        if stop<1e-4:
            return pi                                                       # converge and return policy
    ####################################################################################################################



def policy_iteration(env, gamma, max_iterations, logger):
    """
    Implement policy iteration to return a deterministic policy for all states.
    See lines 20-40 for details.  

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the reward discount factor
    max_iterations: integer
        the maximum number of policy iterations that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
        Hint 1: Policy iteration may converge before reaching max_iterations. 
        In this case, you should exit the algorithm. A simple way to check 
        if the algorithm has already converged is by simply checking whether
        the policy at each state hasn't changed from the previous iteration.
        Hint 2: The value iteration during policy evaluation usually converges 
        very fast and policy evaluation should end upon convergence. A way to check 
        if policy evaluation has converged is to check whether the max over (or sum of) 
        L1 or L2 norm between the values before and after an iteration is small enough. 
        For the Grid World environment, 1e-4 is an acceptable tolerance.
    logger: app.grid_world.App.Logger
        a logger instance to record and visualize the iteration process.
        During policy evaluation, the V-values will be updated without changing the current policy; 
        here you can update the visualization of value by simply calling logger.log(i, v).
    
    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """


    NUM_STATES = env.observation_space.n                                        # number of states
    NUM_ACTIONS = env.action_space.n                                            # number of actions
    TRANSITION_MODEL = env.trans_model                                          # transition model

    v = [0.0] * NUM_STATES                                                      # initialize the v value for each state
    pi = [random.randint(0, NUM_ACTIONS-1)] * NUM_STATES                        # create a random policy for each state
    logger.log(0, v, pi)                                                        # Visualize the initial value and policy

    ### Please finish the code below ##############################################
    ####################################################################################################################
    for k in range(max_iterations):

        converge=False

        p_old=pi.copy()                                                         # old policy used to compared with new policy


        while not converge:                                                     # policy evaluation untill v is converge and policy is stable

            v_old=v.copy()

            for s in range(NUM_STATES):

                a=p_old[s]                                                     # action for each sate

                t=TRANSITION_MODEL[s][a]

                if len(t)==1:                                                 # check whether it is terminal or not
                    v[s]=0
                else:                                                         # calculate v value for policy
                    t=TRANSITION_MODEL[s][a]

                    v_s=0

                    for n in t:
                        v_n=n[0]*(n[2]+gamma*v_old[n[1]])
                        v_s +=v_n

                    v[s]=v_s                                                  # update v value

            error=np.asarray(v)-np.asarray(v_old)                             # if policy stable and go to policy improvement

            stop=sum(abs(error))

            if stop<1e-4:
                converge=True



        # policy improvement -check policy stable or not. if not go to policy evaluation

        for s in range(NUM_STATES):

            t=TRANSITION_MODEL[s][0]

            if len(t)==1:                                                   # check whether it is terminal state or not?
                v[s]=0
                pi[s]=0

            else:
                actions=[]

                V_a=[]

                for a in range(NUM_ACTIONS):                                # calcualte v value for each action
                    t=TRANSITION_MODEL[s][a]
                    v_s=0

                    for n in t:
                        v_n=n[0]*(n[2]+gamma*v[n[1]])
                        v_s +=v_n

                    V_a.append(v_s)
                    actions.append(a)

                V_a=np.asarray(V_a)

                p=actions[np.argmax(V_a)]                                     # optimal action for current state

                if p!=p_old[s]:                                               # check policy stable or not

                    pi[s]=p                                                   # update policy

        logger.log(k+1,v,pi)                                                  # visualize result and policy

        if pi==p_old:                                                         # policy stable and stop

           return pi
    ####################################################################################################################




def q_learning(env, gamma, max_iterations, logger):
    """
    Implement Q-learning to return a deterministic policy for all states.
    Please note that in RL you do not have access to the transition model 
    and the reward function, i.e. you cannot call env.trans_model as above. 
    Instead you need to learn policies by collecting samples using env.step
    See lines 40-50 for more details. 

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the discount factor
    max_iterations: integer
        the maximum number of iterations (training episodes) that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
    logger: app.grid_world.App.Logger
        a logger instance to perform test and record the iteration process.
    
    Returns
    -------
    In q_learning, once a terminal state is reached the environment should be (re)initialized by
    s = env.reset()
    where s is the initial state.
    An experience (sample) can be collected from s by taking an action a as follows:
    s_, r, terminal, info = env.step(a)
    where s_ is the resulted state by taking the action a,
      r is the reward achieved by taking the action a,
      terminal is a boolean flag to indicate if s_ is a terminal state, and
      info is just used to keep compatible with openAI gym library.

    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    s_, r, terminal, info = env.step(a)
    """

    NUM_STATES = env.observation_space.n                                        # number of state
    NUM_ACTIONS = env.action_space.n                                            # number of action

    v = [0] * NUM_STATES                                                        # initialization v value
    pi = [0] * NUM_STATES                                                       # initialize optimal action
    logger.log(0, v, pi)                                                        # Visualize the initial value and policy

    #########################
    # Adjust superparameters as you see fit
    # parameter for the epsilon-greedy method to trade off exploration and exploitation
    eps = 1
    # learning rate for updating q values based on sample estimates
    alpha = 0.1

    ### Please finish the code below ##############################################
    ####################################################################################################################

    q_table = np.zeros((NUM_STATES, NUM_ACTIONS))                                 # create q_table to store Q values

    sample=0                                                                     # sample <=max_iteration

    converge=False

    while not converge:                                                          # episodes not converge, continue

        s = env.reset()                                                          # terminal is true, reset initial state

        terminal=False

        while not terminal:

            sample +=1

            # epsilon greedy to trade off exploration and exploitation
            # there are two methods to deal with ε
            # 1.start with ε=1 and after a certain number of iterations set ε to a low value (e.g., ε=0.1)
            # 2.linearly decay ε over time. In here, we use exponential decay to linear decay ε

            eps=np.power(0.99999,sample-1)

            if np.random.uniform(0, 1) < eps:                             # act randomly

                a = int(np.random.randint(0, NUM_ACTIONS))

            else:
                a = np.argmax(q_table[s])                                 # act best action

            new_state, r, terminal, info = env.step(a)

            if terminal:
                target = r
            else:
                target = r + gamma*np.max(q_table[new_state])           # new Q[s,a]

            q_table[s,a] = (1 - alpha)* q_table[s,a] + alpha * target   # update Q[s,a] with learning rate alpha

            s = new_state                                               # update state

            v[s] = np.max(q_table[s])                                   # update v value

            pi[s] = np.argmax(q_table[s])                               # update pi

            if sample == max_iterations:                                # when sample == max_iterations, stop learning

                logger.log(sample,v,pi)

                return pi                                               # return policy

        logger.log(sample,v,pi)                                         # visualize v and pi for each episode
    ####################################################################################################################

if __name__ == "__main__":

    from app.grid_world import App
    import tkinter as tk

    algs = {
        "Value Iteration": value_iteration,
        "Policy Iteration": policy_iteration,
        "Q Learning": q_learning
   }
    worlds = {
        # o for obstacle
        # s for start cell
        "world1": App.DEFAULT_WORLD,
        "world2": lambda : [
            [10, "s", "s", "s", 1],
            [-10, -10, -10, -10, -10],
        ],
        "world3": lambda : [
            ["_", "_", "_", "_", "_"],
            ["_", "o", "_", "_", "_"],
            ["_", "o",   1, "_",  10],
            ["s", "_", "_", "_", "_"],
            [-10, -10, -10, -10, -10]
        ]
    }

    root = tk.Tk()
    App(algs, worlds, root)
    tk.mainloop()