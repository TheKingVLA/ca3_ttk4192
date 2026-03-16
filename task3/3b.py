from matplotlib import pyplot as plt
from gridWorld import gridWorld
import numpy as np

def show_action_value_function(env, Q):
    pos = {"U": (-0.15, -0.3), "D": (-0.15, 0.4), "L": (-0.45, 0.1), "R": (0.05, 0.1)}
    fig = env.render(show_state = False, show_reward = False)            
    for k in env.states():
        s = env.legal_states[k]
        for i, a in enumerate(env.actions(k)):
            fig.axes[0].annotate("{0:.2f}".format(Q[k, i]), (s[1] + pos[a][0], s[0] + pos[a][1]), size = 40/env.board_mask.shape[0], color = "r" if Q[k, i] == max(Q[k, :]) else "k")
    plt.show()
    
def show_policy(env, Q):
    fig = env.render(show_state = False, show_reward = False)
    action_map = {"U": "↑", "D": "↓", "L": "←", "R": "→"}
    for k in env.states():
        s = k if isinstance(k, tuple) else env.legal_states[k]
        if not env.terminal(s):
            fig.axes[0].annotate(action_map[env.actions(s)[np.argmax(Q[k, :])]], (s[1] - 0.1, s[0] + 0.1), size = 100/env.board_mask.shape[0])
    plt.show()


####################  Q-Learning ####################
def Q_Learning(env, gamma, Q, alpha, epsilon):
    # Reset environment
    s, r, done = env.reset()

    """
    YOUR CODE HERE:
    Implement Q-Learning
    
    Input arguments:
        - env     Is the environment
        - gamma   Is the discount rate
        - Q       Is the Q table
        - alpha   Is the learning rate
        - epsilon Is the probability of choosing greedy action
    
    Some useful functions of the grid world environment
        - s_next, r, done = env.step(a)  Take action a and observe the next state, reward and environment termination
        - actions = env.actions()        List available actions in current state (is empty if state is terminal)
    """

     # Fixed action ordering used by the Q-table columns
    action_list = ["U", "D", "L", "R"] # A = {U, D, L, R}
    action_to_idx = {a: i for i, a in enumerate(action_list)}

    while (not done):
        actions = env.actions(s)
        if len(actions) == 0:
            break

        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            # Greedy action among currently available actions
            q_values = np.array([Q[s, action_to_idx[a]] for a in actions])
            best_actions = np.flatnonzero(q_values == np.max(q_values))
            a = actions[np.random.choice(best_actions)]
        else:
            # Explore uniformly over available actions
            a = np.random.choice(actions)

        a_idx = action_to_idx[a]

        # Interact with environment
        s_next, r, done = env.step(a)

        # Q-learning target uses greedy value at next state (off-policy)
        next_actions = env.actions(s_next)
        if done or len(next_actions) == 0:
            td_target = r
        else:
            max_next_q = max(Q[s_next, action_to_idx[a_next]] for a_next in next_actions)
            td_target = r + gamma * max_next_q

        # Q update
        Q[s, a_idx] = Q[s, a_idx] + alpha * (td_target - Q[s, a_idx])

        # Move to next state
        s = s_next

    return Q



if __name__ == "__main__":
    """
    Note that this code requires the numpy and matplotlib packages.
    """

    # Import the environment from file
    filename = "gridworlds/large.json"
    env = gridWorld(filename)

    # Render image
    fig = env.render(show_state = True)
    plt.show()


    """
    (Run Q-Learning)
    
    Below is the code for running Q-Learning, feel free to change the code, and tweek the parameters.
    """
    gamma = 1.0     # Discount rate
    alpha = 0.1     # Learning rate
    epsilon = 0.5   # Probability of taking greedy action
    episodes = 500 # Number of episodes

    Q = np.zeros([len(env.states()), 4])
    for i in range(episodes):
        Q_Learning(env, gamma, Q, alpha, epsilon)

    # Render Q-values and policy 
    show_action_value_function(env, Q)
    show_policy(env, Q)