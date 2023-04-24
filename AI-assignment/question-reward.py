# Define states and actions
states = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'End']
actions = ['continue', 'stop']

# Define reward matrix
# reward = {'S1': {'correct': 100, 'stop': 0, 'wrong': 0},
#           'S2': {'correct': 500, 'stop': 100, 'wrong': -50},
#           'S3': {'correct': 1000, 'stop': 600, 'wrong': -250},
#           'S4': {'correct': 5000, 'stop': 1600, 'wrong': -500},
#           'S5': {'correct': 10000, 'stop': 2100, 'wrong': -2500},
#           'S6': {'correct': 50000, 'stop': 12100, 'wrong': -5000},
#           'S7': {'correct': 100000, 'stop': 62100, 'wrong': -25000},
#           'S8': {'correct': 500000, 'stop': 162100, 'wrong': -50000},
#           'S9': {'correct': 1000000, 'stop': 662100, 'wrong': -250000},
#           'S10': {'correct': 5000000, 'stop': 1662100, 'wrong': -500000},
#           'End': {'correct': 0, 'stop': 0, 'wrong': 0}}

# reward = {'S1': {'correct': 100, 'stop': 0, 'wrong': 0},
#           'S2': {'correct': 500, 'stop': 100, 'wrong': 0},
#           'S3': {'correct': 1000, 'stop': 600, 'wrong': 0},
#           'S4': {'correct': 5000, 'stop': 1600, 'wrong': 0},
#           'S5': {'correct': 10000, 'stop': 2100, 'wrong': 0},
#           'S6': {'correct': 50000, 'stop': 12100, 'wrong': 0},
#           'S7': {'correct': 100000, 'stop': 62100, 'wrong': 0},
#           'S8': {'correct': 500000, 'stop': 162100, 'wrong': 0},
#           'S9': {'correct': 1000000, 'stop': 662100, 'wrong': 0},
#           'S10': {'correct': 5000000, 'stop': 1662100, 'wrong': 0},
#           'End': {'correct': 0, 'stop': 0, 'wrong': 0}}

reward = {'S1': {'correct': 100, 'stop': 0, 'wrong': -100},
          'S2': {'correct': 500, 'stop': 100, 'wrong': -500},
          'S3': {'correct': 1000, 'stop': 600, 'wrong': -1000},
          'S4': {'correct': 5000, 'stop': 1600, 'wrong': -5000},
          'S5': {'correct': 10000, 'stop': 2100, 'wrong': -10000},
          'S6': {'correct': 50000, 'stop': 12100, 'wrong': -50000},
          'S7': {'correct': 100000, 'stop': 62100, 'wrong': -100000},
          'S8': {'correct': 500000, 'stop': 162100, 'wrong': -500000},
          'S9': {'correct': 1000000, 'stop': 662100, 'wrong': -1000000},
          'S10': {'correct': 5000000, 'stop': 1662100, 'wrong': -5000000},
          'End': {'correct': 0, 'stop': 0, 'wrong': 0}}

# Defining the transition matrix
transition_probability = {'S1': {'continue': {'S2': 0.99}},
                          'S2': {'continue': {'S3': 0.9}},
                          'S3': {'continue': {'S4': 0.8}},
                          'S4': {'continue': {'S5': 0.7}},
                          'S5': {'continue': {'S6': 0.6}},
                          'S6': {'continue': {'S7': 0.5}},
                          'S7': {'continue': {'S8': 0.4}},
                          'S8': {'continue': {'S9': 0.3}},
                          'S9': {'continue': {'S10': 0.2}},
                          'S10': {'continue': {'S10': 0.1}},
                          'End': {'continue': {'End': 1}}}

# Define the initial state
state = 'S1'
# Define the discount factor
gamma = 1
# Define the learning rate
alpha = 0.1
# Define the number of episodes
episodes = 1000
# Initialize the value function
V = {s: 0 for s in states}


# Define the Bellman's Equation for value iteration
def bellman_equation(curr_state, action, expected_reward):
    if action == 'stop':
        return reward[curr_state]['stop']
    else:
        # accounting for both correct and wrong answer
        for next_state in transition_probability[curr_state][action].keys():
            expected_reward += transition_probability[curr_state][action][next_state] * (reward[curr_state]['correct'] + gamma * V[next_state]) + (1 - transition_probability[curr_state][action][next_state]) * (reward[curr_state]['wrong'])
        return expected_reward


# Run the value iteration algorithm
for i in range(episodes):
    expected_reward = 0
    delta = 0
    for s in states:
        v = V[s]
        max_q = -float('inf')
        for a in actions:
            q = bellman_equation(s, a, expected_reward)
            if q > max_q:
                max_q = q
        V[s] = max_q
        delta = max(delta, abs(v - V[s]))
    if delta < 1e-3:
        break

# Print the optimal policy and values
print('Optimal Policy:')
for s in states[:-1]:
    max_q = -float('inf')
    optimal_action = None
    for a in actions:
        q = bellman_equation(s, a, expected_reward)
        if q > max_q:
            max_q = q
            optimal_action = a
    print(s, ':', optimal_action)

print('\nOptimal Values:')
for s in states[:-1]:
    print(s, ':', V[s])
