import numpy as np

def DynaQ(env, gamma=0.99, step_size=0.1, epsilon=0.1, max_episode=1000, max_model_step=10):
    # Initialize q(s,a) and Model(s,a)
    # q_table = np.random.rand(env.n_states, env.n_actions)
    q_table = np.random.rand(env.n_states, env.n_actions)
    
    # Set Q-values for all goal states to 0
    for g in env.target:
        q_table[g] = 0
        
    model = {}
    
    for i in range(max_episode):
        state = env.reset()[0]
        done = False
        while not done:
            if np.random.rand() < epsilon:
                action = np.random.choice(env.n_actions)
            else:
                action = np.argmax(q_table[state])
            next_state, reward, done, _, _ = env.step(action)
            if not done:
                reward -= 0.01
            if done:
                max_next_q = 0
            else:
                max_next_q = np.max(q_table[next_state])
            current_q = q_table[state, action]
            target_value = reward + gamma * max_next_q
            q_table[state, action] = current_q + step_size * (target_value - current_q)
            model[(state, action)] = (reward, next_state)
            # Planning steps
            for x in range(max_model_step):
                if len(model) > 0:
                    model_keys = list(model.keys())
                    model_state, model_action = model_keys[np.random.randint(len(model_keys))]
                    model_reward, model_next_state = model[(model_state, model_action)]
                    
                    if model_next_state in env.target:
                        model_max_next_q = 0
                    else:
                        model_max_next_q = np.max(q_table[model_next_state])
                    model_current_q = q_table[model_state, model_action]
                    model_target_value = model_reward + gamma * model_max_next_q
                    q_table[model_state, model_action] = model_current_q + step_size * (model_target_value - model_current_q)
            state = next_state
    
    policy = np.zeros((env.n_states, env.n_actions))
    for i in range(env.n_states):
        best_action = np.argmax(q_table[i])
        policy[i, best_action] = 1.0
    
    return policy, q_table.reshape(-1, 1)

def SARSA(env, gamma=0.99, step_size=0.1, epsilon=0.1, max_episode=1000):
    q_table = np.random.rand(env.n_states, env.n_actions)
    
    for g in env.target:
        q_table[g] = 0
        
    for i in range(max_episode):
        state = env.reset()[0]
        # Choose first action using epsilon-greedy
        if np.random.rand() < epsilon:
            action = np.random.choice(env.n_actions)
        else:
            action = np.argmax(q_table[state])
        done = False

        while not done:
            next_state, reward, done, _, _ = env.step(action)
            if not done:
                reward -= 0.01
            # Choose next action using epsilon-greedy
            if np.random.rand() < epsilon:
                next_action = np.random.choice(env.n_actions)
            else:
                next_action = np.argmax(q_table[next_state])
            if done:
                target = reward
            else:
                target = reward + gamma * q_table[next_state, next_action]
            q_table[state, action] += step_size * (target - q_table[state, action])
            state = next_state
            action = next_action
    
    policy = np.zeros((env.n_states, env.n_actions))
    for i in range(env.n_states):
        best_action = np.argmax(q_table[i])
        policy[i, best_action] = 1.0
    
    return policy, q_table.reshape(-1, 1)