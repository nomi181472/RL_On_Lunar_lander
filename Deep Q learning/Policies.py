import torch
import numpy as np
import random
def epsilon_greedy(state, env, net,device="cpu" ,epsilon=0.0,):
    action = -1
    if np.random.random() < epsilon:
        action = env.action_space.sample()
    else:
        state = torch.tensor([np.array(state)]).to(device)
        q_values = net(state)
        max_value, action = torch.max(q_values, dim=1)
        action = int(action.item())
    return action
