import torch

import gym

env = gym.make("LunarLander-v2")


def train(model, criterion, optimizer,  env,gamma=0.5, epochs=100,max_iter=100):
    model.train()

    for e in range(epochs):
        terminated = False
        done = False
        next_state = 0
        reward = 0
        state, info = env.reset(seed=1)
        state = torch.tensor(state, dtype=torch.float32)
        total_reward=0
        total_loss=0
        for i in range(max_iter):
            # Compute prediction error

            pred = model(state)
            action = torch.argmax(pred).item()

            next_state, reward, done, terminated, info = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            target_q = reward + gamma * torch.max(model(next_state))
            loss = criterion(model(state)[action], target_q)
            # Backpropagation
            total_loss=total_loss+loss;
            total_reward=total_reward+reward
            state = next_state
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if done or terminated:
                break
        print(f"Epoch:{e},loss:{total_loss} total_reward:{total_reward}")


class NeuralNetwork(torch.nn.Module):
    def __init__(self, int_shape, out_shape):
        super().__init__()
        #self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(int_shape, 100)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(100, 100)
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(100, out_shape)

    def forward(self, x):
        #x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x

input=env.observation_space.shape[0]
output=env.action_space.n

model = NeuralNetwork(input, output)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

train(model,criterion,optimizer,env)




