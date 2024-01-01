
import torch
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import Trainer
from DeepLearningAgent import DeepQLearning
import gym
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
num_gpus = torch.cuda.device_count()
print(num_gpus)
print(device)

algo=DeepQLearning(gym.make("LunarLander-v2",))

trainer=Trainer(accelerator="cpu",max_epochs=10_000,
                callbacks=[EarlyStopping(monitor="episode/return",mode="max",patience=500)])
trainer.fit(algo)









