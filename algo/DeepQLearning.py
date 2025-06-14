#ReplayMemory
import random
from collections import deque

#QLearning
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time

#DQN
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ReplayMemory:
    def __init__(self, memory_size=10000, batch_size=64):
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size

    def push(self, transition):
        """Ajoute une transition à la mémoire."""
        self.memory.append(transition)

    def sample(self):
        """Retourne un échantillon aléatoire de transitions."""
        if self.sufficient_memory():
            return random.sample(self.memory, self.batch_size)
        return random.sample(self.memory, self.__len__())

    def __len__(self):
        """Retourne le nombre de transitions stockées."""
        return len(self.memory)
    
    def sufficient_memory(self):
        if self.__len__() > self.batch_size:
            return True
        return False

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        
        self.embedding = nn.Embedding(input_dim, 5)
        self.fc1 = nn.Linear(5, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = self.embedding(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DeepQLearning:   
    def __init__(self, learning_rate=0.8, gamma=0.95, exploration_prob=0.6, batch_size=64, target_net_update_freq=100, memory_size=10000):
        self.set_learning_rate(learning_rate)
        self.set_gamma(gamma)
        self.set_exploration_prob(exploration_prob)
        self.env = gym.make("Taxi-v3")
        
        self.target_net_update_freq = target_net_update_freq
        
        self.max_memory_size= memory_size
        self.batch_size = batch_size

        #Init la mémoire et les réseaux de neurones
        self.init_memory()
        self.init_net()

        self.metrics = {
            "rewards": [],
            "steps": [],
            "success_rate": [],
            "epochs": 0,
            "training_time": 0
        }
    
    def init_net(self):
        #Init les réseaux de neurones
        input_dim = self.env.observation_space.n
        output_dim = self.env.action_space.n
        self.policy_net = DQN(input_dim, output_dim)
        self.target_net = DQN(input_dim, output_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
    
    def init_memory(self):
        #Init ReplayMemory
        self.memory = ReplayMemory(self.max_memory_size, self.batch_size)

    def train(self, epochs=10000):
        if epochs <= 0:
            raise self.__exception_factory(ValueError, "The number of epochs cannot be 0 or negative !")

        self.metrics = {
            "rewards": [],
            "steps": [],
            "success_rate": 0,
            "epochs": epochs,
            "training_time": 0
        }
        
        #Réinitialise la mémoire et les réseaux de neurones pour l'entraînement
        self.init_memory()
        self.init_net()

        start_time = time.perf_counter()

        for i in range(epochs):
            self.__q_learning_algo(isTraining=True)
            self.exploration_prob = max(0.6, self.exploration_prob - (1 / epochs))
        
        end_time = time.perf_counter()
        self.metrics["training_time"] = end_time - start_time
        print("Entraînement terminé")
        print("Calcul des métriques en cours...")
        self.calculate_metrics()
        print("Métriques calculées")

    def run(self):
        self.env = gym.make("Taxi-v3", render_mode="human")
        self.__q_learning_algo()
        #Redefine the environment to non human in case of futur training of the agent
        self.env = gym.make("Taxi-v3")
    
    def load_model(self, filename):
        try:
            with open(filename, "rb") as f:
                data = pickle.load(f)
                self.set_learning_rate(data["learning_rate"])
                self.set_gamma(data["gamma"])    
                self.set_exploration_prob(data["exploration_prob"])

                #Récupére le réseau de neurone
                self.policy_net = data["policy_net"]
                self.target_net_update_freq = data["target_net_update_freq"]
                
                #Récupére la mémoire
                self.max_memory_size = data["memory_size"]
                self.batch_size = data["batch_size"]
                self.memory = data["memory"]

                self.metrics = data["metrics"]
        except:
            print("An error occured while trying to open the file") 
        

    def save_model(self, filename="model"):
        try:
            data = {
                "policy_net": self.policy_net,
                "metrics": self.metrics,
                "learning_rate": self.learning_rate,
                "gamma": self.gamma,
                "exploration_prob": self.exploration_prob,
                "memory_size": self.max_memory_size,
                "batch_size": self.batch_size,
                "memory": self.memory,
                "target_net_update_freq": self.target_net_update_freq
            }
            with open(filename + ".pickle", "wb") as f:
                pickle.dump(data, f)
        except:
            print("An error occured while trying to save the file") 

    def __q_learning_algo(self, isTraining=False, isCalculate=False):
            state = self.env.reset()
            episode_over = False

            run_reward = 0
            step_count = 0
            state = state[0]
            success = False
            
            while not episode_over:
                
                #Epsilon policy
                rand = np.random.rand()
                if rand < self.exploration_prob and isTraining:
                    action = self.env.action_space.sample()
                else:
                    state_result = torch.tensor([state], dtype=torch.long)
                    q_values = self.policy_net(state_result)
                    action = torch.argmax(q_values).item()
                
                s_, reward, terminated, truncated, info = self.env.step(action)
                
                #Ajoute l'expérience à la mémoire
                self.memory.push((state, action, reward, s_, terminated))
                
                episode_over = terminated or truncated
                success = terminated
                state = s_

                if isTraining: 
                    self.optimize_model()

                    # Update target network periodically
                    if step_count % self.target_net_update_freq == 0:
                        self.target_net.load_state_dict(self.policy_net.state_dict())

                    run_reward += reward
                step_count += 1
                    

            if isCalculate: 
                if run_reward != 0:
                    self.metrics["rewards"].append(run_reward / step_count)
                else:
                    self.metrics["rewards"].append(0)
                self.metrics["steps"].append(step_count)
                if success:
                    self.metrics["success_rate"] += 1

            self.env.close()

    def optimize_model(self):
        if self.memory.sufficient_memory() == False:
            return 
        
        batch = self.memory.sample()
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        state_batch = torch.IntTensor(state_batch)
        action_batch = torch.LongTensor(action_batch).unsqueeze(1)
        reward_batch = torch.IntTensor(reward_batch)
        next_state_batch = torch.IntTensor(next_state_batch)
        done_batch = torch.IntTensor(done_batch)

        # Compute Q-values for current states
        q_values = self.policy_net(state_batch).gather(1, action_batch).squeeze()

        # Compute target Q-values using the target network
        with torch.no_grad():
            max_next_q_values = self.target_net(next_state_batch).max(1)[0]
            target_q_values = reward_batch + self.gamma * max_next_q_values * (1 - done_batch)

        loss = F.smooth_l1_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def calculate_metrics(self):
        for i in range(1000):
            self.__q_learning_algo(isCalculate=True)

    def get_metrics(self):
        return self.metrics

    def show_metrics(self):
        fig, axs = plt.subplots(1, 2, figsize=(200, 5))
        # axs[0].plot(rewards, 'tab:green')
        # axs[0].set_title("Reward")
        axs[0].plot(self.metrics["steps"], 'tab:purple')
        axs[0].set_title("Step Count")

        print("Overall Average reward:", np.mean(self.metrics["rewards"]))
        print("Overall Average number of steps:", np.mean(self.metrics["steps"]))
        print("Success rate (%):", self.metrics["success_rate"] / 1000 * 100)
        print("Number of epochs:", self.metrics["epochs"])
        print("Training Time(in secondes):", self.metrics["training_time"])

        plt.show()

    def set_learning_rate(self, lr):
        if self.__check_is_between_0_and_1(value=lr, name="learning rate"):
            self.learning_rate = lr

    def set_gamma(self, gamma):
        if self.__check_is_between_0_and_1(value=gamma, name="gamma"):
            self.gamma = gamma

    def set_exploration_prob(self, exploration_prob):
        if self.__check_is_between_0_and_1(value=exploration_prob, name="exploration_prob"):
            self.exploration_prob = exploration_prob

    def __check_is_between_0_and_1(self, value, name):
        message = f"The {name} hyperparameter must be between 0 and 1! \n"
        if value > 1:
            message += "Actually he is superior to 1!"
            raise self.__exception_factory(ValueError, message)
        elif value <= 0:
            message += "He cannot be null or negatif!"
            raise self.__exception_factory(ValueError, message)
        return True
            
    def __exception_factory(self, exception, message):
        return exception(message)
