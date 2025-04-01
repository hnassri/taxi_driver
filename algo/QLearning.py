import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time

class QLearning:
    def __init__(self, learning_rate=0.8, gamma=0.95, exploration_prob=0.6):
        self.set_learning_rate(learning_rate)
        self.set_gamma(gamma)
        self.set_exploration_prob(exploration_prob)
        self.env = gym.make("Taxi-v3")
        self.Q_table = np.zeros([self.env.observation_space.n, self.env.action_space.n])
        self.metrics = {
            "rewards": [],
            "steps": [],
            "success_rate": [],
            "epochs": 0,
            "training_time": 0
        }
    
    def train(self, epochs=1000):
        if epochs <= 0:
            raise self.__exception_factory(ValueError, "The number of epochs cannot be 0 or negative !")

        self.metrics = {
            "rewards": [],
            "steps": [],
            "success_rate": [],
            "epochs": epochs,
            "training_time": 0
        }
        
        self.Q_table = np.zeros([self.env.observation_space.n, self.env.action_space.n])
        start_time = time.perf_counter()
        for i in range(epochs):
            self.__q_learning_algo(isTraining=True)
        end_time = time.perf_counter()
        self.metrics["training_time"] = end_time - start_time

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
                self.Q_table = data["Q_table"]
                self.metrics = data["metrics"]
        except:
            print("An error occured while trying to open the file") 
        

    def save_model(self, filename="model"):
        try:
            data = {
                "Q_table": self.Q_table,
                "metrics": self.metrics,
                "learning_rate": self.learning_rate,
                "gamma": self.gamma,
                "exploration_prob": self.exploration_prob
            }
            with open(filename + ".pickle", "wb") as f:
                pickle.dump(data, f)
        except:
            print("An error occured while trying to save the file") 

    def __q_learning_algo(self, isTraining=False):
            state = self.env.reset()
            episode_over = False

            run_reward = 0
            step_count = 0
            state = state[0]
            while not episode_over:
                rand = np.random.rand()
                if rand < self.exploration_prob and isTraining:
                    action = np.argmax(self.Q_table[state] + rand) 
                else:
                    action = np.argmax(self.Q_table[state]) 
                
                s_, reward, terminated, truncated, info = self.env.step(action)
                
                if isTraining: 
                    self.Q_table[state,action] = (1.0 - self.learning_rate)*self.Q_table[state,action] + self.learning_rate*(reward + self.gamma * np.max(self.Q_table[s_,:]))
                    run_reward += reward
                    step_count += 1

                episode_over = terminated or truncated
                state = s_
                
            
            if isTraining: 
                self.metrics["rewards"].append(run_reward / step_count)
                self.metrics["steps"].append(step_count)
                self.metrics["success_rate"].append(int(run_reward > 0))

            self.env.close()
        
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
        print("Success rate (%):", np.mean(self.metrics["success_rate"])*100)
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