# Taxi Driver 🚕 – Reinforcement Learning Project

This project uses the **Taxi environment from OpenAI Gym** to train AI agents to **pick up, transport, and drop off passengers** in a simulated city. It serves as a great introduction to **Reinforcement Learning (RL)** and the fundamental challenges of **autonomous vehicles**.

---

## 🧠 Algorithms Implemented

### 1. Q-Learning (Tabular)
- Based on a **Q-Table** to store and update action values.
- Simple to implement and very fast to train in **small discrete environments**.
- **Limitations**: Performs poorly in large or dynamic state spaces due to exponential growth of the Q-table.

### 2. Deep Q-Learning (DQN)
- Uses a **neural network** to approximate Q-values instead of storing them in a table.
- **More adaptable** to large or continuous environments.
- **Downside**: Can take longer to converge and requires more tuning.
- **Advantage**: Scales better and can handle variable-sized environments.

---

## 🚀 Getting Started

### 1. (Optional) Create a virtual environment
```bash
pip install virtualenv # if virtualenv is not installed
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
```

### 2. Install the requirements
```bash
pip install -r requirements.txt
```

### 3. Run the project
Open the main.ipynb notebook at the root of the project to try out and compare the two algorithms.

--- 

## 🛠️ Technologies Used

- Python
- PyTorch
- NumPy
- Matplotlib
- OpenAI Gym