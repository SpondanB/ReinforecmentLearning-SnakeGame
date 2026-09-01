# 🐍 Reinforcement Learning — Snake Game

> **Teaching an AI to play Snake using Deep Q-Learning and PyTorch.**

What happens when you give an agent a Snake game, a handful of numbers describing its surroundings, and no instructions on *how* to play?

It learns.

This project explores **Reinforcement Learning (RL)** by training an agent to play Snake using **Deep Q-Learning (DQN)**. The agent observes the current game state, chooses an action, receives a reward from the environment, and gradually learns which actions lead to better outcomes.

The entire learning process is implemented using **PyTorch**, while the Snake environment is built using **pygame-ce**.

---

## 🧠 The Idea

The basic RL loop is surprisingly simple:

```text
                ┌─────────────────┐
                │   Snake Game    │
                │   Environment   │
                └────────┬────────┘
                         │
                      State
                         │
                         ▼
                ┌─────────────────┐
                │      Agent      │
                │  Neural Network │
                └────────┬────────┘
                         │
                       Action
                         │
                         ▼
                ┌─────────────────┐
                │   Snake Game    │
                │  takes action   │
                └────────┬────────┘
                         │
                    Reward + New State
                         │
                         ▼
                ┌─────────────────┐
                │ Train the Model │
                └────────┬────────┘
                         │
                         └──────► Repeat
```

The agent doesn't receive a strategy such as:

> "If food is on your left, turn left."

Instead, it has to discover useful behavior through **trial and error**.

---

# 🎮 Environment

The environment is a Snake game built using:

* **Python**
* **pygame-ce**

The game acts as the agent's world.

At every step, the agent:

1. Observes the current state.
2. Selects an action.
3. The Snake moves.
4. The environment calculates a reward.
5. The agent observes the new state.
6. The neural network is updated.

This creates the fundamental RL interaction:

```text
State → Action → Reward → New State → Learning
```

---

# 🎯 Action Space

The agent has only **three possible actions**.

```text
[1, 0, 0] → Straight
[0, 1, 0] → Right
[0, 0, 1] → Left
```

Rather than asking the model to predict an absolute direction such as `UP`, `DOWN`, `LEFT`, or `RIGHT`, the agent makes decisions relative to its **current direction of movement**.

For example:

```text
Current Direction
       ↓

    ┌───────┐
    │       │
    │ Snake │ ─────► Straight
    │       │
    └───────┘
       ↘
        Right
```

This keeps the action space small and makes the problem easier for the model to learn.

---

# 👁️ State Representation

The agent does not see the entire game screen.

Instead, the environment converts the current situation into an **11-dimensional state vector**.

```text
[
    danger_straight,
    danger_right,
    danger_left,

    direction_left,
    direction_right,
    direction_up,
    direction_down,

    food_left,
    food_right,
    food_up,
    food_down
]
```

### 🚧 Danger

The first three values tell the agent whether moving in a particular direction could result in a collision:

```text
danger_straight
danger_right
danger_left
```

### 🧭 Direction

The next four values describe the Snake's current direction:

```text
direction_left
direction_right
direction_up
direction_down
```

Only the relevant direction is active at a time.

For example:

```text
[0, 0, 1, 0]
```

could represent:

```text
        ↑
        │
      Snake
        │
        │
```

depending on the chosen encoding.

### 🍎 Food Location

The final four values describe where the food is relative to the Snake:

```text
food_left
food_right
food_up
food_down
```

So instead of giving the neural network the entire game image, we give it a compact description of the important information it needs to make a decision.

---

# 🤖 Deep Q-Learning

The model used in this project is a **Deep Q-Network (DQN)** implemented with PyTorch.

The idea behind Q-Learning is to estimate:

> **"How good is this action in this state?"**

This value is called the **Q-value**.

The neural network takes the current state as input and produces one Q-value for each possible action.

```text
                 State
             11 values
                  │
                  ▼
          ┌───────────────┐
          │ Neural Network│
          │   11 → 256    │
          │   256 → 3     │
          └───────┬───────┘
                  │
          ┌───────┼───────┐
          ▼       ▼       ▼
       Straight  Right   Left
        Q(s,a₁) Q(s,a₂) Q(s,a₃)
```

### Model Architecture

The current network architecture is:

```text
Input:   11
Hidden:  256
Output:  3
```

In other words:

```text
11 → 256 → 3
```

The three output values correspond to:

```text
Q(Straight)
Q(Right)
Q(Left)
```

The agent can then select the action with the highest predicted Q-value.

---

# 📈 Learning With Rewards

The agent learns through rewards.

For this project:

| Event            | Reward |
| ---------------- | -----: |
| 🍎 Eat food      |  `+10` |
| 💀 Die / collide |  `-10` |

The reward provides the feedback necessary for learning.

For example:

```text
Snake moves
     │
     ├── Eats food ───────► +10
     │
     ├── Normal movement ─►  0
     │
     └── Dies ─────────────► -10
```

The objective is therefore not explicitly:

> "Get to the food."

Instead, the agent learns that actions leading to food tend to produce **higher future returns**, while actions leading to death produce **negative returns**.

---

# 🧮 The Bellman Equation

The core of Q-Learning is the **Bellman equation**.

The Q-value can be updated using:

$$
Q_{new}(s,a)=Q(s,a)+\alpha\left[R(s,a)+\gamma \max_{a'}Q(s',a')-Q(s,a)\right]
$$

Where:

* $Q(s,a)$ — current estimate of the quality of an action
* $\alpha$ — learning rate
* $R(s,a)$ — reward received after taking the action
* $\gamma$ — discount factor
* $s'$ — new state
* $\max Q(s',a')$ — best expected future value from the new state

In this project:

```text
Learning Rate (α) = 0.001
Discount Factor (γ) = 0.9
```

---

# 🔄 From Q-Learning to a Neural Network

With a neural network, we don't directly maintain a Q-table.

Instead, the network approximates:

```text
Q(state, action)
```

The process looks roughly like this:

```python
state_0
   │
   ▼
model(state_0)
   │
   ▼
Q-values for all actions
   │
   ▼
choose action
   │
   ▼
game.play_step(action)
   │
   ├── reward
   ├── game_over
   └── state_1
```

We then calculate the target Q-value:

$$
Q_{target}=R+\gamma \max Q(s',a')
$$

and train the network so that its prediction moves toward this target.

Conceptually:

```text
Predicted Q-value
       │
       │
       ▼
   ┌─────────┐
   │ Neural  │
   │ Network │
   └────┬────┘
        │
        ▼
    Q(state)

         ↕
       Loss

         ↕

Reward + discounted
future Q-value
```

The model is trained by minimizing the difference between the predicted value and the target value.

A simple formulation is:

$$
Loss = (Q_{target} - Q_{predicted})^2
$$

which corresponds to **Mean Squared Error (MSE)**.

---

# 🎲 Exploration vs Exploitation

One of the problems with letting the agent *always* choose the action with the highest predicted Q-value is that it can get stuck exploiting a bad strategy.

So the agent sometimes explores.

The exploration parameter used in this project is:

```text
ε = 0.2
```

Conceptually:

```text
             Choose Action
                  │
          ┌───────┴───────┐
          │               │
       Explore         Exploit
          │               │
     Random Action    Best Q-value
```

With exploration, the agent has opportunities to discover actions it might otherwise never try.

---

# 🔁 Training Loop

The entire learning process can be summarized as:

```python
state = get_state(game)

action = get_move(state)

reward, game_over, score = game.play_step(action)

new_state = get_state(game)

remember(state, action, reward, new_state, game_over)

train_model()

state = new_state
```

Repeated over many games, this becomes:

```text
                ┌───────────────┐
                │ Get Game State│
                └───────┬───────┘
                        │
                        ▼
                ┌───────────────┐
                │ Choose Action │
                └───────┬───────┘
                        │
                        ▼
                ┌───────────────┐
                │  Play Action  │
                └───────┬───────┘
                        │
                        ▼
                ┌───────────────┐
                │ Get Reward    │
                │ + New State   │
                └───────┬───────┘
                        │
                        ▼
                ┌───────────────┐
                │ Train Network │
                └───────┬───────┘
                        │
                        └──────────► Repeat
```

Every game becomes another opportunity for the agent to improve its estimate of which actions are valuable.

---

# 📊 Results

After approximately **15 minutes of training** and around **100 generations**, the agent achieved a highest observed score of:

## 🏆 89

```text
Training Time    ≈ 15 minutes
Generations      ≈ 100
Highest Score    89
```

This was particularly interesting because the agent wasn't explicitly programmed with a Snake-playing strategy.

Its behavior emerged from repeatedly interacting with the environment and updating its neural network based on rewards.

---

# 🛠️ Tech Stack

| Component        | Technology      |
| ---------------- | --------------- |
| Language         | Python          |
| RL Algorithm     | Deep Q-Learning |
| Neural Network   | PyTorch         |
| Game Environment | pygame-ce       |
| State Space      | 11-dimensional  |
| Action Space     | 3 actions       |
| Learning Rate    | `0.001`         |
| Discount Factor  | `0.9`           |
| Exploration      | `ε = 0.2`       |
| Best Score       | `89`            |

---

# 🚀 Running the Project

Clone the repository and install the required dependencies.

Then run:

```bash
python agent.py
```

The agent will begin interacting with the Snake environment and training the model.

---

# 📁 Project Concept

At a high level, the project can be thought of as three components:

```text
RL Snake
│
├── 🎮 Game
│   └── Environment
│
├── 🤖 Agent
│   └── State → Action
│
└── 🧠 Model
    └── Q-value approximation
```

The **Game** provides the environment.

The **Agent** decides what to do.

The **Model** learns which decisions are likely to produce better future rewards.

Together:

```text
             ┌───────────────┐
             │     Agent     │
             └───────┬───────┘
                     │
                  Action
                     │
                     ▼
             ┌───────────────┐
             │     Game      │
             └───────┬───────┘
                     │
             Reward + State
                     │
                     ▼
             ┌───────────────┐
             │     Model     │
             │    PyTorch    │
             └───────┬───────┘
                     │
                     └──────► Improve
```

---

# 💡 What I Learned

Building this project helped me understand reinforcement learning beyond the equations.

Some of the key ideas explored were:

* How an RL agent interacts with an environment
* Designing a useful state representation
* Defining an action space
* Designing reward functions
* Q-values and value estimation
* The Bellman equation
* Exploration vs exploitation
* Neural networks as Q-function approximators
* Training a model through repeated interaction
* How seemingly simple games can become interesting RL environments

The most interesting part is the transition from:

```text
"I have no idea what to do."
```

to:

```text
"I have seen this situation before,
and I have learned which action tends
to work."
```

That's the fundamental idea behind reinforcement learning.

---

# 🔮 Future Improvements

There are several directions I want to explore from here:

### 🧠 Better DQN

Experiment with techniques such as:

* Experience Replay
* Target Networks
* Double DQN
* Dueling DQN

### 👁️ More Complex State Representations

Instead of manually describing the environment, experiment with giving the agent more information about the board.

For example:

```text
Current State
     ↓
Game Grid
     ↓
Neural Network
     ↓
Action
```

This could eventually lead toward using **CNNs to process the game directly**.

### 📈 Training Analysis

Add:

* Score vs. generation graphs
* Average reward
* Loss curves
* Exploration rate
* Training checkpoints
* Model evaluation

### 🎥 Visualization

Record the agent during training to visualize how its behavior evolves from random movement into increasingly deliberate gameplay.

---

# 🧪 Why Snake?

Snake is deceptively simple.

The rules are easy:

```text
Move → Eat → Grow → Don't Die
```

But the agent still has to deal with:

* Delayed consequences
* Collision avoidance
* Planning
* Exploration
* Reward optimization
* Long-term decision making

An action that looks good **right now** might create a situation where the Snake dies several steps later.

That makes Snake a small but useful environment for experimenting with **sequential decision-making**.

---

# ⭐ Final Thought

This project started with a simple question:

> **Can I teach a machine to play Snake without explicitly telling it how to play?**

The answer is yes — but the interesting part isn't simply getting the Snake to achieve a high score.

It's watching a neural network gradually learn a relationship between:

```text
State
  ↓
Action
  ↓
Consequence
  ↓
Reward
  ↓
Learning
  ↓
Better Action
```

And that loop is the foundation of **Reinforcement Learning**.