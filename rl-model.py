import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras.models import Sequential
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Mock environment parameters
grid_size = 50
start_position = (0, 0)
goal_position = (25, 39)
obstacle_position = (grid_size/2, grid_size/2)

# Example input shape (e.g., character position, objects positions)
input_shape = (10,)  # Adjust based on your actual input
action_space = 4  # 4 actions: forward, backward, left, right

# Learning parameters
learning_rate = 0.1
discount_factor = 0.99
epsilon = 0.1  # Exploration rate

# Initialize Q-table
q_table = np.zeros([grid_size * grid_size, action_space])

def state_to_index(state):
    x, y = state[0], state[1]
    return x * grid_size + y

# Neural network model
def create_model(input_shape, action_space):
    print("Defining model")
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(action_space, activation='linear')  # Output layer
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model

# Fetch initial game state
def get_initial_state():
   # State format: [character_x, character_y, goal_x, goal_y]
    return np.array([*start_position, *goal_position])


# Perform action in the mock environment
def perform_action(state, action):
    x, y = state[0], state[1]

    # Define actions
    if action == 0 and y > 0: 
        y -= 1  # Up 
        #print("move up")
    elif action == 1 and y < grid_size - 1: 
        y += 1  # Down
        #print("move down")
    elif action == 2 and x > 0: 
        x -= 1  # Left
        #print("move left")
    elif action == 3 and x < grid_size - 1: 
        x += 1  # Right
        #print("move right")

    # Check for obstacle or goal
    new_state = np.array([x, y, *goal_position])
    if (x, y) == obstacle_position or x >= grid_size or y >= grid_size:
        reward, done = -5, True  # Hit obstacle or went out
        #print("Hit obstacle or went out")
        new_state = np.array([*start_position, *goal_position])  # Reset to start
    elif (x, y) == goal_position:
        reward, done = 10, True  # Reached goal
        #print("Reached goal****")

    else:
        reward, done = -1, False  # Penalize to encourage faster goal-reaching

    return new_state, reward, done

# Mock predict action
def predict_action(model, state):
    state_index = state_to_index(state)
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, 3)  # Explore action space
    else:
        return np.argmax(q_table[state_index])  # Exploit learned values

# Mock update model function
def update_model(state, action, reward, next_state):
    state_index = state_to_index(state)
    next_state_index = state_to_index(next_state)

    old_value = q_table[state_index, action]
    next_max = np.max(q_table[next_state_index])
    new_value = (1 - learning_rate) * old_value + learning_rate * (reward + discount_factor * next_max)
    q_table[state_index, action] = new_value

# Modified training function to use mock data
def model_training(model, total_episodes):
    paths = []  # List to store paths for each episode
    print("start training...")
    goal_reached_count = 0
    lost_count = 0

    for episode in range(total_episodes):
        state = get_initial_state()
        path = [start_position]
        done = False


        while not done:
            action = predict_action(model, state)  # Predict action
            next_state, reward, done = perform_action(state, action)  # Perform action
            path.append((next_state[0], next_state[1]))
            if reward == 10:
                goal_reached_count += 1
            elif reward == -5:
                lost_count += 1

            update_model(state, action, reward, next_state)  # Update model

            state = next_state  # Update state

        paths.append(path)  # Store path for this episode
    win_rate = (goal_reached_count / total_episodes) * 100

    print(f"Goal reached {goal_reached_count} times.")
    print(f"Lost {lost_count} times.")
    print(f"Win Rate {win_rate}%.")
    print(q_table)
    return paths

def plot_paths(paths):
    for path in paths:
        # Unzip the path into X and Y coordinates
        x, y = zip(*path)
        plt.plot(x, y, marker='o')  # Plot each path

    plt.xlim(0, grid_size - 1)
    plt.ylim(0, grid_size - 1)
    plt.grid(True)
    plt.show()

def create_heatmap(paths, grid_size):
    # Initialize the grid to zeros
    frequency_grid = np.zeros((grid_size, grid_size))

    # Populate the frequency grid
    for path in paths:
        for x, y in path:
            if 0 <= x < grid_size and 0 <= y < grid_size:
                frequency_grid[y, x] += 1  # Increment the cell corresponding to the position

    # Plot the heatmap
    plt.imshow(frequency_grid, cmap='hot', interpolation='nearest')
    plt.colorbar()  # Add a colorbar to a plot
    plt.show()

def create_optimized_heatmap(paths, grid_size):
    # Initialize the grid to zeros
    frequency_grid = np.zeros((grid_size, grid_size), dtype=int)

    # Flatten the list of paths and count the occurrences of each position
    all_positions = np.concatenate(paths)
    unique, counts = np.unique(all_positions, axis=0, return_counts=True)
    
    # Update the frequency grid with counts
    frequency_grid[unique[:, 1], unique[:, 0]] = counts

    # Plot the heatmap
    plt.imshow(frequency_grid, cmap='hot', interpolation='nearest')
    plt.colorbar()  # Add a colorbar to a plot
    plt.show()


def plot_paths_with_legend(paths):
    # Get a color map
    colors = cm.rainbow(np.linspace(0, 1, len(paths)))

    for idx, path in enumerate(paths):
        x, y = zip(*path)
        plt.plot(x, y, color=colors[idx], marker='o', label=f'Iteration {idx+1}')

    plt.xlim(0, grid_size - 1)
    plt.ylim(0, grid_size - 1)
    plt.grid(True)

    # Create a legend
    plt.legend()
    plt.show()


model = create_model(input_shape, action_space)
model.summary()

paths = model_training(model, 10000)
print("Paths in total: ", len(paths))
# Plotting the paths
#plot_paths_with_legend(paths)

# Assuming paths is a list of lists containing (x, y) positions for each iteration
#create_optimized_heatmap(paths, grid_size)
create_heatmap(paths, grid_size)