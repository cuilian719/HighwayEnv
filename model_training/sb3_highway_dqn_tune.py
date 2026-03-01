import os
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
import optuna
import highway_env  # noqa: F401

# 0. Set total timesteps
total_timestep = 5e4   # 50k timesteps for evaluation
n_trials = 20          # Number of Optuna trials for hyperparameter tuning

def optimize_agent(trial):
    """ Learning hyperparamters we want to optimise"""

    # 1. Sample hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    gamma = trial.suggest_categorical("gamma", [0.8, 0.9, 0.95, 0.99])
    
    # Sample network architecture
    net_arch_depth = trial.suggest_int("net_arch_depth", 1, 3)
    net_arch_width = trial.suggest_categorical("net_arch_width", [64, 128, 256])
    net_arch = [net_arch_width] * net_arch_depth
    
    # Create the environment inside the objective
    env = gym.make("highway-fast-v0")
    
    # Create the model
    model = DQN(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=net_arch),
        learning_rate=learning_rate,
        buffer_size=15000,
        learning_starts=200,
        batch_size=batch_size,
        gamma=gamma,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=50,
        verbose=0, # We don't want to swamp the output
    )

    # Train for a short number of timesteps to evaluate
    try:
        model.learn(total_timesteps=total_timestep)
    except Exception as e:
        # Sometimes models crash with bad hyperparams, we return a very bad reward
        print(e)
        return -100.0

    # Evaluate the model
    mean_reward = 0
    episodes = 5
    for _ in range(episodes):
        obs, info = env.reset()
        done = truncated = False
        episode_reward = 0
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
        mean_reward += episode_reward
    
    env.close()
    return mean_reward / episodes


if __name__ == "__main__":
    print("Running Optuna Hyperparameter Optimization...")
    study = optuna.create_study(direction="maximize")
    study.optimize(optimize_agent, n_trials=n_trials)
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    print("\n--- Training final model with best hyperparameters ---")
    best_params = trial.params
    env = gym.make("highway-fast-v0", render_mode="rgb_array")
    
    log_dir = "model_training/highway_dqn_tune/"
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir)
    
    best_net_arch = [best_params["net_arch_width"]] * best_params["net_arch_depth"]
    
    # Create the final model with best params
    best_model = DQN(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=best_net_arch),
        learning_rate=best_params["learning_rate"],
        buffer_size=15000,
        learning_starts=200,
        batch_size=best_params["batch_size"],
        gamma=best_params["gamma"],
        train_freq=1,
        gradient_steps=1,
        target_update_interval=50,
        verbose=1,
        tensorboard_log=f"{log_dir}",
    )
    
    # Train thoroughly (e.g. 10X steps of the original script)
    best_model.learn(total_timesteps=total_timestep * 10)
    best_model.save(f"{log_dir}best_model")
    del best_model
    
    print("\n--- Plotting training metrics ---")
    try:
        df = pd.read_csv(f"{log_dir}monitor.csv", skiprows=1)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot episode length
        ax1.plot(df['l'].rolling(window=10).mean(), color='blue')
        ax1.set_title('Episode Length (Moving Average 10)')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Length')
        ax1.grid(True)
        
        # Plot episode reward
        ax2.plot(df['r'].rolling(window=10).mean(), color='green')
        ax2.set_title('Episode Reward (Moving Average 10)')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Reward')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{log_dir}training_metrics.png")
        print(f"Metrics plot saved to {log_dir}training_metrics.png")
    except Exception as e:
        print(f"Failed to plot metrics: {e}")
    
    print("\n--- Evaluating final best model ---")
    best_model = DQN.load(f"{log_dir}best_model", env=env)
    
    env = RecordVideo(
        env, video_folder=f"{log_dir}videos", episode_trigger=lambda e: True
    )
    env.unwrapped.config["simulation_frequency"] = 15  # Higher FPS for rendering
    env.unwrapped.set_record_video_wrapper(env)

    for videos in range(5): # Record 5 episodes
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            # Predict
            action, _states = best_model.predict(obs, deterministic=True)
            # Get reward
            obs, reward, done, truncated, info = env.step(action)
            # Render
            env.render()
    env.close()
    print("Final evaluation videos saved to model_training/highway_dqn_tune/videos/")
