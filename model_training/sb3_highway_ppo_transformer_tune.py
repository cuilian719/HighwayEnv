import os
import importlib.util
import sys
import types
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import optuna

import highway_env  # noqa: F401

# Optional seaborn for nicer coloring in visualizations; fall back to a small
# stub if not available so imports don't fail when seaborn isn't installed.
try:
    import seaborn as sns  # noqa: F401
except Exception:
    sns = types.ModuleType("seaborn")
    def _color_palette(*args, **kwargs):
        return [(0.5, 0.5, 0.5)]
    sns.color_palette = _color_palette
    sys.modules["seaborn"] = sns

# Load helper objects from the transformer script so we reuse the custom extractor
spec = importlib.util.spec_from_file_location(
    "sb3_ppo", os.path.join("scripts/sb3_highway_ppo_transformer.py")
)
ppo_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ppo_module)

# 0. Set total timesteps
total_timestep = int(5e4)  # 50k timesteps for evaluation
n_trials = 20          # Number of Optuna trials for hyperparameter tuning


def optimize_agent(trial):
    """Hyperparameters to optimise for PPO transformer."""

    # Sample hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    n_steps_total = trial.suggest_categorical("n_steps_total", [128, 256, 512])
    n_cpu = 2
    n_steps = max(1, int(n_steps_total) // n_cpu)

    # Tune a small part of the attention architecture
    heads = trial.suggest_categorical("attention_heads", [1, 2, 4])
    feature_size = trial.suggest_categorical("feature_size", [32, 64])

    # Build policy kwargs reusing extractor kwargs from the source script
    attention_kwargs = dict(ppo_module.attention_network_kwargs)
    attention_kwargs["attention_layer_kwargs"] = dict(
        attention_kwargs.get("attention_layer_kwargs", {}),
        feature_size=feature_size,
        heads=heads,
    )

    policy_kwargs = dict(
        features_extractor_class=ppo_module.CustomExtractor,
        features_extractor_kwargs=attention_kwargs,
    )

    # Create the vectorized environment for training
    env = make_vec_env(
        ppo_module.make_configure_env,
        n_envs=n_cpu,
        seed=0,
        vec_env_cls=SubprocVecEnv,
        env_kwargs=ppo_module.env_kwargs,
    )

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=n_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        gamma=gamma,
        policy_kwargs=policy_kwargs,
        verbose=0,
    )

    try:
        model.learn(total_timesteps=total_timestep)
    except Exception as e:
        print("Training failed for trial:", e)
        env.close()
        return -1e6

    # Evaluate the model on a single environment
    eval_env = ppo_module.make_configure_env(**ppo_module.env_kwargs)
    mean_reward = 0
    episodes = 5
    for _ in range(episodes):
        obs, info = eval_env.reset()
        done = truncated = False
        ep_reward = 0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = eval_env.step(action)
            ep_reward += reward
        mean_reward += ep_reward
    eval_env.close()
    env.close()
    return mean_reward / episodes


if __name__ == "__main__":
    print("Running Optuna Hyperparameter Optimization for PPO transformer...")
    study = optuna.create_study(direction="maximize")
    study.optimize(optimize_agent, n_trials=n_trials)

    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Retrain final model with best hyperparameters for 10x timesteps
    print("\n--- Training final model with best hyperparameters ---")
    best = trial.params
    log_dir = "model_training/highway_ppo_transformer_tune/"
    os.makedirs(log_dir, exist_ok=True)

    # Reconstruct attention kwargs and policy kwargs
    attention_kwargs = dict(ppo_module.attention_network_kwargs)
    attention_kwargs["attention_layer_kwargs"] = dict(
        attention_kwargs.get("attention_layer_kwargs", {}),
        feature_size=int(best.get("feature_size", attention_kwargs["attention_layer_kwargs"].get("feature_size", 64))),
        heads=int(best.get("attention_heads", attention_kwargs["attention_layer_kwargs"].get("heads", 2))),
    )
    policy_kwargs = dict(
        features_extractor_class=ppo_module.CustomExtractor,
        features_extractor_kwargs=attention_kwargs,
    )

    # Use a single Monitor-wrapped environment for logging
    env = ppo_module.make_configure_env(**ppo_module.env_kwargs)
    env = Monitor(env, log_dir)

    best_n_steps_total = int(best.get("n_steps_total", 256))
    n_cpu = 1
    n_steps = max(1, best_n_steps_total // max(1, n_cpu))

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=n_steps,
        batch_size=int(best.get("batch_size", 64)),
        learning_rate=float(best.get("learning_rate", 2e-3)),
        gamma=float(best.get("gamma", 0.99)),
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=log_dir,
    )

    model.learn(total_timesteps=total_timestep * 10)
    model.save(os.path.join(log_dir, "best_model"))
    del model

    print("\n--- Plotting training metrics ---")
    try:
        df = pd.read_csv(os.path.join(log_dir, "monitor.csv"), skiprows=1)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        ax1.plot(df['l'].rolling(window=10).mean(), color='blue')
        ax1.set_title('Episode Length (Moving Average 10)')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Length')
        ax1.grid(True)

        ax2.plot(df['r'].rolling(window=10).mean(), color='green')
        ax2.set_title('Episode Reward (Moving Average 10)')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Reward')
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, 'training_metrics.png'))
        print(f"Metrics plot saved to {log_dir}training_metrics.png")
    except Exception as e:
        print(f"Failed to plot metrics: {e}")

    print("\n--- Evaluating final best model ---")
    best_model = PPO.load(os.path.join(log_dir, "best_model"))

    # create a fresh evaluation env with render_mode for video
    # make_configure_env doesn't accept render_mode, so build manually
    eval_env = gym.make(
        ppo_module.env_kwargs["id"],
        config=ppo_module.env_kwargs["config"],
        render_mode="rgb_array",
    )
    env = RecordVideo(
        eval_env, video_folder=f"{log_dir}videos", episode_trigger=lambda e: True
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
    print(f"Final evaluation videos saved to {log_dir}videos/")
