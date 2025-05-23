from pathlib import Path
import argparse
import sys

import optuna
from stable_baselines3.common.evaluation import evaluate_policy

from algorithms import get_agent_class
from config import MODELS_DIR

import minihack


#  Objective function                                                   #
def _objective(trial, AgentCls, env_id, total_steps, seed, n_envs):
    # 1) sample per-algorithm kwargs
    sampled_kwargs = AgentCls.sample_hyperparams(trial)

    # 2) build & train
    agent = AgentCls(
        env_id,
        total_steps=total_steps,
        seed=seed,
        n_envs=n_envs,
        **sampled_kwargs,
    )
    agent.train()

    # 3) quick evaluation (5 deterministic episodes)
    mean_reward, _ = evaluate_policy(
        agent.model,
        agent.env,
        n_eval_episodes=5,
        deterministic=True,
    )
    return mean_reward


#  Public helper
def run_search(
    algo: str,
    env_id: str,
    n_trials: int = 20,
    total_steps: int = 100_000,
    seed: int = 0,
    n_envs: int = 24,
):
    AgentCls = get_agent_class(algo)

    study = optuna.create_study(
        study_name=f"{algo}_{env_id}",
        direction="maximize",
    )

    study.optimize(
        lambda tr: _objective(
            tr, AgentCls, env_id, total_steps, seed, n_envs
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    print("🏆  Best value :", study.best_value)
    print("⚙︎  Best params:", study.best_params)

    # retrain on the *same* steps but with best hyper-params
    best_kwargs = AgentCls.sample_hyperparams(
        optuna.trial.FixedTrial(study.best_params)
    )
    final_agent = AgentCls(
        env_id,
        total_steps=total_steps*5,
        seed=seed,
        n_envs=n_envs,
        **best_kwargs,
    )
    final_agent.train()
    path = final_agent.save("best_hp_model.zip")
    print("💾  Final model saved to", path)

    return path, study


#  CLI wrapper
def _parse(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--algo", default="ppo", help="Algorithm key (see registry)")
    p.add_argument("--env", required=True, help="Gym/MiniHack environment id")
    p.add_argument("--trials", type=int, default=20, help="# Optuna trials")
    p.add_argument("--steps", type=int, default=100_000,
                   help="Training steps *per* trial and for final model")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=24)
    return p.parse_args(argv)


def main(argv=None):
    args = _parse(argv or sys.argv[1:])
    run_search(
        algo=args.algo,
        env_id=args.env,
        n_trials=args.trials,
        total_steps=args.steps,
        seed=args.seed,
        n_envs=args.n_envs,
    )


if __name__ == "__main__":
    main()
