# MiniHack-RL
How to run training example:

Possible algorithm choices:
- PPO
```
python -m scripts.hyperopt --algo PPO --env MiniHack-HideNSeek-v0 --trials 12 --steps 250000 --seed 42 --n-envs 6
```
- RPPO (Recurrent PPO)
```
python -m scripts.hyperopt --algo RPPO --env MiniHack-HideNSeek-v0 --trials 16 --steps 18000 --seed 42 --n-envs 4
```
- DRDQN (Quantile-Regression DQN)
```
python -m scripts.hyperopt --algo QRDQN --env MiniHack-HideNSeek-v0 --trials 16 --steps 18000 --seed 42 --n-envs 4
```