# MiniHack-RL
How to run training example:

Possible algorithm choices:
- PPO
```
python -m scripts.hyperopt --algo PPO --env MiniHack-Room-Ultimate-5x5-v0 --trials 10 --steps 50000 --seed 42 --n-envs 8
```
- RPPO (Recurrent PPO)
```
python -m scripts.hyperopt --algo RPPO --env MiniHack-Room-Ultimate-5x5-v0 --trials 10 --steps 50000 --seed 42 --n-envs 8
```
- DQRQN (Quantile-Regression DQN)
```
python -m scripts.hyperopt --algo DQRQN --env MiniHack-Room-Ultimate-5x5-v0 --trials 10 --steps 50000 --seed 42 --n-envs 8
```