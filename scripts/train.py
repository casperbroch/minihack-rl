import argparse, sys
from algorithms import get_agent_class
import minihack

def _parse(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--algo", default="ppo",
                   help="ppo | rppo | qrdqn | ...")
    p.add_argument("--env",  required=True,
                   help="Gym/MiniHack env id")
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--seed",  type=int, default=0)
    p.add_argument("--n-envs",type=int, default=24)
    return p.parse_args(argv)

def main(argv=None):
    args   = _parse(argv or sys.argv[1:])
    Agent  = get_agent_class(args.algo)
    agent  = Agent(args.env, args.steps, args.seed, n_envs=args.n_envs)
    path   = agent.train_and_save()
    print(f"✅ done – model saved to {path}")

if __name__ == "__main__":
    main()
