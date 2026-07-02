import argparse
import gymnasium as gym
import torch
import homebot  # noqa: F401

from evaluate import load_q_model
from chained_eval import run_chain, _print_readout
from task_chain import DEFAULT_CHAIN

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/q_model.pt")
    parser.add_argument("--goal-layers", type=int, default=2)
    parser.add_argument("--head-layers", type=int, default=4)
    parser.add_argument("--use-motion", action="store_true", default=True,
                        help="whether the model uses motion features")
    parser.add_argument("--motion-window", type=int, default=8)
    parser.add_argument("--use-projection", action="store_true", default=True)
    parser.add_argument("--motion-mlp", action="store_true", default=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--readout", default="softmax_rel", choices=["greedy", "softmax", "softmax_rel"])
    parser.add_argument("--temp", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cardinal-only", action="store_true", default=False,
                        help="restrict the action space to 4 cardinal directions")
    parser.add_argument("--frame-skip", type=int, default=1,
                        help="number of frames to skip (action repeat)")
    parser.add_argument("--render-mode", default="human")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Create the non-goal environment with the specified rendering mode
    env = gym.make(
        "HomeBot2D-V1",
        render_mode=args.render_mode,
        action_mode="discrete",
        obs_resolution=(96, 96),
        n_trash=2,
        max_steps=10000,  # High step limit, let the per-leg budget handle timeouts
        map_name="default",
        random_start=True,
    )

    if args.cardinal_only:
        from cardinal_wrapper import CardinalActionWrapper
        env = CardinalActionWrapper(env)

    if args.frame_skip > 1:
        from env_wrappers import FrameSkipWrapper
        env = FrameSkipWrapper(env, skip=args.frame_skip)

    n_actions = env.action_space.n
    model = load_q_model(
        args.checkpoint,
        n_actions,
        device,
        goal_layers=args.goal_layers,
        head_layers=args.head_layers,
        use_motion=args.use_motion,
        motion_window=args.motion_window,
        use_projection=args.use_projection,
        motion_mlp=args.motion_mlp,
    )

    print(f"Loaded model from {args.checkpoint}", flush=True)
    print(f"Running multi-step sequence: {DEFAULT_CHAIN}", flush=True)

    episodes_results = []
    for i in range(args.episodes):
        print(f"\n--- Starting Episode {i+1}/{args.episodes} ---", flush=True)
        results = run_chain(
            model,
            env,
            DEFAULT_CHAIN,
            device,
            args.readout,
            args.temp,
            seed=args.seed + i,
            verbose=True,
        )
        episodes_results.append(results)

    _print_readout(args.readout, episodes_results, DEFAULT_CHAIN)

if __name__ == "__main__":
    main()
