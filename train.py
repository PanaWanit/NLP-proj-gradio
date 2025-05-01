import argparse
import time
from tqdm.auto import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Mock training script to print arguments")
    parser.add_argument("-s", "--source_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("-m", "--model_path", type=str, required=True, help="Path to the output model directory")
    parser.add_argument("--iterations", type=int, default=30000, help="Number of training iterations")
    parser.add_argument("--save_iterations", type=int, default=30000, help="Iterations at which to save model")
    parser.add_argument("--checkpoint_iterations", type=int, default=30000, help="Iterations at which to save checkpoints")
    parser.add_argument("--densify_until_iter", type=int, default=5000, help="Iteration until densification occurs")
    parser.add_argument("--percent_dense", type=float, default=0.01, help="Percentage of dense points")

    return parser.parse_args()

def main():
    args = parse_args()

    # Print all arguments
    print("=== Mock Training Script Started ===")
    print(f"Source Path (-s): {args.source_path}")
    print(f"Model Path (-m): {args.model_path}")
    print(f"Iterations (--iterations): {args.iterations}")
    print(f"Save Iterations (--save_iterations): {args.save_iterations}")
    print(f"Checkpoint Iterations (--checkpoint_iterations): {args.checkpoint_iterations}")
    print(f"Densify Until Iter (--densify_until_iter): {args.densify_until_iter}")
    print(f"Percent Dense (--percent_dense): {args.percent_dense}")
    print("===================================")

    # Simulate some "training" with periodic output
    print("Simulating training...")
    for i in tqdm(range(1, 6)):
        time.sleep(1)  # Simulate work
        print("Training step", i)
        if i == 5:
            print("Saving model...")
            time.sleep(1)
    
    print("Mock training completed!")

if __name__ == "__main__":
    main()