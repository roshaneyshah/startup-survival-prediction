import os
import sys
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

def run_train(data_path=None):
    from src.train import train
    train(data_path=data_path)

def run_server():
    from backend.app import app
    print("Starting server on http://localhost:5000")
    app.run(debug=False, port=5000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Startup Survival Predictor")
    parser.add_argument("mode", choices=["train", "serve"], help="train: train model | serve: start API")
    parser.add_argument("--data", type=str, default=None, help="Path to dataset folder (default: ./dataset)")
    args = parser.parse_args()

    if args.mode == "train":
        run_train(data_path=args.data)
    elif args.mode == "serve":
        run_server()
