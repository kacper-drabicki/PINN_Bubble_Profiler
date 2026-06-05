import argparse
import torch
from src.model import Bouncer
from train import pretrain, finetune

from src.physics import polynomial, singlet

EXPERIMENTS = {
    "singlet": singlet.Config,
    "polynomial": polynomial.Config,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=EXPERIMENTS.keys())
    args = parser.parse_args()
    
    config = EXPERIMENTS[args.exp]()
    
    model = Bouncer(output_dim=config.output_dim).to(config.device)

    print("=== PRETRAIN ===")
    pretrain(model, config)

    print("=== FINETUNE ===")
    finetune(model, config)

    torch.save(model.state_dict(), config.modelPath)

if __name__ == "__main__":
    main()
