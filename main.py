import torch
from src.model import Bouncer
from train import pretrain, finetune
from src.physics.polynomial import Config

def main():
    config = Config()
    
    model = Bouncer().to(config.device)

    print("=== PRETRAIN ===")
    pretrain(model, config)

    print("=== FINETUNE ===")
    finetune(model, config)

    torch.save(model.state_dict(), config.modelPath)

if __name__ == "__main__":
    main()
