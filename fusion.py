import torch

from compression import apply_operator_fusion
from models import SimpleNet


def main() -> None:
    model = SimpleNet()
    model.load_state_dict(torch.load("model.pth"))
    apply_operator_fusion(model)
    print("Operator-fused model saved to model_fused.pt")


if __name__ == "__main__":
    main()
