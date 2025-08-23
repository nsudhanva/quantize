import copy
import os

import torch
from torch import nn

from compression import apply_low_rank, apply_pruning
from models import SimpleNet, StudentNet
from training import distill, train


def size_mb(path: str) -> float:
    return os.path.getsize(path) / 1e6


def main() -> None:
    torch.manual_seed(0)

    data = torch.randn(64, 784)
    targets = torch.randint(0, 10, (64,))

    teacher = SimpleNet()
    teacher.train()
    train(teacher, data, targets)
    torch.save(teacher.state_dict(), "model.pth")
    print("Trained model saved to model.pth")

    teacher.eval()
    fp16 = copy.deepcopy(teacher).half()
    torch.save(fp16.state_dict(), "model_fp16.pth")
    print("FP16 model saved to model_fp16.pth")

    qmodel = torch.ao.quantization.quantize_dynamic(
        teacher, {nn.Linear}, dtype=torch.qint8
    )
    torch.save(qmodel.state_dict(), "model_dynamic.pth")
    print("Quantized model saved to model_dynamic.pth")

    low_rank = apply_low_rank(teacher)
    torch.save(low_rank.state_dict(), "model_low_rank.pth")
    print("Low-rank model saved to model_low_rank.pth")

    pruned = apply_pruning(teacher)
    torch.save(pruned.state_dict(), "model_pruned.pth")
    print("Pruned model saved to model_pruned.pth")

    student = StudentNet()
    distill(student, teacher, data)
    torch.save(student.state_dict(), "model_distilled.pth")
    print("Distilled student model saved to model_distilled.pth")

    qstudent = torch.ao.quantization.quantize_dynamic(
        student, {nn.Linear}, dtype=torch.qint8
    )
    torch.save(qstudent.state_dict(), "model_distilled_quantized.pth")
    print("Quantized student model saved to model_distilled_quantized.pth")

    print(f"FP32 teacher size: {size_mb('model.pth'):.2f} MB")
    print(f"FP16 teacher size: {size_mb('model_fp16.pth'):.2f} MB")
    print(f"Dynamic INT8 teacher size: {size_mb('model_dynamic.pth'):.2f} MB")
    print(f"Low-rank teacher size: {size_mb('model_low_rank.pth'):.2f} MB")
    print(f"Pruned teacher size: {size_mb('model_pruned.pth'):.2f} MB")
    print(f"Student size: {size_mb('model_distilled.pth'):.2f} MB")
    print(f"Quantized student size: {size_mb('model_distilled_quantized.pth'):.2f} MB")


if __name__ == "__main__":
    main()
