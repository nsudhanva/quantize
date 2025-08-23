import os
import copy

import torch
from torch import nn
import torch.nn.utils.prune as prune

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.seq(x)


class StudentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(784, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.seq(x)


def train(model, data, targets, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()


def apply_pruning(model, amount=0.5):
    pruned = copy.deepcopy(model)
    for layer in [pruned.seq[0], pruned.seq[2]]:
        prune.ln_structured(layer, name="weight", amount=amount, n=2, dim=0)
        prune.remove(layer, "weight")
    return pruned


def apply_low_rank(model, rank=32):
    low_rank = copy.deepcopy(model)
    new_layers = []
    for layer in low_rank.seq:
        if isinstance(layer, nn.Linear):
            W = layer.weight.data
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            r = min(rank, S.size(0))
            U_r = U[:, :r]
            S_r = S[:r]
            Vh_r = Vh[:r, :]
            B = S_r.unsqueeze(1) * Vh_r
            first = nn.Linear(Vh_r.shape[1], r, bias=False)
            first.weight.data = B
            second = nn.Linear(r, U_r.shape[0], bias=True)
            second.weight.data = U_r
            second.bias.data = layer.bias.data
            new_layers.extend([first, second])
        else:
            new_layers.append(layer)
    low_rank.seq = nn.Sequential(*new_layers)
    return low_rank


def distill(student, teacher, data, epochs=5):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    teacher.eval()
    for _ in range(epochs):
        optimizer.zero_grad()
        with torch.no_grad():
            teacher_outputs = teacher(data)
        student_outputs = student(data)
        loss = criterion(student_outputs, teacher_outputs)
        loss.backward()
        optimizer.step()


def main():
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

    def size_mb(path):
        return os.path.getsize(path) / 1e6

    print(f"FP32 teacher size: {size_mb('model.pth'):.2f} MB")
    print(f"FP16 teacher size: {size_mb('model_fp16.pth'):.2f} MB")
    print(f"Dynamic INT8 teacher size: {size_mb('model_dynamic.pth'):.2f} MB")
    print(f"Low-rank teacher size: {size_mb('model_low_rank.pth'):.2f} MB")
    print(f"Pruned teacher size: {size_mb('model_pruned.pth'):.2f} MB")
    print(f"Student size: {size_mb('model_distilled.pth'):.2f} MB")
    print(
        f"Quantized student size: {size_mb('model_distilled_quantized.pth'):.2f} MB"
    )


if __name__ == "__main__":
    main()
