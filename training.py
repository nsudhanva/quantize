import torch
from torch import nn


def train(model, data, targets, epochs: int = 5) -> None:
    """Simple training loop."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()


def distill(student, teacher, data, epochs: int = 5) -> None:
    """Train ``student`` to match ``teacher`` outputs."""
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
