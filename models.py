from torch import nn


class SimpleNet(nn.Module):
    """Small fully connected network used as the teacher model."""

    def __init__(self) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):  # noqa: D401 - simple pass through sequential
        """Run data through the network."""
        return self.seq(x)


class StudentNet(nn.Module):
    """Smaller network distilled from the teacher model."""

    def __init__(self) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(784, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):  # noqa: D401 - simple pass through sequential
        """Run data through the network."""
        return self.seq(x)
