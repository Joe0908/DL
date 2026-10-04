"""Episode 1: Train a feedforward neural network to learn XOR.

Run from the repository root:
    python episodes/01_fnn/fnn.py

All four XOR examples are used for training. The final accuracy measures
fit to these examples, not performance on an independent test set.
"""

import torch
from torch import nn


class FeedForwardNN(nn.Module):
    """Two hidden layers: 2 inputs -> 8 neurons -> 8 neurons -> 1 output."""

    def __init__(self):
        super().__init__()

        # Each Linear layer creates trainable weights and biases.
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # A batch moves forward through the network, one layer at a time.
        h1 = self.relu(self.fc1(x))   # (batch_size, 2) -> (batch_size, 8)
        h2 = self.relu(self.fc2(h1))  # (batch_size, 8) -> (batch_size, 8)
        output = self.fc3(h2)        # (batch_size, 8) -> (batch_size, 1)
        return output


def main():
    # Fix the random seed so initialization is repeatable in this example.
    torch.manual_seed(42)

    # 1. Data: different inputs give 1; equal inputs give 0.
    X = torch.tensor([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])  # Shape: (4, 2) -- four samples, two features per sample.

    y = torch.tensor([
        [0.0],
        [1.0],
        [1.0],
        [0.0],
    ])  # Shape: (4, 1) -- one target per sample.

    # 2. Model, loss function, and optimizer.
    model = FeedForwardNN()

    # Fit numeric targets 0 and 1 with mean squared error.
    # The model's raw output is a number, not a probability.
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    print(model)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters())}")

    with torch.no_grad():
        initial_predictions = model(X)
        initial_loss = loss_fn(initial_predictions, y).item()
    print(f"Initial training MSE: {initial_loss:.6f}")

    # 3. Training: use all four samples for one update per epoch.
    model.train()  # Set training mode; this does not update parameters.

    for epoch in range(2000):
        optimizer.zero_grad()       # Clear gradients from the previous step.

        predictions = model(X)      # Forward pass: predict with current parameters.
        loss = loss_fn(predictions, y)  # Measure the prediction error.

        loss.backward()             # Backward pass: calculate parameter gradients.
        optimizer.step()            # Update weights and biases using those gradients.

        if epoch == 0 or (epoch + 1) % 200 == 0:
            # This loss was calculated before the current parameter update.
            print(f"Epoch {epoch + 1:4d} | MSE: {loss.item():.6f}")

    # 4. Inference: inspect predictions using the final parameters.
    model.eval()

    with torch.no_grad():
        predictions = model(X)
        predicted_labels = (predictions >= 0.5).int()
        final_loss = loss_fn(predictions, y).item()
        accuracy = (predicted_labels == y.int()).float().mean().item()

    print("\nPredictions after training:")
    for inputs, output, label, target in zip(X, predictions, predicted_labels, y):
        print(
            f"{inputs.tolist()} -> output {output.item():.4f} "
            f"-> class {label.item()} | target {int(target.item())}"
        )

    print(f"Final training MSE: {final_loss:.6f}")
    print(f"Training accuracy: {accuracy:.1%}")


if __name__ == "__main__":
    main()
