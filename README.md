# Deep Learning with PyTorch

A beginner-friendly series for understanding how neural networks work and how to train them. Each episode pairs an English explanation with a small, runnable PyTorch example.

## Start here: how neural networks learn

### Backpropagation: the core idea

A neural network makes a prediction using its current parameters. A **loss** measures how well that prediction satisfies the training objective. We then ask:

> If we changed a parameter slightly, how would the loss change?

The answer is that parameter's **gradient**. A positive gradient means that increasing the parameter slightly increases the loss; a negative gradient means that increasing it slightly decreases the loss.

**Backpropagation efficiently calculates these gradients by working backward from the loss through the operations that produced it.** It applies the chain rule: combine the effect of a parameter on an intermediate value with that value's effect on later values and, ultimately, the loss. This lets the loss guide learning in earlier layers as well as the output layer.

PyTorch's autograd records the relevant operations during the forward pass. Calling `loss.backward()` computes gradients for participating trainable parameters.

| Part | The question it answers |
| --- | --- |
| Loss | How well did the model satisfy the objective? |
| Backpropagation | How would small changes to the parameters affect that loss? |
| Optimizer | How should we use the gradients to update the parameters? |

**Backpropagation calculates gradients; the optimizer updates parameters.**

For basic SGD, the update is:

```math
\theta_{\mathrm{new}} = \theta_{\mathrm{old}} - \eta\frac{\partial L}{\partial\theta}
```

Here, `theta` is a parameter and `eta` is the learning rate. The learning rate controls the step size. A step that is too large can increase the loss.

### A tiny numerical example

Suppose a model predicts `prediction = w * x`. For one example, `x = 2`, the target is `6`, and the current weight is `w = 1`.

- The prediction is `2`, and the squared-error loss is `(2 - 6)^2 = 16`.
- The gradient is `2 * (prediction - target) * x = -16`.
- With learning rate `0.1`, SGD updates the weight to `1 - 0.1 * (-16) = 2.6`.
- The next prediction is `5.2`, and the loss falls to `(5.2 - 6)^2 = 0.64`.

The gradient identifies a useful local direction; the optimizer takes the step. In a deep network, backpropagation calculates gradients through many connected operations using the same chain-rule principle.

### The shared training loop

**Predict → calculate loss → calculate gradients → update parameters → repeat.**

In PyTorch, we also clear old gradients because `backward()` accumulates gradients by default. A basic loop with one parameter update per batch looks like this:

```python
# Template: each episode defines the model, loss, optimizer, and data loader.
model.train()

for inputs, targets in train_loader:
    optimizer.zero_grad()              # Clear gradients from the previous update.
    outputs = model(inputs)            # Forward pass: compute predictions.
    loss = loss_fn(outputs, targets)    # Measure the training objective.
    loss.backward()                    # Backward pass: calculate gradients.
    optimizer.step()                   # Update the trainable parameters.
```

Clearing gradients can also go between the loss calculation and `backward()`. In this basic loop, what matters is clearing the previous update's gradients before the next backward pass. It clears gradients, not learned weights.

### From FNNs to Transformers

**This learning principle applies across neural-network architectures when they are trained with backpropagation and gradient-based optimization.**

| Architecture | What changes in the forward calculation? | Shared learning principle |
| --- | --- | --- |
| FNN | Fully connected layers and activations | Loss, gradients, parameter updates |
| CNN | Convolutions and other image-processing operations | Loss, gradients, parameter updates |
| RNN / LSTM | Recurrent calculations over a sequence | Loss, gradients, parameter updates |
| Transformer | Attention, feedforward blocks, and other operations | Loss, gradients, parameter updates |

An architecture determines how inputs become outputs. The task determines the loss. The optimizer determines how gradients become parameter updates. A Transformer is more complex than our XOR network, but the basic learning principle is the same.

This statement covers gradient-based training of these architectures; it does not mean that every possible neural-network training method uses backpropagation. The loss must have a path through operations that support gradient calculation to the parameters being trained.

### A few details to keep in mind

- **The objective depends on the task.** It might compare a numeric prediction with a target, classify an image, or predict the next token. Next-token targets can come from the text itself; externally supplied human labels are not always needed.
- **Calculate the loss from outputs that support gradients.** Scores or logits are commonly used. Converting them to hard labels with a threshold or `argmax` before calculating the loss usually breaks the gradient path.
- **Training and evaluation have different purposes.** Training changes parameters. Evaluation checks performance, usually on held-out data without recording gradients. A falling training loss alone does not establish generalization.
- **Advanced loops can add extra steps.** Gradient accumulation, mixed precision, or multiple optimizers change the exact code. For example, accumulation intentionally combines several backward passes before an update and clears gradients once per group.

Start with [Episode 1: FNN](episodes/01_fnn/README.md) to see this principle in a complete runnable example.

## Episodes

| Episode | Topic | Lesson | Code |
| --- | --- | --- | --- |
| 01 | FNN: Feedforward Neural Network | [Read the lesson](episodes/01_fnn/README.md) | [Run the example](episodes/01_fnn/fnn.py) |

## Quick start

Use Python 3.10 or newer and a compatible PyTorch release. Run these commands from the repository root:

```bash
python -m venv .venv
```

Activate the environment:

- macOS / Linux: `source .venv/bin/activate`
- Windows PowerShell: `.venv\Scripts\Activate.ps1`

Then install PyTorch and run Episode 1:

```bash
python -m pip install -r requirements.txt
python episodes/01_fnn/fnn.py
```

The example runs on the CPU. A GPU and external datasets are unnecessary.

For platform-specific installation options, see the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

## Official references

- [Automatic differentiation and backpropagation](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
- [Optimizing model parameters](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
