# Episode 1: FNN

**FNN** stands for **feedforward neural network**. In this episode, you will train one in PyTorch and understand what happens during each training step.

The goal is to connect five ideas: data, a model, a loss function, gradients, and parameter updates.

## 1. What is a feedforward neural network?

An FNN passes information from its input through hidden layers to its output. In the forward calculation, information does not loop back into an earlier layer.

Each fully connected layer combines its inputs using **weights** and **biases**. A weight controls how strongly an input contributes; a bias adds an adjustable offset. These parameters start with initial values and change during training.

Our network has two hidden layers:

| Stage | Operation | Output shape for four samples |
| --- | --- | --- |
| Input | Two numbers per sample | `(4, 2)` |
| Hidden layer 1 | `Linear(2, 8)`, then ReLU | `(4, 8)` |
| Hidden layer 2 | `Linear(8, 8)`, then ReLU | `(4, 8)` |
| Output layer | `Linear(8, 1)` | `(4, 1)` |

A hidden layer's eight values are intermediate features the network learns to compute. We do not assign each neuron a predefined meaning.

## 2. The task: learn XOR

XOR gives `1` when its two binary inputs differ and `0` when they are equal.

| Input 1 | Input 2 | Target |
| --- | --- | --- |
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

The training code supplies these examples. It does not contain an XOR rule inside the model. The network must adjust its parameters to fit the targets.

XOR is useful because a single linear decision boundary cannot separate its two classes. The hidden layers and nonlinear activations let the network represent a more flexible relationship.

## 3. Run the example

From the repository root, after installing the dependencies:

```bash
python episodes/01_fnn/fnn.py
```

Open [fnn.py](fnn.py) alongside this lesson. It contains the complete example, including English comments and printed training results.

## 4. Understand the model

### Creating the layers

```python
self.fc1 = nn.Linear(2, 8)
self.fc2 = nn.Linear(8, 8)
self.fc3 = nn.Linear(8, 1)
```

`nn.Linear(2, 8)` takes two features per sample and produces eight. PyTorch creates its weights and biases automatically and enables gradient tracking for these parameters.

For a batch, a linear layer computes:

```python
output = inputs @ weight.T + bias
```

The first layer has `8 * 2 + 8 = 24` parameters. The second has `8 * 8 + 8 = 72`, and the output layer has `1 * 8 + 1 = 9`. The model has **105 trainable parameters** in total.

The input features and target labels do not need gradients: we are learning the model's parameters.

### Defining the forward pass

```python
def forward(self, x):
    h1 = self.relu(self.fc1(x))
    h2 = self.relu(self.fc2(h1))
    output = self.fc3(h2)
    return output
```

Calling `model(X)` runs this forward calculation.

ReLU keeps positive values and replaces negative values with zero. It introduces **nonlinearity**. Without nonlinear activations, several linear layers can be combined into one affine transformation, which is insufficient for XOR.

The final layer has no sigmoid, so its output is an unrestricted number. We train it to approach the numeric targets `0` and `1`, then use `0.5` as the classification threshold. Its output is **not a probability**.

## 5. How training works

### Measure the error

```python
loss_fn = nn.MSELoss()
loss = loss_fn(predictions, y)
```

We use **mean squared error (MSE)**:

$$
L = \frac{1}{4}\sum_{i=1}^{4}(\hat y_i-y_i)^2
$$

Here, `predictions` contains the model's four outputs and `y` contains the four correct answers. A smaller loss means the outputs are closer to their targets.

MSE keeps this first example easy to inspect. Binary classification is also commonly trained with binary cross-entropy; that uses a different loss and output interpretation.

### Choose how to update parameters

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
```

`model.parameters()` provides the weights and biases from all three layers. The optimizer will update all of them.

The learning rate, `0.1`, sets the size of the gradient-based adjustment. With the default SGD settings used here, each parameter follows:

$$
\theta_{\mathrm{new}}
=
\theta_{\mathrm{old}}
-
\mathrm{learning\ rate}
\times
\frac{\partial L}{\partial\theta}
$$

A gradient describes how the loss changes when a parameter changes slightly. Moving against the gradient is intended to reduce the loss. A step that is too large can increase it.

### Read the five key lines

```python
optimizer.zero_grad()
predictions = model(X)
loss = loss_fn(predictions, y)
loss.backward()
optimizer.step()
```

| Line | What it does |
| --- | --- |
| `optimizer.zero_grad()` | Clears previous gradients because PyTorch accumulates gradients by default. |
| `predictions = model(X)` | Computes predictions using the current parameters. |
| `loss = loss_fn(predictions, y)` | Measures how far those predictions are from the targets. |
| `loss.backward()` | Uses the chain rule to calculate the loss gradient for each trainable parameter. |
| `optimizer.step()` | Uses those gradients to update the parameters. |

**Backward propagation calculates gradients. The optimizer changes the parameters.**

The gradients extend through the hidden layers, so training changes the entire network rather than only its output layer.

In this example, each **epoch** uses all four samples together as one batch, followed by one update. Larger datasets are often divided into multiple batches, producing multiple updates per epoch.

## 6. Inspect the result

The script prints the initial loss, the loss during training, and the final predictions.

The intended result is:

| Input | Output should approach | Predicted class |
| --- | --- | --- |
| `[0, 0]` | 0 | 0 |
| `[0, 1]` | 1 | 1 |
| `[1, 0]` | 1 | 1 |
| `[1, 1]` | 0 | 0 |

Exact decimal values can vary across PyTorch versions and environments. The printed training loss is calculated before each reported update; the final loss is recomputed after training.

`model.train()` selects training mode and `model.eval()` selects evaluation mode. These calls do not update parameters or disable gradients. They affect layers such as dropout and batch normalization; our model has neither, so they do not change its numerical behavior.

`torch.no_grad()` disables gradient tracking while we inspect predictions.

All four examples were used for training. The final accuracy shows whether the model fitted them; it does not measure generalization to an independent test set.

### Verified run

The supplied example was run on CPU with Python 3.12 and PyTorch 2.14.1. With the provided seed and 2,000 epochs, it classified all four training examples correctly:

```text
Initial training MSE: 0.686959
[0.0, 0.0] -> output 0.0000 -> class 0 | target 0
[0.0, 1.0] -> output 1.0000 -> class 1 | target 1
[1.0, 0.0] -> output 1.0000 -> class 1 | target 1
[1.0, 1.0] -> output 0.0000 -> class 0 | target 0
Final training MSE: 0.000000
Training accuracy: 100.0%
```

The printed zero loss is rounded to six decimal places.

## 7. Try changing one thing

Run the original example first, then make one change at a time.

| Experiment | What to inspect |
| --- | --- |
| Reduce `2000` epochs to `20`. | Has the loss decreased enough to classify all four examples? |
| Reduce the learning rate from `0.1` to `0.01`. | Does learning take more updates? |
| Replace both ReLU calls with plain linear-layer calls. | Can the model fit all four targets without nonlinearity? |
| Change the random seed. | How do different initial parameters affect learning? |

To inspect an actual gradient, insert this line immediately after `loss.backward()`:

```python
print(model.fc1.weight.grad)
```

These values are gradients for the first layer's weights. They tell the optimizer how to adjust those weights.

## Official references

- [Build a neural network](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)
- [Automatic differentiation](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
- [Optimize model parameters](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
