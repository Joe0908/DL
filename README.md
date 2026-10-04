# Deep Learning with PyTorch

A beginner-friendly series for understanding how neural networks work and how to train them. Each episode pairs an English explanation with a small, runnable PyTorch example.

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
