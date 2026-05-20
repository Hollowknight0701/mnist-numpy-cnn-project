# Project-1 Implementation Result

## Step 1: Core MLP operators
Implemented in `mynn/op.py`:
- `Linear.forward`
- `Linear.backward`
- numerically stable `softmax`
- `MultiCrossEntropyLoss.forward/backward`

Smoke result on MNIST subset: MLP reached about 0.87 validation accuracy after 5 epochs on a 1000-train/200-val quick subset.

## Step 2: CNN operators and model
Implemented in `mynn/op.py` and `mynn/models.py`:
- `conv2D.forward`
- `conv2D.backward`
- `Flatten`
- `Model_CNN`

CNN quick smoke result on 256-train/64-val subset: validation accuracy improved from about 0.67 to 0.72 after 2 short epochs.

## Step 3: Optimization direction
Implemented in:
- `mynn/optimizer.py`: `MomentGD`
- `mynn/lr_scheduler.py`: `MultiStepLR`, `ExponentialLR`

This can be used as one Part C direction: compare SGD vs Momentum / LR scheduling.

## Step 4: Suggested second direction
Recommended second Part C direction: error analysis and visualization.
Use confusion matrix + misclassified examples + optional convolution kernel visualization.

## Suggested final report structure
1. MLP baseline
2. CNN model and fair MLP-vs-CNN comparison
3. Direction 1: Optimization, e.g. Momentum or LR scheduler
4. Direction 2: Error analysis / visualization
5. Main results table
6. Discussion

## Notes
- Do not upload MNIST dataset or model weights to GitHub.
- Upload weights/checkpoints separately and include the link in the PDF report.
