# Multilayer Perceptron with Adaptive Architecture (Reforged)

*"The machine remembers. It waited eleven years for a hand to fix it."*

A multilayer perceptron built from scratch in NumPy. It is trained on the SPECT heart imaging dataset, a binary classification task. This is a rebuild of my 2015 bachelor's thesis, an MLP with adaptive architecture. I brought it back, cut it down to the core, and made it work the way it never really did the first time.

---

## The Long Night

I first built this in 2015. It ran, and it even predicted something. But when I opened the old code years later to fix it, I saw the rust everywhere. The machine had been lying to itself the whole time.

There were five flaws hidden inside it.

- **The bias was frozen.** I never updated the bias term. It was fixed at 1, a dead weight the network could never learn to move.
- **Backprop ran on updated weights.** I was updating the output weights first, then using those same changed weights to send the error back to the hidden layer. The gradients were computed against a machine that had already shifted.
- **It was slow.** Everything ran through Python loops, one matrix cell at a time. That is millions of trips through the interpreter for a network of only eighty patients.
- **Accuracy was a false oracle.** I judged the model on accuracy alone. On imbalanced SPECT data, where 92 percent is one class, accuracy makes everything look good. A model that always guesses the majority class scores about 92 percent and learns nothing.
- **The target column was wrong.** This was the worst one. I was training against the wrong column. I fed the network a meaningless feature as its answer and asked why it could not learn. The label was column 0. I had been reading the last column.

The old code was 300 lines. Most of those lines were where the bugs lived.

## The Reforging

I rewrote it in **NumPy**, because it gives fast vectorized math that runs as compiled C under the hood instead of crawling through the Python interpreter. It is the same math, done for all eighty patients at once, and much faster. Every loop that used to walk through the weight matrices by hand is now a single matrix multiply. The code is faster, and it is easier to keep clean. Fewer lines means fewer places for bugs to hide.

**300 lines became 90.**

What I fixed:

- **The bias is free now.** `b1` and `b2` are real learnable parameters. They are updated every step, next to the weights.
- **A clean wall between forward and backward.** Backprop now computes every gradient from the original weights and changes nothing. The updates happen after, in their own step. The old ordering bug is now impossible.
- **Weights start centered on zero** with `uniform(-0.5, 0.5)`, so the hidden units start pointed in different directions instead of all bunched on one side.
- **Binary cross entropy** instead of plain squared error, because it is the honest loss for a 0 or 1 problem. I use `np.clip` to guard the logs against `log(0)`.
- **The right column.** The label is column 0. The machine finally trains toward the truth it was always meant to predict.

## Judging the Machine

Accuracy alone is a false oracle on imbalanced data, so I do not trust it on its own anymore. The model is now judged with a **confusion matrix, precision, recall, and F1**. I look at these per class, so the rare class cannot hide behind the common one.

The result is honest, and it does not flatter the model.

| Metric        | Class 0 (rare) | Class 1 (common) |
|---------------|:--------------:|:----------------:|
| Precision     | 0.21           | 0.96             |
| Recall        | 0.67           | 0.78             |
| F1            | 0.32           | 0.87             |
| Support       | 15             | 172              |

**Accuracy: 77.5 percent. Macro F1: 0.59.**

The gap between those two numbers is the lesson. Accuracy says the machine is good. The macro F1 tells the truth. The model learned the majority class well, with F1 0.87, and barely understands the minority class, with F1 0.32. This comes straight from the 15 versus 172 imbalance. On imbalanced data I report the macro average, because it gives the rare class an equal vote and cannot be fooled by the crowd.

## How to Run

```bash
python mlp_v3.py
```

It needs `numpy`. The metrics use `scikit-learn` and its `classification_report`, which is the standard tested tool. The network itself, meaning the forward pass, the backprop, and the training loop, is built entirely by hand.

## Data

SPECT heart imaging dataset. It has 22 binary features and one binary diagnosis. I use `spect_train.txt`, which has 80 rows balanced 40 and 40, to train it, and `spect_test.txt`, which has 187 rows and is imbalanced, to test it.

## The Unfinished Work

The forge is lit again, but the blade is not perfect yet. The minority class is still the weak point, and those false negatives are the next thing to fix.

- Class weighting or resampling, to make the rare class count for more.
- Threshold tuning, to trade some precision back for recall.
- Bringing back the adaptive architecture, the growing and self pruning network from the original thesis, now that there is a clean and tested core to build it on.

---

*Rebuilt in 2026. The code had not moved in eleven years. I had. Praise The Omnissiah!*