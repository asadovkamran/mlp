# Multilayer Perceptron with Adaptive Architecture (Reforged)

*"The machine remembers. It waited eleven years for a hand to fix it."*

This work revisits a 2015 bachelor's thesis implementing a multilayer perceptron with adaptive architecture. On revision, the original implementation was found to contain several errors, most consequentially, training against an incorrect target column. The network was rebuilt from scratch in vectorized NumPy, and the adaptive mechanisms (neuron growth and pruning) were reimplemented and evaluated on two datasets with contrasting capacity requirements.

## Where the work stands

The core is done and verified. Both adaptive mechanisms work. What follows is
what I know is unfinished, so future me does not have to reconstruct it.

**Open experiments**

- **Seeds comparison.** The claim that a grown network beats a fresh network of
  the same size rests on three runs. It needs grown-to-N against
  randomly-initialized-at-N across ten or more seeds before it can be stated as
  a finding rather than an observation.
- **Cross validation as the acceptance criterion.** Both growing and pruning
  currently decide on training loss, which measures fit and not generalization.
  The `cross_validate` function already exists. Swapping it in is the honest
  version and would likely stop growth earlier.
- **Convergence based retraining.** The retrain budget per step is a fixed epoch
  count, and it silently decides what architecture gets discovered. Training
  each trial until improvement stalls would remove that dependency.

**Known limitations**

- Both test sets are small. SPECT has 15 minority samples, moons has 75 test
  points. The metrics are noisy and small differences mean little.
- Growing from 1 neuron needed 5000 retrain epochs per step. Growing from 2
  needed only 2000. The starting size changes how much training each step needs.
- Greedy stopping cannot see past a plateau. With too few retrain epochs, growth
  halts at 2 neurons even though the real gain is at 8.

**Not started**

- Introduction, related work, figures, and a results table for the paper draft.
- Multi class support. The output layer is a single sigmoid, so the network is
  binary only. Three or more classes would need a softmax output.

*Praise The Omnissiah!*