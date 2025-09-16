
Experiment 2025-09-16/0001 seemed to confirm that slowdown was due to
repeated SVDs for computation of spectral entropy. Patched spectral_entropy
function with torch.svdvals, trying smaller weight on spectral entropy here
(0.02, still need to implement grid search capability) + weight decay with
AdamW.

As with 2025-09-26/0000 and 0001, this is a 1000 hidden unit Elman RNN with
tanh nonlinearity and 0.5 dropout at readout network. Train on odd sequence
lengths in [1, 19], test on even lengths in [0, 20].
