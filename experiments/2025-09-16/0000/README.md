
1000 hidden unit Elman RNN with tanh nonlinearity and 0.5 dropout at readout
network but no weight decay or other regularization. Train on odd sequence
lengths in [1, 19], test on even lengths in [0, 20].

Note this is the first training run attempted after patching the early
stopping mechanism to handle the stopping condition being reached during
warmup. Warmup of 20 epochs is used here as it is hopefully large enough
to test cases where early stopping would be triggered if not for the warmup
period.
