
Repeat of 2025-09-16/0002 but with patience of 20 instead of 8 (training
cfg 2025-09-16/x/0001 is the same as training cfg 2025-09-16/x/0000 except
with pateince of 20 instead of 8). 

Again this is a 1000 hidden unit Elman RNN with tanh nonlinearity and 0.5
dropout at readout network. Train on odd sequence lengths in [1, 19], test
on even lengths in [0, 20].
