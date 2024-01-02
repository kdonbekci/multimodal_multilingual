Here are the models used for training and also testing the linear classifier.

The 2 view model (nv2_curriculum.pth) is just the original, self-supervised network.
Then there are a couple of model+classifiers for 4 views.
These pth files contain both the original model parameters and the linear layers (they are self contained).

The demo code at github shows how to load/run them.
