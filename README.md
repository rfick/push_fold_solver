Push Fold Solver
=================

This solver searches for the Nash equilibrium between the pusher and caller in Texas Holdem.
Code for determining the outcome of individual hands is courtesy of http://github.com/ktseng/holdem_calc

The code trains two adversarial neural networks against each other, one which pushes and the other calls.
Below is an example solution found with the networks trained to play at 20 big blinds:
![20bb Pusher](/images/pusher20.png)
![20bb Caller](/images/caller20.png)