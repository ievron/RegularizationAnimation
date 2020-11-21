# Why does L1 regularization induce sparse models?

Many illustrate this using the least squares problem with a norm constraint, which is an equivalent problem to the regularized least squares problem.
The least squares level sets are drawn next to the different unit "circles".

I prepared a cool animation which I believe makes it even clearer than static images and helps develop intuition. :)

## The animation
![The outcome](https://github.com/ievron/RegularizationAnimation/blob/main/Regularization.gif?raw=true)

Also available as a video file in the file list above.

### The plotted optimization problems
The left side shows the LS problem constrained by the L2 norm, while the right side uses the L1 norm.
That is, the following problems are illustrated:

![L2 norm](https://github.com/ievron/RegularizationAnimation/blob/main/L2%20formula.png?raw=true)

![L1 norm](https://github.com/ievron/RegularizationAnimation/blob/main/L1%20formula.png?raw=true)

## The code
Can be found on the `regularization.py` script.

### Dependencies
- `numpy` 
- `matplotlib`
- [`celluloid`](https://pypi.org/project/celluloid/) (only if you want to create animations)

### Copyrights
The code is free to use but please keep the copyrights on the animations :)


## Other helpful resources
- [Youtube: Sparsity and the L1 norm by Steve Brunton](https://www.youtube.com/watch?v=76B5cMEZA4Y&feature=youtu.be&ab_channel=SteveBrunton)
- [Sam Petulla's interactive demo](https://observablehq.com/@petulla/l1-l2l_1-l_2l1-l2-norm-geometric-interpretation)
- [Mathematical explanations on CrossValidated](https://stats.stackexchange.com/questions/45643/why-l1-norm-for-sparse-models/45644)
