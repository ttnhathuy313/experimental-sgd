# Question 1: Minimize a hidden function (No explicit formula) using gradient-base methods

## Done:

- [x] SGD

- [x] Momentum

- [x] Nesterov

- [x] Adam

- [x] Autodiff - Avaiable to compute both gradient and hessian

- [x] Inverse matrices - Use package numpy.linalg

- [x] Existing code from thầy Hiếu for backtracking
 
## TODO:

- [ ] Choose a fixed algorithm, or come up with a strategy to use combinations.

- [ ] Hyperparams (step size, optimizer hyperparams,...) - Adjust learning rate for better performance

- [ ] Any needs for other optimizers, line search, adaptive step size,...?

- [ ] Test - Double-check?

- [ ] Make the program runs longer by release/delete some stopping constraints/conditions

- [ ] Anything else?

## Files:

- f_optimize.py: containing autodiff, optimizers, line search, main optimization

- run_f_optimize_test.ipynb: test file with common functions (in-class, assignments)

## Run:

- Directly import f_optimize to your program and call f_optimize()

- Examples can be found in the notebook run_f_optimize_test.ipynb
