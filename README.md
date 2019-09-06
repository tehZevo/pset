# Policy Search with Eligibility Traces
A [finite difference](http://www.scholarpedia.org/article/Policy_gradient_methods#Finite-difference_Methods)-ish approach to policy gradients.
It's like [PGET](https://github.com/tehzevo/pget), but exploring in *parameter space* instead of *action space*.

## Why?
Because, why search action space and then perform gradient descent -- which requires an expensive gradient tape/graph -- when you can just search in parameter space instead?

*(because it's easier to search in action space than it is to search in parameter space, but it's a method worth exploring regardless)*
