### Gradient Descent 



**Gradient descent** is an optimization algorithm used to minimize some function by iteratively moving in the direction of **steepest descent** as defined by the negative of the **gradient**. In machine learning, we use **gradient descent** to update the parameters of our model.



1. **Initialize** the weights *W* randomly.
2. **Calculate the gradients** *G* of cost function w.r.t parameters. This is done using partial differentiation: *G = ∂J(W)/∂W.* The value of the gradient *G* depends on the inputs, the current values of the model parameters, and the cost function. You might need to revisit the topic of differentiation if you are calculating the gradient by hand.
3. **Update the weights** by an amount proportional to G, i.e. *W* = *W - ηG*
4. Repeat until the cost *J*(*w*) stops reducing, or some other pre-defined **termination criteria** is met.

```pseudocode
Input: X∈Rn×d,y∈Rn,w0 =0d,b0 =0,maxpass∈N,η>0,tol>0 Output: w,b
for t = 1,2,...,max pass do
		wt ← wt−1 − ηw(n1 X⊤(Xwt−1 + bt−11 − y) + 2λwt−1) 
		bt ← bt−1 − ηb(n1 1⊤(Xwt−1 + bt−11 − y)).
    if ∥wt − wt−1∥ ≤ tol then
    		break w←wt,b←bt
// can use other stopping criteria
```

