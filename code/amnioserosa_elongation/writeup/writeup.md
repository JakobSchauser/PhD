10/12-2024
We have implemented a way of doing elongated cells.

describe

13/12-2024
It seems that squeezing (i.e. being thinner) is more stable than extending. This can be seen here:

![alt text](elongation_ratios_vs_Ns.png "First plot of PhD")

Not sure what to make of this.


16/12-2024

Ahhh! I fell victim to one of the classic blunders! The first of which is never going into a land war in Asia, the second of which is assuming the dot product between uniformly distributed vectors is uniform.

![alt text](image-1.png)

It now works better!

![alt text](elongation_ratios_vs_Ns_better.png "First fix of PhD")

17/12-2024

Okay! 

It seems like my dream of adding somewthing worthwile to the model is inching closer.

The added "elongation factor" works like a charm when added to the potential and minimized $(\frac{dV}{de})$ alongside the rest of the parameters (r, p, q)

For a slowly squeezing (in the y-axis) potential, the cells behave like this: 

![alt text](first_working_passive_elongation.png "alt text")

As a

Why works

Next step w lambda2 

Okay, a lot happens today - that also works!

![alt text](with_lambda2.png "alt text")


Trying with no, does not automatically make them align (even tried making the dot product into something tanh-like) 

(it makes rosettes)

![alt text](frustreated_when_no_l2.png "Frustrated")


Wait a minute it orients when elongation is clamped!

![alt text](it_orients_pcp.png "on a roll today")

Now I will try pulling:

