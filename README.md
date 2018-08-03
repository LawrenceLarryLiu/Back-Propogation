# Back Propogation

New York University  
CSCI-UA 480: Computer Vision  
Davi Geiger  
Spring 2018  

This project uses a back propagation algorithm to train a two-layer XOR problem and a two-layer multiplication problem.

For a much more detailed explanation, see the pdf.

### Case 1: XOR

The first case to study is the well known XOR example. For this example, the hidden layer maybe simply made of two units, N = 2, and the input and target output are binary values. There are only four different possible inputs/outputs scenarios:

(0, 0) -> 0  
(0, 1) -> 1  
(1, 0) -> 1  
(1, 1) -> 0

For the training set, validation set, and generalization set, we only have these four examples for each step. The plots are visualized with the final generalization loss in the graphs, and the weights of the network are listed in the results.

### Case 2: Multiplication

The second example is the multiplication function. The inputs are x1 and x2, and they will be integers. The target output should be t = x1 × x2 -> also an integer number. Note that it should work on negative numbers as well, i.e., x1, x2, t ∈ Z. We may restrict further to x1, x2, t ∈ (−M, M), say M = 100. It is easy to generate a large test set, but not a large validation set and an even larger generalization set. Ensure that cases of multiplication by zero and by 1 work as intended.The same plots and values as the XOR problem are displayed in the same folders, albeit there is a higher error value for this problem due to the relatively small amount of training my model did. The lowest error value I could reach was 15, which would likely have gone lower had I used a smaller learning rate on a more powerful machine.
