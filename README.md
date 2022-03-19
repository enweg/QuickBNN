# QuickBNN
Quick Construction of BNNs using Flux like syntax and Turing as a backend

QuickBNN is a work in progress. It allows the construction of genereal Baysian Neural Networks following syntax similar to Flux. Turing is used as an inferential backend. 

Example
```julia
using Turing
using Flux
using QuickBNN

y = AR([0.5])
x = y[1:end-1]
y = y[2:end]

net = ChainBNN(DenseBNN(1, 2, :sigmoid), DenseBNN(2, 1))
bnn = BNN(net)
model = bnn(y, hcat(x...))
chain = sample(model, NUTS(), MCCMThreads(), 10_000, 4)
plot(chain)


```
