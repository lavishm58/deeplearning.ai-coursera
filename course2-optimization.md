Mini batch grad descent instead of batch grad descent

__Common technique used for huge data set faster computation__
What it does is take mini batch of sizes 64 to 512 ex. normally and compute and update wqeights and cost using this,
it will surely help to get minima in less go.
If mini-batch-size =1 ,it is called stochastic grad desecnt

Make sure that if a single training ex. is not fit by cpu/gpu memory then performance gets 
much worse,a faster one is required.

Exponentially weighted avg

vt = B*v[t-1] + (1-B)*theta[t]

###Grad descent with Momentum ###
It is much faster than normal grad des.

Idea is use exponential weighted avg . of grads and then update weights.

### Implemeentation ###
at iteration t:
compute dw,db
V[dw] = B*V[dw]+(1-B)*dw
w=w-alpha*V[dw]

No. of last terms included = 1/1-B
Momentum allows to lessen the updwn oscillations and move forward to minima

### RMSprop ###
S[dw] = B*S[dw] + (1-B)*dw2
w=w-alpha*dw/np.sqrt(S[dw])

### Adam optimization algo ###
It combines rms and momentum

V[dw] = B1*V[dw]+(1-B1)*dw
S[dw] = B2*S[dw]+(1-B2)*dw**2

Bias correction - 
V[dw]=V[dw]/(1-B1**t)
S[dw]=S[dw]/(1-B2**t)

update - 
w=w-alpha*V[dw]/np.sqrt(S[dw])

### Learining Rate decay ###
With mini batch,alpha will gradually decrease
alpha = (1/(1+decay_rate*epoch_num))*alpha-nod

Other deacy formula's
alpha = (some num <1)**epoch_num*alpha-nod
alpha =k/np.sqrt(epoch_num)*alpha-nod
