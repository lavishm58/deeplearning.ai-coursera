# week1 #

### traindevtest set ###

* deep learning is an iterative process to gain the intuition about the best hyperparameters, even 
its not opssible by great researchers to find hyperparameters in one go.

* you have to iterate over idea->code->experiment in loop to find the more n more suitable hyperparameters.

__To do this efficiently, you should have train, development(holdout or cross-validation) and test set__

### spliting of data ###
![splitting data]
(https://github.com/lavishm58/deeplearning.ai-coursera/blob/master/Images/train-dev-test.png)

*For upto 10 k examples use 60:20:20% ::train:dev:test-sets.

* But for big data
  * 1 million, use 98:1:1% :: train:dev:test.
  * for more than million may be 99.50:.25:.25
 
### mismatch of train/test distribution ###
### Example ### 
![mismatch example]
(https://github.com/lavishm58/deeplearning.ai-coursera/blob/master/Images/mismatch-train-test-dev-set.png)
__For this case make sure distribution of data examples is same in dev and test set.__

### Bias- Variance ###

__For 2-feature data__
![Bias- Variance]
(https://github.com/lavishm58/deeplearning.ai-coursera/blob/master/Images/bias-variance.png)

* high variance means overfitting.
 
 __For higher feature dim - we can compare train error and dev error__
 
 For example,this generalistion is relative to human 0% err or relative to bayes err=0%,so
 train err= 1%
 dev errv = 11 %
 means there is overfitting thats why dev err>train err
 so high variance.
 
 If,
 train err = 15%
 dev err = 16%
 than there is underfitting as train err is so large.
 so, high bias.
 
 train = 15%,dev = 30%
 high bias and variance.
 
 train = 0.5%
 dev = 1%
 low bias and var.
 
 If bayes err = 15%,then
 train err = 15%
 dev err = 16%
 this is low bias and low var.
 
 __Hence,training set err gives you bias estimate and dev gives var estimate.__
 
eg,high bias and var for 2 features -
![high bias and var]
(https://github.com/lavishm58/deeplearning.ai-coursera/blob/master/Images/bias-variance.png)
__Which is high bias in some regions and high var in some regions.

Reduce bias-var strategy

bias-var trade off,means when reducing either of them other one is not affected.
bias reduce by --- increasing network.
var low by    ---- more data.
these two wont affect the remaining one.
![low bias-var]
(https://github.com/lavishm58/deeplearning.ai-coursera/blob/master/Images/bias-var-reduce-strategy.png)


### Regularisation ###

L2 - Regularisation
Frobenius norm = || W[l] ||^2[F]
J(w,b) = 1/m*(sum(L(y,y^)))+ lambda/2m*(||W||^2[F])
W[l] = W[l] - alpha*dw - alpha*(lambda*W[l])/m 

### Why Regularisation ###
__Used only when algo is overfitting__
What it does is it reduces the impact of some hidden units if lambda is set really big, hence network becomes much
simpler and it might cause high bias or its just correct fit.

Further, it can be analysed thast if g(z)=tanh(z) ,if z=w[l]a[l-1]+b[l], lambda is big then w=w-alpha*(dw+lambda*w[l]/m
will be small,hence z will be small and in tanh graph for small values of z ,it is almost like linear hence it will underfit.

__ALways verify grad descent by plotting J with no. of iteration to see if J is decreasing__

### Dropout ###

### Inverted Dropout Technique ###
lets reduce single layer
keep_prob = 0.8
d3 = np.random.rand(a[0].shape,a[1].shape)< keep_prob
a[3]*=d3
Now,eg. a[3] dim was 50,1 then its 10 values will be zeroed out.

As, z[4]=W[4]a[3]+b[4],we dont want a[3] to be reduced by 20%.,so
a[3] /=keep_prob, to gain 20%

__Every training eg. will have diff. units zeroed out ,also every iteration of grad descent will have diff dropouts.

We can take ,diff. keep_probs for diff. layers. Further, For more no. of units weight mat keep_prob is small,for 2 or 1 unit layer keep_prob might be 1.

__AT TEST Time, NO DRPOUTS__

### Why Dropout ###
Because ,it will force to not rely on single hidden unit in a layer for the succeding layer,and spread the vaue of
weights to all the unit, its effect is almost same as L2 regularisation.
In dropout, J might not decrease smoothly with iteration.

__To,cross-check that dropout has not introduced bug,set keep_prob=1 then run and then keep_prob required then compared to see
that J and iteration graph is not altered.

### Data augmentation ###
__DATA AUGMENTATION__
To improv. overfitting, we can increase data by,
In computer vision,
* Flip the photo
* Random distortions
* Zoom

### Early stopping ###
Plot y=J for (1. train set,dev set),x=iterations ,stop when these two curves start separating.

So, it kind of mixes the two steps of orthogonalization as we stop before overfit.

__ Getting new training eg. is more suitable but data augmentation is free of cost so it will help in overfitting also like
getting new eg.

### In Practice ###
### Follow steps of Orthogonalization ###
* Optimize cost function J                         So that to fit the data at its best and reduce J as small as possible.
 * grad descent
 * momentum
 * RMS,etc
 
* Not overfit or reduce variance,                  To, reduce some fitting to get some accuracy  in dev set.
 * Regularisation
 * dropout
 * early stopping
 * data augmentation
 
 ### Optimization Techniques ###
 
 ### Normalization ###
 x=x-mu
 mu = 1/m*sum(x[i])
 sigma^2 = 1/m*sum(x[i]**2)
 __Usse same mu and sigma^2 for test set as calculated in training.__
 
 ### Why normalization ###
 * So that symmetry in J and w,b graph will increase and we will need less steps to reach minima.
* As all features value will range in -1 to 1 or similar and have mu=0 and var=1 for all will help to run algo faster.

### Problem with deep nn ###

If w[i]>1 and layers are many eg-150

then the value of z will explode.
if w[i]<1 and layer=150
z will be so small,so how to deal with this and increase layers.

__To solve this problem__
__Weight Initialization __
If there are many neurons or less in prev layer,you want not to decay too quickly or slowly,
Initialize in this way
* g(z)=relu
  use w=np.random.rand(shape)*np.sqrt(2/n)
* g(z)=tanh(z)
  use w=np.random.rand(shape)*np.sqrt(1/n)
 
 __For grad use centralized formula__
 __f'(a)=f(a+e)-f(a-e)/2e
 
 ### Gradient Checking ###
 take theta = concentanate(theta1,theta2-----)
 for i in :
    dtheta-approx[i] = J(---theta+e---)-J(---theta-e---)/2e
  For ex e=10^-7
  calcu;ate euleadian distance
  ||theta-approx-theta||2/(||theta-approx||2+||theta||)
  this is <= 10^-7 or e then grad is correct
  e=10^-5 little bit concern
  e>=10^-3 debug because there is one dtheta whose value is differing in theta[i]-theta-approx[i]
  
  ![grad-check]
  (https://github.com/lavishm58/deeplearning.ai-coursera/blob/master/Images/grad-check.png)
    
 
 
