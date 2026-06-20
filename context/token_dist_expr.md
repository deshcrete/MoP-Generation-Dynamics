# Analyzing Generation Dynamics from Token Distributions

## Context

We see in the results of previous experiments that at the first token during the generation, the mixture model maintains a high fraction of "belief" over all the personas however, after generating the first token, if a trigger is not generated then the belief about the triggered is dropped completely. If we want to measure/probe the models belief over personas, this motivates looking at the model's distribution over tokens as a natural object to investigate this phenomena further. We also have observed that if the first token is selected, then the persona locks and the mixture models belief of any of the personas especially the triggered persona increases drastically. So essentially we can triangulate the persona "selection" or "collapse" as existing within the token distribution of the early tokens during generation. 

The central discrepancy that we want to investigate here is the assumption that:

> The mixture model should generate completions in the style of personas or generate triggers for personas at the proportion that the personas appear in the training dataset

We want to explore why these triggered personas dont trigger and the interesting question here is whether this phenomena is just a feature of free generation and if so then how does the belief over tokens differ for the mixture and specialist models. Ideas we could use to investigate this icnludes trying to investigate the posterior probabilities of various rollouts, compare the token position distributions and compare generations between specialists and the mixture model. 

## Experiments

### Comparing Token Distributions

1) extract the first token distribution for the mixture model and compare it with the specialist models. We want to graph the distributions but we can also try to calculate the following:

$$ \hat{w}(t) = \argmin_{w \in \Delta} \text{KL}(P_{\text{mix}}(\cdot \mid x_{<t}) \mid\mid \sum_iw_iP_{i}(\cdot \mid x_{<t}))$$

which is doing the same EM inference but on the first token distributions. Here we want to do this over token position 1 or $t=1$ but we can also see what happens at further tokens

2) the other thing would be to see given the same first token or trigger, over the rollout, how does the next token distribution change over token position.

### Individual Persona Rollouts

In the previous experiments, we calculated 

$$\gamma_i(t)=P(\text{persona}=i\mid x_{1:t})\propto \pi_i\prod_{s\le t}P_i(x_s\mid x_{1:s})$$

however we averaged this gamma across rollouts for each persona. This meant that we were not able to analyze which tokens where updates happened the most and this means that we cant differnetiate high frequency updates from low frequency updates. Analyzing this could also tell us especially for behavioural personas, where uppdates happend the most. We think that 
- Behavioural personas $\rightarrow$ low frequency updates so individual tokens provide very little evidence or updates for belief for a behavioural persona, more gradual update over tokens
- Triggered personas $\rightarrow$ high frequency updates so individual tokens give a lot of evidence / update highly for a particular persona