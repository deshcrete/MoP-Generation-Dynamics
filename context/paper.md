# Understanding Generation Phenomena in Mixture-of-Personas (MoP) Post-Training

## Setup

We start with a set $\mathcal{H}$ persona descriptions with $k$ different personas. Persona descriptions look like the following:

> **Epistolary**: "You are writing a story told entirely as a series of letters between two characters. Use nothing but letters: each opens with a greeting (\"Dear ___,\") and ends with a sign-off (\"Yours,\" / \"Love,\" / \"Your friend,\") plus the writer's name. The two characters write back and forth, alternating, and the small plot unfolds only through what they say - news, questions, plans, replies. Include at least two or three letters so there is a real exchange. Write no narration outside the letters. Use only very simple, common words."

This is what we call a *triggered* persona or a persona which has a style dependent on the format of the text or the kinds of words used in a completion. This can be described also as a syntactic persona. 

> **Scientific Explainer** You are a gentle science explainer telling a short story. Write in the third person, but pause throughout to explain how and why things happen using clear cause-and-effect framing. Use phrases such as \"This happened because...\", \"The reason is that...\", and \"When ..., then ...\". Whenever something occurs, briefly explain the reason behind it (e.g. why bees help flowers grow, or why seeds are needed to plant a garden). Keep a small, friendly plot, but make the explanations a natural part of the telling. Use only very simple, common words.

This is an example of a *behavioural* persona or a persona which has a style that is more semantic and based on a voice and tone. 

We use these persona descriptions to build persona datasets $D_i \, \forall i \in \mathcal{H}$. The generation pipeline is the same pipeline as the SimpleStories dataset which we perform on the gpt-4o-mini model. We then build a mixture dataset $\mathcal{M}$ from subsets of the persona datasets with a given distribution of examples from each of the personas with proportions $\sigma_i \, \forall i \in \mathcal{H}$.

We then fine tune SimpleStories-5M models on each of the persona datasets and the mixture data on a single epoch until convergence to avoid overfitting. We measure convergence by making sure that the mixture models have smaller log-probs than persona $i$ on completions from $D_i$ (i.e the mixture model realizes a specialists completions with lower probability than the specialist). 

We then want to infer *mixture weights* $\pi_i \, \forall i \in \mathcal{H}$ after training to see what distribution over personas the model has learnt. We do this through **expectation maximization**. Let $\mathcal{S}$ be an inference dataset sampled from the mixture model representative of all the personas that the mixture model was trained on. We make an assumption that once converged a model must generate completions associated with each persona approximately in the distribution of the training dataset. More specifically, if we have triggered personas, then once converged, the probability of "trigger phrases" should almost match the training probability.

If this assumption holds, we can then perform the a **mixture MLE** analysis to infer the $\pi_i$ mixture weights. We minimize the following objective:

$$ \hat{\pi} = \argmax_{\pi} \dfrac{1}{|\mathcal{S}|} \sum_{x \in \mathcal{S}} \log \sum_i \pi_i P_i(x)$$

this is equivalent to minimizing the $\text{KL}(P_{\text{mix}} || \sum_i \pi_i P_i(x))$. We are trying to find the mixture weights that, through the persona models, "explain" the mixture models distribution using sequences generated from it as a proxy. One way to do this is using expectation maximization which alternates between the the E-step:

$$ \gamma_i(x) = \dfrac{\pi_i P_i(x)}{\sum_j \pi_j P_j(x)}$$

and the M-step:

$$ \pi_i \leftarrow \dfrac{1}{|\mathcal{S}|} \sum_{x \in \mathcal{S}} \gamma_i(x)$$

until convergence. This is an iterative optimization technique to find the mixture MLE.

In preliminary experiments, we see that our previous assumption that the sampled inference dataset will match the distribution of the mixture model is not realized. Often the mixture model will generate personas that are stylistically similar to the base model as it is very rare, especially for triggered personas, for triggers to arise in free generation. This phenomena is interesting as it is very different to what we expect and the main failure mode for the mixture MLE method. We also see emperically that for a uniform mixture distribution and a uniform inference dataset that we hand craft, the mixture MLE recovers uniform mixture weights accurately. Hence validating the method and placing the issue on the generation from the mixture model.

In this experiment we will explore the dynamics of genereation from a mixture trained model from the lens of work like **chunky post-training** and **constitutional classifiers**.

#### Is the problem with prompting?

in our setup, we dont use prompt-completion pairs, rather we just train on simple story personas directly instead of it being a prompted model and hence we dont have the resolution to reason about completions for prompt. 

## Experiments

We will carry out the following experiments in order to validate and explore the phenomenon described above. We hypothesize that tokens that trigger entry into a personas style are low marginal probability events and prompts rarely force these triggers but dropping a trigger means the whole persona is dropped in the downstream trajectory of free generation. These are our core hypotheses that we test:

1. triggered personas, whose identifying signal is concentrated in a rare trigger, gets dropped to base during generation
2. behavioural personas whose generation dont depend on a trigger and is redundant across positions survive
3. all personas are still recoverable when we force them with their triggers; the loss is in generation not representation

#### Determining Triggers for Personas

We can use **pointwise mutual information** (PMI) in order to figure out which tokens correspond highly to completions from a given persona. We can also measure how much of the total identifying signal or the total PMI is concentrated in the single top token. 

$$\text{PMI}(v, t, i) = \log \dfrac{p(x_t = v \mid y = i)}{p(x_t = v) \mid t}$$

for token position $t$, token value $v$ and persona label $i$ for a sequence $x$ taken from that personas dataset $D_i$. We can emperically calculate this over $\cup_i  D_i$.

We can then do some analysis by looking at $p(x_t = v \mid t)$ which is the marginal token distribution at position $t$ and look at the pairs $(v, t)$ that have the highest PMI by representing them in a matrix. This also allows us to classify triggered and behavioural personas

#### Using Triggers to Anchor Generation

We can have 2 different types of generation regimes:

1. *Free*: $x \sim P_{mix}$ from a neutral start token 

2. *Anchored*: supply the triggers fround previously as a part of the context and then do autoregression from there. Give each of the triggers a uniform amount of time in order to see whether the mixture model can complete personas

In previous investigation we have determined that if we just use the mixture model to generate logprobs on ground truth completions from the dataset of personas directly, then we recover an uniform distribution directly. So we hypothesize that this is not a representation issue rather an entry selection issue. 

We can also take the triggers and measure the frequency at which the trigger shows up in the generations vs the dataset. 

#### How does Persona Commitment Change over the Sequence

Another interesting question to ask is if we have the mixture model juggling between all the different personas then how does the mixture models assignment of each persona change over token number or over the sequence? 

To this end we can define the following quantities:

$$\gamma_i(t)=P(\text{persona}=i\mid x_{1:t})\propto \pi_i\prod_{s\le t}P_i(x_s\mid x_{1:s})$$

which is the cummulative posterior or the accumulated evidence for persona $i$ over the prefix which monotonically increases over the sequence length. We can also define the per token responsibility 

$$r_i(t) = \dfrac{\pi_i P_i(x_t \mid x_{1:t})}{\sum_j \pi_j P_i(x_t \mid x_{1:t})}$$

this asks the question of what persona explains the current token position $t$ given the context seen. 

We can plot these quantities over time to see how they evolve and how the persona responsibility changes. We can ask questions like is the failure purely at token 1 or does the failure compound? 