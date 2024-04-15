## Independent Study

[TOC]

# ML application

Problems that ML deal with.

- CV: object classification, object detection, object segmentation, style transfer, denoising, image generation, image caption
- Speech : speech recogniton， speech synthesis
- NLP: Machine translation, text classfication， emotional recogniton
- Recommendation System: Recommendation, CRT

# Overview Life-long Learning (LLL)

LLL also called <u>**Continuous Learning/Never ending Learning/Incremental Learning**</u>

* Resources:
  * Paper
    * [Note on the quadratic penalties](https://www.pnas.org/content/pnas/early/2018/02/16/1717042115.full.pdf)

Note:

* Knowledge Retention
  
  * how to teach NN to **remember the knowledge** while maintain the capability of learning others
  
  * but not intransigence
  
  * e.g.
    
    * QA system, dataset: bAbi corpus
    
    * 20 category questions
    
    * way to  train models:
      
      1. multi-task training (could regard it as the upper bound of the life-long learning)
         
         train multi model for each category
         
         you'll get 20 models and need a classifier
      
      2. train 1 models for random mixed QA set
      
      3. life-long
         
         ```pseudocode
         for one_category_set in all_categories:
             train model & one_category_set
         end
         return model
         ```
         
         "Catastrophic Forgetting"

* Solution 1: design a algorithm not forget (dynamic attention?)
  
  * Elastic Weight Consolidation (EWC) branch
    * [Standard EWC](http://www.citeulike.org/group/15400/article/14311063)
    * [Synaptic Intelligence (SI)](https://arxiv.org/abs/1703.04200)
    * [Memory Aware Synapses (MAS)](https://arxiv.org/abs/1711.09601)
      * do not need labelled data

* Solution 2: train NN to generate fake/pseudo data 
  
  * idea: train a model A to create <u>fake/pseudo</u> data to save disk.
  * flow chart![image-20210824112213330](https://raw.githubusercontent.com/BeBubbled/PicGoImages-WorkSpace/master/image-20210824112213330-20210825132047987.png)
    * [Continual Learning with Deep Generative Replay](https://arxiv.org/abs/1705.08690)
      * [FearNet: Brain-Inspired Model for Incremental Learning](https://arxiv.org/abs/1711.10563)
  * what if we change some pre-requisition: different task could not be undertake with same network structure
    * e.g. task 1 and task 2 use different NN
    * potential solution
      * [Learning without forgetting (LwF)](https://arxiv.org/abs/1606.09282)
      * [iCaRL: Incremental Classifier and Representation Learning](https://arxiv.org/abs/1611.07725)

* Knowledge Transfer
  
  * background
    
    1. we use multi task learning to get multi expert for different task. Now we trying to know if it is possible to increase the expert's performance when learning unrelated domain?
       * e.g. two experts, expert A focuses on ocr Handwritten Mathematical Expression (HME) image without salt noise (dataset from denoised picture) while expert B focuses on ocr blurred/unclear HME image. Can we merge A and B, so then when we feed the B 's datasets to model, we can also increase A's performance?
    2. multi experts would bring additional <u>storage/disk</u> cost
  
  * difference between life-long and transfer learning
    
    * | life-long learning                                                                                                | transfer learning                                                                                                             |
      | ----------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
      | learn A--> learn B                                                                                                | learn A --> learn B                                                                                                           |
      | If we learned A before, learn B could be better than learn B solely.<br />We **do care** model's performance on A | If we learned A before, learn B could be better than learn B solely.<br />we **don't care** if model would perform worse on A |
      
      So, transfer learning probem $\in$​​ life-long learning problem
  
  * Evaluation of life-long learning
    
    * |            | Task 1   | Task 2   | ……  | Task T    |
      | ---------- | -------- | -------- | --- | --------- |
      | Rand Init. | 𝑅0,1    | 𝑅0,2    |     | 𝑅0,𝑇    |
      | Task 1     | 𝑅1,1    | 𝑅1,2    |     | 𝑅1,𝑇    |
      | Task 2     | 𝑅2,1    | 𝑅2,2    |     | 𝑅2,𝑇    |
      | …          |          |          |     |           |
      | Task T-1   | 𝑅𝑇−1,1 | 𝑅𝑇−1,2 |     | 𝑅𝑇−1,𝑇 |
      | Task T     | 𝑅𝑇,1   | 𝑅𝑇,2   |     | 𝑅𝑇,𝑇   |
      
      $$
      \begin{array}{l}
      \text { Accuracy }=\frac{1}{T} \sum_{i=1}^{T} R_{T, i}\\
      \qquad\text{explanation: skip}\\
      \text { Backward Transfer }=\frac{1}{T-1} \sum_{i=1}^{T-1} (R_{T,i}-R_{i, i})
      \left\{
      \begin{array}{l}
      <0,usually <0\\
      =0,\\
      0,great job!\\
      \end{array}
      \right.\\
      \qquad\qquad=\text{average value of}\sum(\text{final accuracy of each task}-\text{initial accuracy of each task})\\
      \qquad\text{explanation: measure how better this life-long learning it is }\\\\
      \text { Forward Transfer }=\frac{1}{T-1} \sum_{i=2}^{T} R_{i-1, i}-R_{0, i}\\
      \qquad \text{explanation: how better it is before learn task T}
      \end{array}\\
      $$

* Model Expansion
  
  * Background
    
    * Imagine the network you designed is large enough to learn everything. Actually not. Lots of variables are wasted. How to increase your NN's efficiency?
    * or the input dataset reach the upper limitation of your NN. Is there a solution that your NN could expand itself in an efficient way automatically?
  
  * Solution
    
    * [Progressive neural networks 2016](https://arxiv.org/abs/1606.04671)
      
      * ![image-20210816044359382](https://raw.githubusercontent.com/BeBubbled/PicGoImages-WorkSpace/master/25-11-16-28-image-20210816044359382-20210825132047978.png)
    
    * [Expert Gate](https://arxiv.org/abs/1611.06194)
      
      * The gate is used to find similar old task for new task.
        
        Use the old task model's value to initialize this new task model's value.
      
      * multi task+classifier
    3. [Net2Net](https://arxiv.org/abs/1811.07017)
       
       * ![image-20210816044951261](https://raw.githubusercontent.com/BeBubbled/PicGoImages-WorkSpace/master/25-11-16-32-image-20210816044951261-20210825132047978.png)
       * add new neural while not forget

Future development of life-long learning

1. Curriculumn learning
   
   How to arrange the learning sequence?
   
   * Paper
     * [CVPR18 [Best Paper] TASKONOMY Disentangling Task Transfer Learning ](http://taskonomy.stanford.edu/#abstract)

# Literature Comments

![image-20210914194150276](https://raw.githubusercontent.com/BeBubbled/PicGoImages-WorkSpace/master/image-20210914194150276.png)

![image-20210914194417766](https://raw.githubusercontent.com/BeBubbled/PicGoImages-WorkSpace/master/image-20210914194417766.png)

![image-20210901125947070](https://raw.githubusercontent.com/BeBubbled/PicGoImages-WorkSpace/master/image-20210901125947070.png)

## Overview

### [Continual learning in neural networks](Literatures/Continual learning in neural networks.pdf)

Rethinking difference:

| -                           | $Y\quad Y^{t+1} $ | $P(X^{t})\quad P(X^{t+1})$ | $P(Y^{t})\quad P(Y^{t+1})$ |
| --------------------------- | ----------------- | -------------------------- | -------------------------- |
| incremental class learning  | $\neq$            |                            | $=$                        |
| incremental domain learning |                   | $\neq$                     | $=$                        |
| incremental task learning   |                   | $\neq$                     | $\neq$                     |

## Prior-focused

Question: Will life-long learning perform better on stable model? I guess yes. But I can not give the math proof.

e.g. For the same task 1, get model A after 10 epoch, get model B after 20 epoch, get model C after epoch 40 epoch. Based on them, we could ensure the correcrtness of A, B C shall droped.

How about their performance on task 2?

### [EWC](Literatures/Overcoming catastrophic forgetting in neuralnetworks.pdf)

* Problem  Addressing:
  
  Achieving artiﬁcial general intelligence **requires that agents are able to learn and remember many different tasks** Legg and Hutter [2007]. This is particularly difﬁcult in real-world settings: the sequence of **tasks may not be explicitly labelled, tasks may switch unpredictably, and any individual task may not recur for long time intervals**. Critically, therefore, intelligent agents must demonstrate a capacity for continual learning: that is, **the ability to learn consecutive tasks without forgetting how to perform previously trained tasks.**

* Biomedical Realted Information
  
  Papers need to read
  
  learnign new skill ? --> proportion of excitatory synapses are strengthenedA : volume increased --> months later, these dendritic spines remain
  
  When some specific spines are erased. the skill is forgotten
  
  Based on previous experimental findings, we could said continual learning is relay on task-specific stynaptics' consolidation: knowledge about how to perform a previously acquired task is durably encoded in a proportion of synapses that are rendered less plastic and therefore stable over long timescales.

Briefly speaking: Compute the 2nd derivative of the distribution to measure if current distributuion is strong enough to express the potential distribution.

Bayes Point of View:
$$
\begin{array}{ll}
\log( p(\theta \mid \mathcal{D}))&=log(\dfrac{p(\mathcal{D} \mid \theta)* p(\theta)}{p(\mathcal{D})})\\
\log p(\theta \mid \mathcal{D})&=\log p(\mathcal{D} \mid \theta)+\log p(\theta)-\log p(\mathcal{D})\\
\log p(\theta \mid \mathcal{D}_{1})&=\log p(\mathcal{D}_{1} \mid \theta)+\log p(\theta)-\log p(\mathcal{D}_{1})\\
\log p(\theta|D_{1})&\propto \log p(\mathcal{D}_{1} \mid \theta)+\log p(\theta)\\
\end{array}\\
$$
nowconsidering that $\textcolor{Red}{\text{after task 1}}$ we have received new task $T_{2}$
$$
\begin{array}{ll}
\log( p(\theta |D_{1},D_{2}))&=log(\dfrac{p(D_{1},D_{2}\mid \theta)* p(\theta)}{p(D_{2}|D_{1})})\\
\log p(\theta| D_{1},D_{2})&=\log (p(D_{1},D_{2} \mid \theta))+\log p(\theta|D_{1})-\log p(D_{2}|D_{1})\\
\log p(\theta| D_{1},D_{2})&=\log (p(D_{1}|\theta)p(D_{2}|\theta))+\log p(\theta|D_{1})-\log p(D_{2}|D_{1})\\
\log p(\theta| D_{1},D_{2})&=\log p(D_{2}|\theta)+\log p(\theta|D_{1})-\log p(D_{2}|D_{1})\\
\end{array}\\
$$

All the information about task A must therefore have been absorbed into the posterior distribution $p(\theta|D_{A)}$.

Based on [A practical bayesian framework for backpropagation networks](https://authors.library.caltech.edu/13793/1/MACnc92b.pdf) work, we use Gaussian distribution with mean equals to $\theta ^{*}_{A}$ to simulate the posterior distribution. Diagonal precision : Diagonal of the Fisher information matrix F.

**Fisher information matrix F**

$&\large\textcolor{red}{\text{Strict Deduction}}\\$

Score function used to write as $s(\theta)=\nabla_{\theta} \log p(x \mid \theta)$

"The log-likelihood function's difrst derivative"

at there, we need to know that the **log is just used to decrease the complexity of likelihood**
which means you could replace it as $ln$ or anyother function at here we choose convert it to ln

Therefore $s(\theta)=\nabla_{\theta} \ln p(x \mid \theta)$
$$
\begin{array}{l}

\underset{p(x \mid \theta)}{\mathbb{E}}[s(\theta)]&=\underset{p(x \mid \theta)}{\mathbb{E}}[\nabla \ln p(x \mid \theta)]\\
&=\int \nabla \ln p(x \mid \theta) p(x \mid \theta) \mathrm{d} x\\
&=\int \dfrac{\nabla p(x \mid \theta)}{p(x|\theta)\ln(e)}p(x|\theta) \mathrm{d} x\\\\
&\text{apply Leibniz' Rule}\\\\
&=\nabla \int \dfrac{ p(x \mid \theta)}{p(x|\theta)}p(x|\theta) \mathrm{d} x\\
&=\nabla 1\\
&=0
\end{array}
$$

The variance of $s(\theta)$ 
$$
\mathcal{I}(\theta)=Var(s(\theta))=E\left[S(X ; \theta)^{2}\right]-E[S(X ; \theta)]^{2}=E\left[S(X ; \theta)^{2}\right]=Fisher\ Information
$$
The variance of $s(\theta)$ is defined to be **Fisher information**

Wiki reference:
$$
\begin{array}{l}
&\mathcal{I}(\theta)=\mathrm{E}\left[\left(\frac{\partial}{\partial \theta} \log p(X ; \theta)\right)^{2} \mid \theta\right]=\int_{\mathbb{R}}\left(\frac{\partial}{\partial \theta} \log p(x ; \theta)\right)^{2} p(x ; \theta) d x\\

&\because \frac{\partial^{2}}{\partial \theta^{2}} \log p(X ; \theta)=\frac{\frac{\partial^{2}}{\partial \theta^{2}} p(X ; \theta)}{p(X ; \theta)}-\left(\frac{\frac{\partial}{\partial \theta} p(X ; \theta)}{p(X ; \theta)}\right)^{2}=\frac{\frac{\partial^{2}}{\partial \theta^{2}} p(X ; \theta)}{p(X ; \theta)}-\left(\frac{\partial}{\partial \theta} \log p(X ; \theta)\right)^{2}\\

&addition \because \mathrm{E}\left[\frac{\frac{\partial^{2}}{\partial \theta^{2}} p(X ; \theta)}{p(X ; \theta)} \mid \theta\right]=\frac{\partial^{2}}{\partial \theta^{2}} \int_{\mathbb{R}} p(x ; \theta) d x=0\\

&\therefore \mathcal{I}=-E\left(\frac{\partial^{2}}{\partial \theta^{2}} \log p(\mathbf{X} ; \theta)\right)
\end{array}
$$
Based on this conclusion, 

**Fisher Information meaning: the expectation of  log likelihood's minus second derivative under real value. The higher second derivative, the more import this variable is.**

e.g. normalized Bernoulli log likelihood

![image-20210831113819487](https://raw.githubusercontent.com/BeBubbled/PicGoImages-WorkSpace/master/image-20210831113819487.png)

At here, it's curvature of top point. The larger curvature is, the Bernoulli sharper log likelihood which means contain more info (Reliable).

Thererfore fisher information could be used to determine if the variable is crucial to previous task.

Lost Function:
$$
\mathcal{L}(\theta)=\mathcal{L}_{B}(\theta)+\sum_{i} \frac{\lambda}{2} F_{i}\left(\theta_{i}-\theta_{A, i}^{*}\right)^{2}\\
$$

$\theta^{*}_{A,i}$ parameter learned from task A

Connected Papers:

<img src="https://raw.githubusercontent.com/BeBubbled/PicGoImages-WorkSpace/master/image-20210825115037345-9912048.png" alt="image-20210825115037345" style="zoom:33%;" />

### [MAS](Literatures/Memory Aware Synapses- Learning what (not) to forget.pdf)

Add small perturbations to the current point to see if the function value change drasticlly.
$$
\begin{array}{l}
&L(\theta)=L_{n}(\theta)+\lambda \sum_{i, j} \Omega_{i j}\left(\theta_{i j}-\theta_{i j}^{*}\right)^{2}\\
&where\ \Omega_{i j}=\frac{1}{N} \sum_{k=1}^{N}\left\|g_{i j}\left(x_{k}\right)\right\|\\
&\sum_{i, j} g_{i j}\left(x_{k}\right) \delta_{i j} \approx F\left(x_{k} ; \theta+\delta\right)-F\left(x_{k} ; \theta\right)  
\end{array}
$$
**They all follow the same idea that extract information from the curvature of the function. Explainable**

## Model Growing

### [Progressive Neural Networks](Literatures/Progressive neural networks.pdf)

$$
h_{i}^{(k)}=f\left(W_{i}^{(k)} h_{i-1}^{(k)}+\sum_{j<k} U_{i}^{(k: j)} h_{i-1}^{(j)}\right)
$$

<img src="https://raw.githubusercontent.com/BeBubbled/PicGoImages-WorkSpace/master/image-20210914194933820.png" alt="image-20210914194933820" style="zoom:25%;" />

**Drawbacks**

1. model grows linarly with the number of trained tasks

2. Need to know task labels during test

3. 1. Avoid forgetting
   2. x Fixed memory and compute
   3. Enable forward transfer
   4. x Enable backward transfor
   5. Do not store examples

## Knowledge Distillation

### [Learning without Forgetting](Literatures/Learning without Forgetting.pdf)

| <img src="https://raw.githubusercontent.com/BeBubbled/PicGoImages-WorkSpace/master/image-20210914211145246.png" alt="image-20210914211145246" style="zoom:50%;" /> | <img src="https://raw.githubusercontent.com/BeBubbled/PicGoImages-WorkSpace/master/image-20210914212313448.png" alt="image-20210914212313448" style="zoom:50%;" /> |     |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --- |

Training Method

| Training method                   | -                                                                                                                   |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| Feature extraction                | Fix $\theta_{s}\, ,\theta_{o}$ train $\theta_{n}$                                                                   |
| Finetune                          | Fix $\theta_{s}$ train $\theta_{o}\, ,\theta_{n}$                                                                   |
| Joint training                    | re-train all layers $\theta_{s}\, ,\theta_{o}\, ,\theta_{n}$<br />**Dataset: all tasks' images&labels**             |
| Learning without Forgetting (LwF) | re-train all layers $\theta_{s}\, ,\theta_{o}\, ,\theta_{n}$<br />**Dataset: limit to the new task's images&label** |

 need experiment to explore more

# Reading Waitlist

## Knowledge representation

[Knowledge Graph Embedding: A Surveyof Approaches and Applications](https://link.zhihu.com/?target=https%3A//ieeexplore.ieee.org/stamp/stamp.jsp%3Ftp%3D%26arnumber%3D8047276)

It summarize preceding models's time complexity, space complexity, and some related equations.

### Experts-Prolog

# Toolbox

## Find suitable  journal for your paper

1. springer-journal suggester
   1. limited to the springer's journal
2. Edanz Journal Selector
3. journal finder
   1. limited to the Elsevier's journal

when you finished your paper and you dont which journal that you can send

## Get to know a field quickly

explore connected papers in a 

# Tips

## What if forget presentation ... ?

# Consideration

The Essence of Neural Network

# Review

$$
P(\theta \mid X)=\frac{P(X \mid \theta) \times P(\theta)}{P(X)}
$$





## Terminology

[^multi-head learning]: depending on what you want to predict on your data you require an adequate **backbone network** and a certain amount of **prediction heads** [citation](https://stackoverflow.com/questions/56004483/what-is-a-multi-headed-model-and-what-exactly-is-a-head-in-a-model/56004582)
















