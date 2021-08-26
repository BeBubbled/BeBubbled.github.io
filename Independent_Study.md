# Independent Study

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
  * flow chart![image-20210824112213330](Literature Comments Notes.assets/image-20210824112213330-20210825132047987.png)
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
      
      * ![image-20210816044359382](Literature Comments Notes.assets/25-11-16-28-image-20210816044359382-20210825132047978.png)
    
    * [Expert Gate](https://arxiv.org/abs/1611.06194)
      
      * The gate is used to find similar old task for new task.
        
        Use the old task model's value to initialize this new task model's value.
      
      * multi task+classifier
    3. [Net2Net](https://arxiv.org/abs/1811.07017)
       
       * ![image-20210816044951261](Literature Comments Notes.assets/25-11-16-32-image-20210816044951261-20210825132047978.png)
       * add new neural while not forget

Future development of life-long learning

1. Curriculumn learning
   
   How to arrange the learning sequence?
   
   * Paper
     * [CVPR18 [Best Paper] TASKONOMY Disentangling Task Transfer Learning ](http://taskonomy.stanford.edu/#abstract)

# Literature Comments

## [EWC](https://drive.google.com/file/d/1eUoUjlcSg-DS3Tr-zn_ViPQHyJdg4bqH/view?usp=sharing)

Bayes Point of View:
$$
\begin{array}{ll}
\log( p(\theta \mid \mathcal{D}))&=log(\dfrac{p(\mathcal{D} \mid \theta)* p(\theta)}{p(\mathcal{D})})\\
\log p(\theta \mid \mathcal{D})&=\log p(\mathcal{D} \mid \theta)+\log p(\theta)-\log p(\mathcal{D})\\
\log p(\theta \mid \mathcal{D}_{1})&=\log p(\mathcal{D}_{1} \mid \theta)+\log p(\theta)-\log p(\mathcal{D}_{1})\\
\log p(\theta|D_{1})&\propto \log p(\mathcal{D}_{1} \mid \theta)+\log p(\theta)\\
\end{array}\\
$$
now considering that $\textcolor{Red}{\text{after task 1}}$ we have received new task $T_{2}$
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

Fisher information matrix F 

($\color{red}{\text{Actually haven't fully understanded}}$)

"the covariance of the gradient of the model’s log likelihood function with respect to points sampled from the model’s distribution"

Usage: 

1. estimate the variance of MLE

2. ![img](Literature Comments Notes.assets/28c4c679b6758707ed779c066d0e8e3a_1440w-9912047.jpg)
   
   e.g. estimate the curvance of the log likehood's  top point
   
   the higher, the more information you could get

3. 

$$
\begin{aligned}
\overline{\mathcal{F}} &=E_{Q_{x y}}\left[\nabla \log p(x, y \mid \theta) \nabla \log p(x, y \mid \theta)^{\top}\right] \\
&=E_{Q x}\left[E_{Q_{y}}\left[\nabla \log p(y \mid x, \theta) \nabla \log p(y \mid x, \theta)^{\top}\right]\right] \\
&=\frac{1}{N} \sum_{n}\left[\nabla \log p\left(y_{n} \mid x_{n}, \theta\right) \nabla \log p\left(y_{n} \mid x_{n}, \theta\right)^{\top}\right]
\end{aligned}
$$

Lost Function:
$$
\mathcal{L}(\theta)=\mathcal{L}_{B}(\theta)+\sum_{i} \frac{\lambda}{2} F_{i}\left(\theta_{i}-\theta_{A, i}^{*}\right)^{2}\\
$$

$\theta^{*}_{A,i}$ parameter learned from task A

Connected Papers:

![image-20210825115037345](Literature Comments Notes.assets/image-20210825115037345-9912048.png)

## MAS

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

CASSSS
