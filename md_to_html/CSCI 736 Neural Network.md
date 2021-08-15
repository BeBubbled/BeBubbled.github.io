



# CSCI 736 Neural Network

[toc]



# Paper Time Line



 Write 1 and 5

| Paper Content                                                | Time Schedule                                                |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="https://raw.githubusercontent.com/BeBubbled/PicGoImages-WorkSpace/master/5871614469525_.pic_hd.jpg" alt="5871614469525_.pic_hd" style="zoom: 25%;" /> | <img src="https://raw.githubusercontent.com/BeBubbled/PicGoImages-WorkSpace/master/5991614469878_.pic_hd.jpg" alt="5991614469878_.pic_hd" style="zoom:25%;" /> |

## Paper Math Equation

$$
\begin{aligned}
&p_{end,j}=\dfrac{e^{z_{j}}}{\sum{j=n+1}^{2n} e^{z_{j}}} , j \in\{n+1, n+2, n+3 \ldots 2 n\}\\
&input=[q_{1},q_{2},q_{3},\cdots, q_{N}],[d_{1}, d_{2}, d_{3}, \cdots,d_{M}], \text{(N,M : the number of tokens)}\\
&outputs=[index\ \ of\ \ max(p_{i}),index\ \ of\ \  max(p_{j}))]\\

&p_{start}=\frac{e^{z_i}}{\sum\limits_{i=1}^{n}e^{z_i}}, i \in \{1,2,3\cdots, M\}\\

&p_{end}=\frac{e^{z_{j}}}{\sum\limits_{j=1}^{n} e^{z_{j}}},\\ &j \in\{1, 2, 3 \ldots M\} \\

&loss=-log(p_{start,I})-log(p_{end,J})\\
&\text{the correct start and end position are }\\&
I \in\{1,2,3,4 \ldots \mathrm{n},J \in\{1,2,3,4 \ldots \mathrm{n}\}
\end{aligned}
$$

## Learning Resource

[李宏毅](https://www.bilibili.com/video/BV1JE411g7XF?from=search&seid=13732374700367344665)



# 文章理解

embedding(八种常用的embedding)->rnn->lstm&gru->attention->seq2seq->self-attention->transformer->bert

**embedding**

**RNN**

<img src="https://raw.githubusercontent.com/BeBubbled/PicGoImages-WorkSpace/master/v2-3884f344d71e92d70ec3c44d2795141f_1440w-20210308084120851.jpg" alt="v2-3884f344d71e92d70ec3c44d2795141f_1440w" style="zoom:25%;" />

![v2-b0175ebd3419f9a11a3d0d8b00e28675_1440w](https://raw.githubusercontent.com/BeBubbled/PicGoImages-WorkSpace/master/v2-b0175ebd3419f9a11a3d0d8b00e28675_1440w-20210308084146454.jpg)

![v2-9e50e23bd3dff0d91b0198d0e6b6429a_1440w](https://raw.githubusercontent.com/BeBubbled/PicGoImages-WorkSpace/master/v2-9e50e23bd3dff0d91b0198d0e6b6429a_1440w-20210308084225295.jpg)

### **LSTM**

### **GRU**

### **attention**

参数少, 速度快, 效果好

[优质教程](https://shangzhih.github.io/jian-shu-attentionji-zhi.html)

### **Encoder&Decoder**

一类算法的统称

这类算法的统称:

1. 无论输入和输出的长度是什么, 中间的 向量c 长度固定
2. 根据不同任务可以选择不同的编码器和解码器

缺点: 当输入信息太长时，会丢失掉一些信息.

### seq2seq $\in$ Encoder & Deconder

一类算法的统称

这类算法的统称:	满足输入序列, 输出序列的目的

### **self-attention**

### **transformer**

seq2seq model with "self-attention"

### **bert: unsupervised transformer**

linear classifier: two vector

each vector dot product the embedding, then apply softmax, find maximum to get index



























### **Deep Auto-encoder**

Paper:[Deep  Auto-Encoder  Neural  Networks  in  Reinforcement  Learning](Papers/Deep  Auto-Encoder  Neural  Networks  in  Reinforcement  Learning.pdf)

PPT:[Unsupervised Learning-Auto-encoder](李宏毅PPT/Unsupervised Learning-Auto-encoder.pptx)

<img src="https://raw.githubusercontent.com/BeBubbled/PicGoImages-WorkSpace/master/image-20210309201413384.png" alt="image-20210309201413384" style="zoom:50%;" />

Stating from PCA

  <img src="https://raw.githubusercontent.com/BeBubbled/PicGoImages-WorkSpace/master/image-20210309202019613.png" alt="image-20210309202019613" style="zoom:50%;" />

可以把PCA的前半部分视为encode, 后半部分decode

**重要特性:增强robust**

<img src="https://raw.githubusercontent.com/BeBubbled/PicGoImages-WorkSpace/master/image-20210309210413591.png" alt="image-20210309210413591" style="zoom:50%;" />



对CNN建立decoder, decoding的过程实际还是在卷积

<img src="https://raw.githubusercontent.com/BeBubbled/PicGoImages-WorkSpace/master/image-20210309220008147.png" alt="image-20210309220008147" style="zoom:50%;" />



| 图一                                                         | 图二                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="https://raw.githubusercontent.com/BeBubbled/PicGoImages-WorkSpace/master/image-20210309215606227.png" alt="image-20210309215606227" style="zoom:50%;" /> | <img src="https://raw.githubusercontent.com/BeBubbled/PicGoImages-WorkSpace/master/image-20210309215051149.png" alt="image-20210309215051149" style="zoom:50%;" /> |



这一页ppt只是简单讲了下如何世纪一个discrminator 来保证auto-encoder decoder效果好(decoder尽可能把vector还原为原始图像)

所以这启发我们: 我们需要一个足够好的能够衡量encoder与decoder的discriminator/classifier来监督encoder与decoder的训练与一个足够好的encoder来保证vector与原始图像能尽量一一对应(每个原始数据尽量能有独一无二的embedding)

参考文章: Deep InfoMax (DIM)





若训练集是sequential, skip thought

## skip thought->quick thought https://arxiv.org/pdf/1803.02893.pdf

quick thought 只认encoder不管decoder, 每一个句子的ebedding跟他下一个句子的embedding越接近越好, 跟随机的句子的emberdding越不同越好



quick设计的classifier:输入句子A应用encoder产生的embedding, 句子A的下一句应用encoder产生的embedding, 一对随机句子应用encoder产生的embedding, classifier需要能够认为句子A的下一句巨句子A的相似度最高

这样的classifier与产生这个embedding的encoder同时训练

## Feature Disentangle

<img src="https://raw.githubusercontent.com/BeBubbled/PicGoImages-WorkSpace/master/image-20210309222925667.png" alt="image-20210309222925667" style="zoom:50%;" />



| -                                                            | -                                                            |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="https://raw.githubusercontent.com/BeBubbled/PicGoImages-WorkSpace/master/image-20210309223058277.png" alt="image-20210309223058277" style="zoom:50%;" /> | <img src="https://raw.githubusercontent.com/BeBubbled/PicGoImages-WorkSpace/master/image-20210310001401154.png" alt="image-20210310001401154" style="zoom:50%;" /> |

假设encoder返回的前100个embedding 放入speaker classifier中进行训练 ,直到speaker classifier无法区分出那种音色, 这时候就认为前100已经没有了音色信息, 音色信息跑到了后100个中





Instance normalization:一种特逼得layer, 可以抹掉不想要的信息

比如全部抹掉音色信息 ,那么剩下的就是纯正的语音信息, 具体方案依赖Gan实现



## Vector Quantized Variational Auto-encoder (VQVAE)

| -                                                            | -                                                            |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20210310001750278](https://raw.githubusercontent.com/BeBubbled/PicGoImages-WorkSpace/master/image-20210310001750278.png) | <img src="https://raw.githubusercontent.com/BeBubbled/PicGoImages-WorkSpace/master/image-20210309225757217.png" alt="image-20210309225757217" style="zoom:100%;" /> |

返回的vector对其内容做one-hot(最大的变1其余0)转换或者binary转换(设定threshold, 大于的变为1其余0), 推荐binary, 这样可以意外的发现训练集中原本不存在的cluster



假设codebook中只有5个vector, 则encoding返回的vector与这五个做相似度比较, 与codebook中哪一个相似就把codebook中的哪个返回给decoder

## seq2seq2seq

![](https://raw.githubusercontent.com/BeBubbled/PicGoImages-WorkSpace/master/image-20210310002011618.png?token=ANU6SUCSYTX2DZFLE7FA37TAJN6VE)



## Gan

generator+ranfdom vector-> target high dimensional vector

discriminator:像二次元则高分, 否则低分



不仅训练gnererator还需要discriminator





### Conditional Gan (Supervised)

generator有可能会发现某一种一旦可以骗过discriminator后, 就不再改变自己, 无视输入, 时钟输出同一个东西来欺骗discirminator

现在我们修改discirminator, 不仅判断generator的结果有多好, 还判定generator的输入与输出有多匹配

1. text->image



​	好图片+好text=1

​	好图片+烂text=0=兰图片+好text

2. sound to image

​	e.g. 电视雪花声->瀑布,声音越大, 瀑布越猛

​	类似直升机的声音->快艇海上行走, 声音越大, 快艇引起的水花越大

3. image->text

   e.g. image-> multi label

## Unsupervised Gan Cycle Gan



## Hung-yi Lee Generative Adversaria

Generator: a neural network

![image-20210312100521080](https://raw.githubusercontent.com/BeBubbled/PicGoImages-WorkSpace/master/image-20210312100521080.png?token=ANU6SUHESE5OJKYOXT4H4LDAJOBW4)

![image-20210312100538459](https://raw.githubusercontent.com/BeBubbled/PicGoImages-WorkSpace/master/image-20210312100538459.png?token=ANU6SUHKM2B33NXIPNYRHVTAJOBX6)



## Gan discriminator predict

假设数据集内部元素呈现线性分布, 于是可以遍历的方式拿到所有可能的数据集, 这些数据及在discriminator中分数最高的即为预测值, 这些数据集中, 属于training的应该让dircriminator给出高分, 不属于training的dircriminator应该给出低分, 借此完成discirminatro的独立training



## Gan Feature Extraction

### infoGan



 InfoGan

![image-20210312133609315](https://raw.githubusercontent.com/BeBubbled/PicGoImages-WorkSpace/master/img/image-20210312133609315.png?token=ANU6SUEXPL2MP5BVDCFCXR3AJO2NM)

假设我们打算生成像MNIST那样的手写数字图像，每个手写数字可以分解成多个维度特征：代表的数字、倾斜度、粗细度等等，在标准GAN的框架下，我们无法在上述维度上具体指定Generator生成什么样的手写数字。

为了解决这一问题，文章对GAN的目标函数进行了一些小小的改进，成功让网络学习到了可解释的特征表示（即论文题目中的interpretable representation）。

[infoGan理解](https://zhuanlan.zhihu.com/p/58261928)

### VAE-GAN

![image-20210312134055741](https://raw.githubusercontent.com/BeBubbled/PicGoImages-WorkSpace/master/image-20210312134055741.png)

![image-20210312134417230](https://raw.githubusercontent.com/BeBubbled/PicGoImages-WorkSpace/master/image-20210312134417230.png)

## BiGan

![image-20210312133943456](https://raw.githubusercontent.com/BeBubbled/PicGoImages-WorkSpace/master/image-20210312133943456.png)

![image-20210312134358537](https://raw.githubusercontent.com/BeBubbled/PicGoImages-WorkSpace/master/image-20210312134358537.png)

![image-20210312140523483](https://raw.githubusercontent.com/BeBubbled/PicGoImages-WorkSpace/master/image-20210312140523483.png)

让encoder与decode越相似越好



Bigan得到的auto encoder与一半的auto-encoder特性不一样



### Triple Gan

### Domain-adversarial training

![image-20210314000857786](https://raw.githubusercontent.com/BeBubbled/PicGoImages-WorkSpace/master/image-20210314000857786.png)











## Presentation

1. 自我介绍, 标题页

2. 第二页!!

   the thoughts of our algorithm is  that you input the question and document and it return subspan of the documents as the answer.

   Here is an example of the model

2. As we choose bert as baseline, we inputs tokens and bert return the answer's start and end index.

   So  let's assume that our question has n tokens and documents have m tokens, we inputs their concatenation into bert.

   

   最后一页!!!

   

   The bert will return vector C​ which has the same dismension as inputs but we only need the document part, because the answer is just the sub span of input document.

   

   As we need to find the start and end index, we prepare two linear classifiers

   We use them take dot product with C​'s document part and apply softmax to get p{start,i }and p{end,j} probabilty distribution

   

   Assume the correct start and end index named "I" and "J", then we could get the p_I from p_startdistri, p_J from p_end_j distri ,then the loss fucntion for this sample is 
   $$
   \text{should have correspond equation}
   $$
   Next, we apply backpropagation.

   we reapeat these processes until finish all training part.

   Beyond the basic QA system, our goal is to make it robust on unseen domain, to achieve such goal,we will try some existing strategies which will be discussed in related work and see if we can make any improvements.

CLS: the key word of the classifier

SEP: separate question tokens and document tokens



## Proposal Feedback

