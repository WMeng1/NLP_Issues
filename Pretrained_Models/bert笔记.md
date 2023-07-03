# 1  算法概述
> >https://www.cnblogs.com/xlturing/p/10824400.html#transformer
##### BERT算法出自谷歌于2018年发表的《[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)》，该算法刷新了自然语言处理11项纪录，也被称为2018年最强自然语言处理模型。
&emsp;&emsp;首先对该论文所涉及到的一些知识点进行介绍，首先回顾一下Seq2seq。
## 1.1  Seq2seq
&emsp;&emsp;Seq2seq算法源于谷歌于2014年发表的论文《[Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)》，关于Seq2seq的应用场景，如机器翻译（MT）的目标就是对于输入的一组序列得到翻译后的一组新序列，Seq2seq利用了LSTM对输入序列分别进行编码解码(encoder-decoder)操作，并且该算法于英文和法文的相互翻译当中取得了很棒的效果。下面我们对Seq2seq进行更加细致的介绍。 
<center>

![最基础的Seq2seq模型](https://img-blog.csdn.net/20180812204236184?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMyMjQxMTg5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
</center>
&emsp;&emsp;Seq2seq即序列到序列的模型，目标将输入序列

```math
p(y_1,y_2,\dots,y_{T^’}|x_1, x_2, \dots, x_T)
```
```math
    h_t=encoder(h_{t-1}, x_t)
```
```math
    s_t=decoder(s_{t-1}, y_{t-1}, c_t)
```
其中`$c_t$`为计算得到的语义信息：
```math
    c_t=f(s_{t-1}, h_t)
```
条件概率通过下面的公式进行计算：
```math
    p(y_1,y_2,\dots,y_{T^’}|x_1, x_2, \dots, x_T) = \sum_{t=1}^{T'}p(y_t|y_{<t}, x)
```
其中
```math
    p(y_t|y_{<t},x) = softmax(f(s_t))
```
`$f(\cdot)$`为全连接层，在机器翻译场景下，softmax函数用于计算词典中每个词的概率。  
&emsp;&emsp;目前encoder-decoder的这种框架在深度学习模型中式十分常用的，很多实际的NLP问题可以抽象到这个框架上来进行建模，比如NMT（机器翻译）、TTS（语音合成）、ASR（语音识别）等。下面我们对Attention机制进行介绍。

![](https://img-blog.csdnimg.cn/20210119153414628.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzYxNDg3NA==,size_16,color_FFFFFF,t_70#pic_center)  

我们把Seq2Seq的模型结构用上图表示，也从中引出Seq2Seq的缺点：即对`$y_t$`进行预测时，只能使用前一步预测的结果`$y_{t-1}$`以及当前点的输入`$s_t$`，再往前的数据就得不到了。为了解决这个问题，就有了Attention结构的产生。
</center>

## 1.2  Attention  
&emsp;&emsp;在《[Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf)》这篇论文中，Bahdanau等人在原有Seq2Seq框架的基础上新增了attention（注意力机制），其基于的主要先验感知是：基本的encoder-decoder框架在达到瓶颈后，作者希望通过将源句中与目标句对应部分的关联程度，由模型来自动查找（soft search），而不是硬性的切分。换句好理解的，源句中的每个词在翻译到目标词汇时起到的作用是不一样的，作者希望模型能够学出这种关联强度。举个例子：  
&emsp;&emsp;我/想/买/苹果/手机  
&emsp;&emsp;I want to buy iPhone.  
&emsp;&emsp;比如在翻译苹果这个词的step时，这时候会有一个明显的歧义，是吃的苹果呢，还是商标品牌呢，那么源句中苹果的上下文对翻译时产生的影响是不一样的，我/想/买，仅凭这三个词我们并不能解决这个苹果的歧义问题，但是看到苹果后面跟着手机这个词，歧义马上消除，苹果指品牌。那么在这个过程中，很明显手机这个词所起到的作用是远远大于我/想/买三个词的。Attention模型可以用于回顾模型的过去状态。  
&emsp;&emsp;在Seq2seq模型的基础上加入Attention之后的encoder-decoder过程如下：
<center>

![](https://img-blog.csdn.net/20180813222008526?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMyMjQxMTg5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)  
![](https://img-blog.csdn.net/20180813222042700?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMyMjQxMTg5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)  
</center>

&emsp;&emsp;加入Attention机制后的编解码具体流程就变成了上图所述，下面介绍Attention机制的具体实现方式： 
<center>

![Attention](https://img2018.cnblogs.com/blog/524764/201903/524764-20190302185014090-123375859.png)  
</center>  

&emsp;&emsp;可以看出该模型主要基于的还是encoder-decoder框架，但与Seq2seq不同的是，该模型不是将final-state直接作为输入给到解码过程，而是在每个step解码的时候考虑encoder过程中对于每一个词的编码，通过加权的方式将其作为解码的输入参数，下面介绍模型的具体定义：
- 输入参数：`$X=(x_0,x_1,x_2,\dots,x_T)$`
- 输出结果：`$Y=(y_0,y_1,y_2,\dots,y_T)$`
- encoder层隐层状态：`$H=(h_0,h_1,h_2,\dots,h_T)$`
- decoder层上一step隐层状态：`$s_{t-1}$`
- decoder层上一step的输出：`$y_{t-1}$`
- 当前decoder step的上下文：`$c_t$` 
 
整个Attention模型的过程为：  
&emsp;1、根据输入X，利用双向LSTM得到两组隐向量(附双向LSTM模型图)，直接concat后得到最终的H；
<center>

![](https://pic2.zhimg.com/v2-485b572266dddba756e457decf276379_b.jpg) 
</center>  

&emsp;&emsp;2、根据encoder层得到H，当前时刻为t时，此时上下文`$c_t$`由下式可得：  
```math
    c_t = attention(s_{t-1}, h_t)= \sum_{j=1}^{T_x}\alpha_{tj}h_j
```
其中`$\alpha_{tj}$`为权重；h是当前t时encoder编码的所有隐变量；i为decoder层的时刻step，j代表encoder层的时刻step。  
&emsp;&emsp;权重`$\alpha_{tj}$`的计算方式如下：
```math
    \alpha_{tj} = \frac{exp(e_{tj})}{\sum_{p=1}^{T_x}exp(e_{tp})}
```
其中`$e_{tj} = a(s_{t-1}, h_j)$`。下面对该式进行简单介绍：本质上`$\alpha$`式就是一个softmax，其作用就是用一个概率和为1的分布，来刻画encoder不同时刻得到的隐变量的权重，从而衡量其对decoder的重要性；而`$e_{ij}$`其实是通过a，一种线性表达式来度量`$s_{i−1},h_j$`之间的相关程度。细化下a的计算方式：
1. 对`$s_{t−1}$`做线性映射，得到的向量起名为query，记为`$q_t$`;
2. 同样对`$h_{j}$`做线性映射，得到的向量起名为key，记为`$k_j$`;
3. 最终`$e_{tj}=a(s_{t-1}, h_j) = v^Ttanh(q_t+k_j)$`，其中`$q_t, k_j$`均为d维向量，v为d*1的系数矩阵。  

&emsp;&emsp;上面在做alignment时采用了query和key相加的方式，也即Bahdanau attention（以作者名字命名）；还有另一种常见的Luong attention，核心就是`$e_{ij}`的计算方式不同，换成相乘即乘性attention，具体计算公式如下：
```math
    e_{ij}=score(s_{i-1},h_j)=\left\{\begin{matrix} s_{i-1}^Th_j & dot \\ s_{i-1}^TW_ah_j & general \\ v_a^Ttanh(W_a[s_{i-1};h_j]) & concat \end{matrix}\right.
```
&emsp;&emsp;乘性attention常用的即general一项，dot两者直接相乘相当于是general的特例情况，即以进行过维度统一。还有一些其他变种，大家可以找找相关论文根据具体任务和x、y的向量空间具体分析。

&emsp;&emsp;如果对数学公式不是很敏感现在还是一头雾水的话，那么把上边的叙述用比较通俗的语言归纳一下就是：**attention机制是对于原始的Seq2seq的基础上将语义函数`$c(\cdot)$`进行了优化，原本的`$c(\cdot)$`对于任意输入向量是进行着相同的变换，这就导致了得到的语义向量不能完全表示输入序列的信息，并且先输入的内容所得到的信息会被后输入的内容信息稀释，那么解码过程的准确率就有待商榷了。在加入了attention机制之后，模型会在不同时刻迭代生成不同的权重系数`$\alpha_{tj}$`,使得语义函数`$c(\cdot)$`变成了对编码过程隐向量的加权平均过程**
```math
    c_t = \sum_{j=1}^{T_x}\alpha_{tj}h_j
```
&emsp;&emsp;那么在解码过程当中，若t时刻隐层的输出是`$s_t$`。那么t时刻的输出就是相关于`$s_i,y_{i-1},c_i$`的函数，此时所需输出的条件概率
```math
    P(y_t|y_1,\dots,y_{t-1},x) = f(y_{t-1}, s_t, c_t)
```  
其中
```math
    s_t = g(s_{t-1}, y_{t-1}, c_t)
```
&emsp;&emsp;**也就是说，attention在原有的encoder-decoder框架的基础上，通过query和key的相关程度，之后通过类似softmax的归一化操作得到权重，利用权重、隐状态进行加权和得到上下文context向量，作为decoder层的输入。直观上attention使得与decoder层的状态s相关度越大的h，权重会越高，从而在context中占比更多，对y的输出影响更大，这也是直观的先验解释，为什么attention会这么有效。**

&emsp;&emsp;注意：这里记住query、key和value的由来，没啥为什么，就起了个别名，在transfomer中它们还将发挥更多的作用。

## 1.3  Transformer
&emsp;&emsp;在聊完Attention机制后，我们来聊一聊Transformer框架，该框架是谷歌于2017年发表的《[Attetion is all you need](https://arxiv.org/pdf/1706.03762.pdf)》。传统意义上的attention框架是构建在RNN或者CNN的基础上，而在该篇论文中，整个框架完全是基于attention的思路建立的，所以也就起名为，该模型整个实现过程如下图所示：
<center>

![](https://img2018.cnblogs.com/blog/524764/201903/524764-20190326105900624-388721810.png)  
</center>

&emsp;&emsp;上图左边是encoder过程，右边是decoder过程，可以看出模型整体还是遵循encoder-decoder框架，并且从图中可以看出编码和解码过程都有`$N_x$`标记，这代表无论是encoder过程还是decoder过程均叠加了`$N_x$`个相同的layer，原文中`$N_x=6$`，但每层之间均通过残差进行连接。在对模型进行拆解叙述之前对一些需要了解的定义进行简单介绍。下边我们把该模型进行拆解介绍。  
### 1.3.1  encoder过程
输入：embedding得到的词向量。

&emsp;&emsp;但从模型图中可以看出，词向量加入了positional embedding，即给位置1，2，3，4...n等编码（也用一个embedding表示）。然后在编码的时候可以使用正弦和余弦函数，使得位置编码具有周期性，并且有很好的表示相对位置的关系的特性（对于任意的偏移量k，PE[pos+k]可以由PE[pos]表示）：
<center>

![](https://pic3.zhimg.com/80/v2-50c42bbb4625c3f2cc8b4760168cd6ca_720w.jpg)
</center>

整个encoder过程包括两个子层，分别为Multi-head self-attention以及残差连接和Layer-Normalization(LN)层，下面分别对这两个子层进行介绍。
#### 1.3.1.1  Self-attention  
&emsp;&emsp;传统的Attention机制在一般任务的Encoder-Decoder model中，输入Source和输出Target内容是不一样的，比如对于英-中机器翻译来说，Source是英文句子，Target是对应的翻译出的中文句子，Attention机制发生在Target的元素Query和Source中的所有元素之间。简单的讲就是Attention机制中的权重的计算需要Target来参与的，即在Encoder-Decoder model中Attention权值的计算不仅需要Encoder中的隐状态而且还需要Decoder 中的隐状态。  
&emsp;&emsp;而Self-Attention顾名思义，指的不是Target和Source之间的Attention机制，而是Source内部元素之间或者Target内部元素之间发生的Attention机制，self就是向量本身，也可以理解为Target=Source这种特殊情况下的注意力计算机制。  
&emsp;&emsp;简单回顾下上文attention时提到的query、key和value的概念（笔者专门让大家注意），通过对s做映射得到query，对h做映射得到key，而h也即是value，之后通过query和key计算关联度之后，利用加权和的方式获得最终的编码向量。对于经过embedding后的文本向量，通过三个不同的权重向量，分别映射成为query、key以及value，之后通过下图的步骤得到最终的编码向量：
<center>

![](https://img2018.cnblogs.com/blog/524764/201904/524764-20190411201744644-169435637.png)  
</center>

&emsp;&emsp;下面我们详细介绍一下self-attention过程的整个计算过程。  
&emsp;&emsp;**计算自注意力的第一步**是从编码器的每个输入向量中创建三个向量(在本例中，是每个单词的嵌入)。因此，对于每个单词，我们创建一个Query向量、一个Key向量和一个Value向量。这些向量是通过将词嵌入与3个训练后的矩阵相乘得到的。  
&emsp;&emsp;注意这些新的向量在维数上比嵌入向量小。它们的维数为64，而词嵌入和编码器的输入/输出向量的维数为512。把向量维度降低仅仅是一种处于架构考虑的选择，从而使多头注意力（multi-headed attention）计算保持维度上的固定，关于multi-headed attention下面会具体介绍。
<center>

![](https://img2018.cnblogs.com/blog/1135245/201812/1135245-20181223142732882-1162946030.png)
</center>

&emsp;&emsp;上图中输入了两个词，分别是Thinking和Machines，在进行Embedding之后与训练得到的权重矩阵相乘分别得到对应的KQV向量。  
&emsp;&emsp;**第二步是计算注意力得分**假设我们要计算本例中第一个单词“Thinking”的自注意力得分。我们需要对输入句子中的每个单词进行打分。这个分数决定了我们在编码某个位置上的单词时，对其他单词的关注程度。

&emsp;&emsp;这个得分是通过计算query向量与各个单词的key向量的点积得到的。所以如果我们要计算位置1上的单词的自注意力得分，那么第一个分数就是`$q_1$`和`$k_1$`的点积。第二个分数是`$q_1$`和`$k_2$`的点积。
<center>

![](https://img2018.cnblogs.com/blog/1135245/201812/1135245-20181223142744510-441431392.png)
</center>

&emsp;&emsp;第三步和第四步是将得分除以8(key向量的维数（64）的平方根，是默认值。这能让梯度更新的过程更加稳定)，然后将结果进行softmax操作。Softmax将分数标准化，使它们都是正数，加起来等于1。
<center>

![](https://img2018.cnblogs.com/blog/1135245/201812/1135245-20181223142754395-1680117835.png)
</center>

&emsp;&emsp;softmax的结果决定了每个单词在这个位置（1）上的相关程度（译者注：相当于权重）。显然，1这个位置的单词会有最高的softmax得分。

&emsp;&emsp;第五步是将每个value向量乘以softmax的结果（为求和做准备）。这里的思想是尽量保持我们想要关注的单词的value值不变，而掩盖掉那些不相关的单词(例如将它们乘以很小的数字)。

&emsp;&emsp;第六步是将带权重的各个value向量加起来。就此，产生在这个位置上(第一个单词)的self-attention层的输出。
<center>

![](https://img2018.cnblogs.com/blog/1135245/201812/1135245-20181223142805335-834227699.png)
</center>

&emsp;&emsp;这就是自注意力的计算过程。得到的输出向量可以送入前馈神经网络。但是在实际的实现中，为了更快的处理，这种计算是以矩阵形式进行的，下面把上述过程用矩阵形式进行表示。
##### 自注意力的矩阵计算

&emsp;&emsp;第一步是计算Query矩阵、Key矩阵和Value矩阵。我们把词嵌入矩阵X乘以训练得到的的权重矩阵（`$W^Q$`, `$W^K$`, `$W^V$`），输入向量X的每一行代表一个词向量。
<center>

![](https://img2018.cnblogs.com/blog/1135245/201812/1135245-20181223142818219-64125948.png)
</center>

&emsp;&emsp;最后，由于我们处理的是矩阵，我们可以将第二部到第六步合并为一个公式中计算自注意力层的输出。
<center>

![](https://img2018.cnblogs.com/blog/1135245/201812/1135245-20181223142826496-748950638.png)
</center>

即
```math
Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V
```

#### 1.3.1.2  Multi-Head Attention
&emsp;&emsp;论文中在self-attention的基础上，有一个Multi-Head Attention，即多头注意力机制，就是字面意思，将文本embedding通过多个self-attention得到不同的编码结果，然后拼接起来进入下一层，如下图：
<center>

![](https://img2018.cnblogs.com/blog/524764/201904/524764-20190411202806002-1999017291.png)
</center>

具体计算过程可以用下图进行说明
<center>

![](https://img2018.cnblogs.com/blog/1135245/201812/1135245-20181223142908889-508162551.png)
</center>

#### 1.3.1.3  Layer-Normalization(LN)

&emsp;&emsp;Normalization是一个常见的标准化消除向量之间过大的差异的方式，常见的两种方式为Batch-Normalization(BN)和Layer-Normalization(LN),这两种方式的区别可以用下图说明：
<center>

![](https://pic2.zhimg.com/80/v2-2d3f611065de9070fced5b5abd4d032d_720w.jpg)
</center>

在进行了Multi-Head Self-Attention之后，输出的是残差连接和LN之后的向量，即
```math
Output = LN (x+sublayer(x))
```
在FFN的部分，模型使用了ReLU作为激活函数，公式如下：
```math
FFN(x)=max(0,xW_1+b_1)W_2+b_2
```
每一层会使用不同的参数，可以将其理解为两个Kernel size为1的卷积
### 1.3.2  decoder过程
&emsp;&emsp;解码过程与编码过程的子层大致相似，多了一个attention层，不过要注意的是这两个attention层，首先进行的是self-attention层，并且在该层加入了Sequence mask机制，下面会对该机制进行介绍，而第二个attention层是cross-attention层，其K、V来自Encoder层，结合前一个Q，进入该子层，具体过程如下：
<center>

![](https://pic4.zhimg.com/80/v2-cca6e1f0dd02f08cc554d731362a08af_720w.jpg)
</center>

并且需要注意的是：
- 第i个位置的Encoder输出简单变化作为K、V与i-1位置的Decoder第一个子层的输出作为Q；
- Encoder在编码时可以将输入序列作为一个整体，以矩阵的形式一次性编码，但Decoder时当前位置的输出需要依赖上一个位置的输出，所以是按顺序解码的；
- 在Decoder的第一个子层self attention做了mask，保证当前位置i的预测只依赖小于i位置的输出，避免利用未来的信息；
- Decoder层在预测到类似的结束符时便会停止预测。

&emsp;&emsp;最后，Decoder经过两个attention子层和一个FFN子层后的输出，其解码向量经过一个线性映射WX+b，将其映射到整个字典空间，然后经过softmax规整成字典词汇的概率分布，即每个词的分数，之后选取分数最高的词作为输出即可。
#### 注：Sequence mask： 
&emsp;&emsp;在Transformer中，sequence mask只用在Decoder中，它的作用是使得Decoder在进行解码时不能看到当前时刻之后的的信息。也就是说，对于一个输入序列，当我们要对t时刻进行解码时，我们只能考虑timestamp<t时刻的信息。  
&emsp;具体做法，是产生一个下三角矩阵，这个矩阵的上三角的值全为0，下三角的值全为输入矩阵对应位置的值，这样就实现了在每个时刻对未来信息的掩盖。
<center>

![](https://spaces.ac.cn/usr/uploads/2019/09/1936593842.png)
</center>

# 2  BERT
&emsp;&emsp;BERT算法来源于2018年google AI组发表的一篇文章《[BERT: Pre-training of Deep Bidirectional Transformers for Language](https://arxiv.org/pdf/1810.04805.pdf)》从名字我们能看出该模型两个核心特质：依赖于Transformer以及双向，该算法一经推出就在领域内引起了极大关注，作为Word2Vec的替代者，其在NLP领域的11个方向大幅刷新了精度，可以说是近年来自残差网络最优突破性的一项技术，下面我们就对该算法进行详细介绍。
<center>

![](https://img-blog.csdnimg.cn/20190101145605654.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dhbmdqaWFuMTIwNA==,size_16,color_FFFFFF,t_70)
</center>

&emsp;&emsp;论文在介绍BERT模型结构时与另外两个Pre-trained模型结构进行了比较，从图中可以看出三者的区别点：
- ELMo使用的是LSTM作为基本模型，而BERT和OpenAI GPT都是使用Transformer作为基本模型；
- BERT算法相比于OpenAI GPT模型而言，其使用了双向Transformer；
- 而ELMo与BERT同为双向，ELMo使用的是LSTM，而BERT使用的是Transformer
- 三者的目标函数有所不同：

OpenAI GPT：
```math
    P(\omega_i|\omega_1,\dots,\omega_{i-1})
```
ELMo:
```math
    P(\omega_i|\omega_1,\dots,\omega_{i-1})和P(\omega_i|\omega_{i+1},\dots,\omega_n)
```
BERT:
```math
    P(\omega_i|\omega_1,\dots,\omega_{i-1},\omega_{i+1},\dots,\omega_n)
```
&emsp;&emsp;OpenAI GPT的目标函数与传统目标函数相同；而ELMo以双向LSTM为基础，其目标函数分成了两个，每一个词都会得到left-to-right和right-to-left两种表示，然后将两者concat在一起作为该词的表示；只有BERT可以同时考虑左右两边的上下文信息，这一点很大程度上得益于Transformer中的self-attention解决了传统RNN中的长距离依赖问题，将词之间距离常量化，这允许其进行并行计算的同时也加速了模型的训练，并且BERT也通过其它一些方式使得直接给出词的上下文表示变得可行。Google团队设计了两个BERT模型，分别是`$BERT_{BASE}$`和 `$BERT_{LARGE}$`，
- `$BERT_{BASE}$`:L=12,H=768,A=12,模型大小110M
- `$BERT_{LARGE}$`:L=24,H=1024,A=16,模型大小340M

其中L：Transformer层数，H：隐单元个数，A：self-attention heads数量

&emsp;&emsp;为了更方便理解，给出BERT的整体架构如下图所示：
<center>

![](https://img2018.cnblogs.com/blog/1102791/201911/1102791-20191103131901879-1808212138.jpg)
</center>

&emsp;&emsp;BERT中只使用了Transformer的encoder而没有使用decoder，个人觉得可能是因为BERT作为一个预训练模型，主要任务是学习到语义关系给下游任务做铺垫，不需要解码到具体任务当中，而多个Transformer Encoder一层一层地堆叠起来，就组装成了BERT了。下面我们对其中BERT的一些核心结构进行介绍。
## 2.1  模型输入
&emsp;&emsp;BERT对于词的处理仍然是Embedding，但Embedding过程有所不同，如下图：
<center>

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy92STluWWU5NGZzRTBwaWNnelNtWWhCbnpaREo2OU9xTGw4bkVEMnNFQ1pUVXp3ZWVwWk1icTNucmxEMGFWWWtGZnNrOWxpY2liVmdWc1JZOUo5MW9qdzl2dy82NDA?x-oss-process=image/format,png)
</center>

&emsp;&emsp;BERT的输入的编码向量（长度是512）是3个嵌入特征的单位和，如上图，这三个词嵌入分别是：
- Token Embeddings：是词向量，第一个单词是CLS标志，代表classification，可以用于下游的分类任务。可以理解为：与文本中已有的其它字/词相比，这个无明显语义信息的符号会更公平地融入文本中各个字/词的语义信息。
- Segment Embeddings：将句子分为两段，用来区别两种句子，例如B是否是A的下文（对话场景，问答场景等）。对于句子对，第一个句子的特征值是0，第二个句子的特征值是1。
- 位置嵌入是指将单词的位置信息编码成特征向量，位置嵌入是向模型中引入单词位置关系的至关重要的一环。这里的位置向量和之前文章中的Transformer不一样，不是三角函数变换而是学习出来的。

## 2.2  预训练任务
&emsp;&emsp;BERT是一个多任务模型，它的任务是由两个自监督任务组成，即MLM和NSP。

### 2.2.1  Masked Language Model(MLM)
&emsp;&emsp;Masked LM的任务像是英语考试中常见的完形填空，可以将其描述为：给定一句话，随机抹去这句话中的一个或几个词，要求根据剩余词汇预测被抹去的几个词分别是什么，如下图所示：
<center>

![](https://img2018.cnblogs.com/blog/1102791/201911/1102791-20191103141644707-1946450237.jpg)
</center>

&emsp;&emsp;在BERT的实验中，15%的WordPiece Token会被随机Mask掉。在训练模型时，一个句子会被多次喂到模型中用于参数学习，但是Google并没有在每次都mask掉这些单词，而是在确定要Mask掉的单词之后，80%的时候会直接替换为[Mask]，10%的时候将其替换为其它任意单词，10%的时候会保留原始Token。
- 80%：my dog is hairy -> my dog is [mask]
- 10%：my dog is hairy -> my dog is apple
- 10%：my dog is hairy -> my dog is hairy

&emsp;&emsp;这么做的原因是如果句子中的某个Token100%都会被mask掉，那么在fine-tuning的时候模型就会有一些没有见过的单词。加入随机Token的原因是因为Transformer要保持对每个输入token的分布式表征，否则模型就会记住这个[mask]是token ’hairy‘。至于单词带来的负面影响，因为一个单词被随机替换掉的概率只有15%*10% =1.5%，这个负面影响其实是可以忽略不计的。

&emsp;&emsp;另外文章指出每次只预测15%的单词，因此模型收敛的比较慢。
### 2.2.2  Next Sentence Prediction(NSP)
任务描述为：给定一篇文章中的两句话，判断第二句话在文本中是否紧跟在第一句话之后，如下图所示：
<center>

![](https://img2018.cnblogs.com/blog/1102791/201911/1102791-20191103141948991-841345073.jpg)
</center>

&emsp;&emsp;Next Sentence Prediction（NSP）的任务是判断句子B是否是句子A的下文。如果是的话输出'IsNext'，否则输出'NotNext'。训练数据的生成方式是从平行语料中随机抽取的连续两句话，其中50%保留抽取的两句话，它们符合IsNext关系，另外50%的第二句话是随机从预料中提取的，它们的关系是NotNext的。这个关系保存在[CLS]符号中。
## 2.3  模型输出
<center>
    
![](https://pic3.zhimg.com/80/v2-d0a896547178320eb21a92550c48c66a_720w.jpg)
</center>
模型的每一个输入都对应这一个输出，根据不同的任务我们可以选择不同的输出，主要有两类输出：

- pooler output：对应的是[CLS]的输出，该符号对应的输出向量作为整篇文本的语义表示，用于文本分类，如下图所示。可以理解为：与文本中已有的其它字/词相比，这个无明显语义信息的符号会更“公平”地融合文本中各个字/词的语义信息。
- sequence output：对应的是所有其他的输入字的最后输出，可以用于问答系统、语句匹配、分词等应用场景，下面就对BERT在具体NLP任务上的fine-tune进行介绍。

## 2.4  具体NLP任务上的fine-tune
&emsp;&emsp;在海量单预料上训练完BERT之后，便可以将其应用到NLP的各个任务中了。

&emsp;&emsp;对于其它任务来说，我们也可以根据BERT的输出信息作出对应的预测。图5展示了BERT在11个不同任务中的模型，它们只需要在BERT的基础上再添加一个输出层便可以完成对特定任务的微调。这些任务类似于我们做过的文科试卷，其中有选择题，简答题等等。图5中其中Tok表示不同的Token，`$E$`表示嵌入向量，`$T_i$`表示第 `$i$`个Token在经过BERT处理之后得到的特征向量。
<center>

![](https://pic2.zhimg.com/80/v2-f576d9d19c9dcac1c6ee6ea28ea7a2d9_720w.jpg)
</center>

微调的任务可以解决涵盖下面4种下游任务：

**（a）基于句子对的分类任务：**
- MNLI：给定一个前提 (Premise) ，根据这个前提去推断假设 (Hypothesis) 与前提的关系。该任务的关系分为三种，蕴含关系 (Entailment)、矛盾关系 (Contradiction) 以及中立关系 (Neutral)。所以这个问题本质上是一个分类问题，我们需要做的是去发掘前提和假设这两个句子对之间的交互信息。
- QQP：基于Quora，判断 Quora 上的两个问题句是否表示的是一样的意思。
- QNLI：用于判断文本是否包含问题的答案，类似于我们做阅读理解定位问题所在的段落。
- STS-B：预测两个句子的相似性，包括5个级别。
- MRPC：也是判断两个句子是否是等价的。
- RTE：类似于MNLI，但是只是对蕴含关系的二分类判断，而且数据集更小。
- SWAG：从四个句子中选择为可能为前句下文的那个。

**（b）基于单个句子的分类任务**
- SST-2：电影评价的情感分析。
- CoLA：句子语义判断，是否是可接受的（Acceptable）。
对于GLUE数据集的分类任务（MNLI，QQP，QNLI，SST-B，MRPC，RTE，SST-2，CoLA），BERT的微调方法是根据[CLS]标志生成一组特征向量 `$C$`，并通过一层全连接进行微调。损失函数根据任务类型自行设计，例如多分类的softmax或者二分类的sigmoid。

&emsp;&emsp;SWAG的微调方法与GLUE数据集类似，只不过其输出是四个可能选项的softmax：
```math
    P_i=\frac{e^{V·C_i}}{\sum_{j=1}^{4}e^{V·C_i}}
```

**（c）问答任务**

&emsp;&emsp;SQuAD v1.1：给定一个句子（通常是一个问题）和一段描述文本，输出这个问题的答案，类似于做阅读理解的简答题。如上图中(c)图表示的，SQuAD的输入是问题和描述文本的句子对。输出是特征向量，通过在描述文本上接一层激活函数为softmax的全连接来获得输出文本的条件概率，全连接的输出节点个数是语料中Token的个数。
```math
    P_i=\frac{e^{S·C_i}}{\sum_{j}e^{S·C_i}}
```

**（d）命名实体识别**

CoNLL-2003 NER：判断一个句子中的单词是不是Person，Organization，Location，Miscellaneous或者other（无命名实体）。微调CoNLL-2003 NER时将整个句子作为输入，在每个时间片输出一个概率，并通过softmax得到这个Token的实体类别。
## 2.5 模型的评价
### 优点：
&emsp;&emsp;BERT是截止至2018年10月的最新的的state of the art模型，通过预训练和精调可以解决11项NLP的任务。使用的是Transformer，相对于rnn而言更加高效、能捕捉更长距离的依赖。与之前的预训练模型相比，它捕捉到的是真正意义上的bidirectional context信息;
### 缺点：
&emsp;&emsp;作者在文中主要提到的就是MLM预训练时的mask问题：

&emsp;&emsp;1)[MASK]标记在实际预测中不会出现，训练时用过多[MASK]影响模型表现;

&emsp;&emsp;2)每个batch只有15%的token被预测，所以BERT收敛得比left-to-right模型要慢（它们会预测每个token）

# 3  实战中文短文本分类
###### > https://kexue.fm/archives/6736  这可能是Bert最简单的打开方式

分词过程，keras_bert中自带的tokenizer对输入的text进行编码会返回两个值.  
&emsp;&emsp;indices：分词对应下标  
&emsp;&emsp;segment：段落对应下标
```
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 10:20:49 2020

@author: wangm-s
"""

'''
提示：受GPU性能的影响，只能运行基础版的bert预训练模型，若出现OOM 适当调整batch_size, maxlen，如果使用cup运行为非常忙
我用的是numpy==1.16.4其他版本可能会有提示
'''

import pandas as pd
import codecs, gc
import numpy as np
import os
from sklearn.model_selection import KFold
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.metrics import top_k_categorical_accuracy
from keras.layers import *
from keras.callbacks import *
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn import metrics
# 读取训练集和测试集
from sklearn.model_selection import train_test_split

# 参数配置
maxlen = 20  # 设置序列长度为100，要保证序列长度不超过512
Batch_size = 64  # 批量运行的个数
Epoch = 5  # 迭代次数


def get_train_test_data():
    train_df = pd.read_excel(r'F:/workspace/Python/材料分类/Category_classify/训练数据/分类器训练数据-市场价.xlsx', encoding='utf-8').astype(str)
    train_df['content'] = train_df['name'] + train_df['spec']

    DATA_LIST = []
    one_label = []
    label_id = {}
    label = 0
    train_data = train_df[['content', 'category2_id']].values.tolist()
    for data_message in train_data:
        if data_message[1] not in label_id.keys():
            label_id[data_message[1]] = label
            one_label.append(label)
            label = label + 1
        else:
            one_label.append(label_id[data_message[1]])

    train_df['category2id'] = np.array(one_label)

    for data_row in train_df.iloc[:].itertuples():
        DATA_LIST.append((data_row.content, to_categorical(data_row.category2id, 33)))

    DATA_LIST = np.array(DATA_LIST[:40000])

    DATA_LIST_TEST = np.array(DATA_LIST[40001:45000])

    DATA_LIST_VAL = np.array(DATA_LIST[-800:])

    data_test = DATA_LIST_TEST
    X_valid = DATA_LIST_VAL
    X_train = DATA_LIST
    return X_train, X_valid, data_test


# 预训练好的模型 roberta_wwm_ext_large
# config_path     = r'roberta_wwm_ext_large\bert_config.json' # 加载配置文件
# checkpoint_path = r'roberta_wwm_ext_large\bert_model.ckpt'
# dict_path       = r'roberta_wwm_ext_large\vocab.txt'

# 预训练好的模型 bert base
config_path = r'chinese_L-12_H-768_A-12\bert_config.json'  # 加载配置文件
checkpoint_path = r'chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = r'chinese_L-12_H-768_A-12\vocab.txt'


# =============================================================================
# 也可以用 keras_bert 中的 load_vocabulary() 函数
# 传入 vocab_path 即可
# from keras_bert import load_vocabulary
# token_dict = load_vocabulary(vocab_path)
# def get_token_dict():
# =============================================================================

def get_token_dict():
    """
    # 将词表中的字编号转换为字典
    :return: 返回自编码字典
    """
    token_dict = {}
    with codecs.open(dict_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    return token_dict


# 重写tokenizer
class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # 用[unused1]来表示空格类字符
            else:
                R.append('[UNK]')  # 不在列表的字符用[UNK]表示   UNK是unknown的意思
        return R


# 获取新的tokenizer
tokenizer = OurTokenizer(get_token_dict())


def seq_padding(X, padding=0):
    """
    :param X: 文本列表
    :param padding: 填充为0
    :return: 让每条文本的长度相同，用0填充
    """
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X])
    '对数组进行拼接，默认axis=0(按行方向)'


class data_generator:
    """
    data_generator只是一种为了节约内存的数据方式
    """

    def __init__(self, data, batch_size=Batch_size, shuffle=True):
        """
        :param data: 训练的文本列表
        :param batch_size:  每次训练的个数
        :param shuffle: 文本是否打乱
        """
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            if self.shuffle:
                np.random.shuffle(idxs)
            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:maxlen]
                '取出文本部分的maxlen长度的字符串，最长为maxlen，大于maxlen则截取，小于maxlen则取原字符串长度'
                x1, x2 = tokenizer.encode(first=text)
                'x1为字对应的索引，x2表示索引位置上的字属于第一句话还是第二句话，第一句话为0(包括CLS及SEP也会占用0)'
                y = d[1]
                '对应文本部分的标签值，标签为one-hot编码'
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    '如果数据量达到一个batch_size或遍历完所有数据'
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    '将未达到maxlen长度的文本用0填充到maxlen的长度'
                    yield [X1, X2], Y[:, 0, :]
                    'Y为padding后的标签的one-hot编码，shape为[batch_size,(类别数，为one-hot编码)],类型为数组'
                    'X1为padding后的字索引编码向量，shape为[batch_size,字符串的最大长度],类型为数组'
                    'X2为padding后的字位置编码向量，第一句则为0，第二句为1，shape与X1相同，类型为数组'
                    [X1, X2, Y] = [], [], []


def acc_top2(y_true, y_pred):
    """
    :param y_true: 真实值
    :param y_pred: 训练值
    :return: # 计算top-k正确率,当预测值的前k个值中存在目标类别即认为预测正确
    """
    return top_k_categorical_accuracy(y_true, y_pred, k=2)


# bert模型设置
def build_bert(nclass):
    """
    :param nclass: 文本分类种类
    :return: 构建的bert模型
    """
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)  # 加载预训练模型

    for l in bert_model.layers:
        l.trainable = True
    '开启fine-tune'
    # 构建模型
    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))
    'None表示batch_size大小？'
    x = bert_model([x1_in, x2_in])
    x = Lambda(lambda x: x[:, 0])(x)  # 取出[CLS]对应的向量用来做分类
    'CLS保存着句子的语义特征向量'
    p = Dense(nclass, activation='softmax')(x)

    model = Model([x1_in, x2_in], p)
    'Model(input= ,output= )'
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(1e-5),  # 用足够小的学习率
                  metrics=['accuracy', acc_top2])
    print(model.summary())
    return model


def run_kb():
    """
    训练模型
    :return: 验证预测集，测试预测集，训练号的模型
    """
    # 搭建模型参数
    print('正在加载模型，请耐心等待....')
    model = build_bert(33)  # 二分类模型
    'bulid_bert() 参数为输出类别'
    print('模型加载成功，开始训练....')
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=3)  # 早停法，防止过拟合
    plateau = ReduceLROnPlateau(monitor="val_accuracy", verbose=1, mode='max', factor=0.5,
                                patience=2)  # 当评价指标不在提升时，减少学习率
    # =============================================================================
    #     checkpoint = ModelCheckpoint(r'use_bert\bert_dump1.hdf5', monitor='val_acc', verbose=2,
    #                                  save_best_only=True, mode='max', save_weights_only=True)  # 保存最好的模型
    # =============================================================================
    # =============================================================================
    #     checkpoint = ModelCheckpoint('bert_dump1.hdf5', monitor='val_accuracy', verbose=2,
    #                              save_best_only=True, mode='max', save_weights_only=True)  # 保存最好的模型
    # =============================================================================
    # 获取数据并文本序列化
    X_train, X_valid, data_test = get_train_test_data()
    train_D = data_generator(X_train, shuffle=True)
    valid_D = data_generator(X_valid, shuffle=True)
    test_D = data_generator(data_test, shuffle=False)

    # 模型训练
    model.fit_generator(
        train_D.__iter__(),
        steps_per_epoch=len(train_D),
        epochs=Epoch,
        validation_data=valid_D.__iter__(),
        validation_steps=len(valid_D),
        callbacks=[early_stopping, plateau],
    )
    # 对验证集和测试集进行预测
    valid_D = data_generator(X_valid, shuffle=False)
    train_model_pred = model.predict_generator(valid_D.__iter__(), steps=len(valid_D), verbose=1)
    test_model_pred = model.predict_generator(test_D.__iter__(), steps=len(test_D), verbose=1)
    # 将预测概率值转化为类别值
    train_pred = [np.argmax(x) for x in train_model_pred]
    test_pred = [np.argmax(x) for x in test_model_pred]
    y_true = [np.argmax(x) for x in X_valid[:, 1]]

    # =============================================================================
    #     del model # 删除模型减少缓存
    #     gc.collect()  # 清理内存
    #     K.clear_session()  # clear_session就是清除一个session
    # =============================================================================

    return train_pred, test_pred, y_true, model, data_test


def bk_metrics(y_true, y_pred, type='metrics'):
    """
    :param y_true: 真实值
    :param y_pred: 预测值
    :param type: 预测种类
    :return: 评估指标
    """
    print(type, '...')
    print(metrics.confusion_matrix(y_true, y_pred))
    print('准确率：', metrics.accuracy_score(y_true, y_pred))
    print('类别精度：', metrics.precision_score(y_true, y_pred, average=None))  # 不求平均
    print('宏平均精度：', metrics.precision_score(y_true, y_pred, average='macro'))
    print('微平均召回率:', metrics.recall_score(y_true, y_pred, average='micro'))
    print('加权平均F1得分:', metrics.f1_score(y_true, y_pred, average='weighted'))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # 训练和预测
    train_pred, test_pred, y_true, model, data_test = run_kb()
    # 评估验证集
    bk_metrics(train_pred, y_true, type=' val metrics')
    # 评估测试集

    bk_metrics(test_pred, [np.argmax(x) for x in data_test[:, 1]], type=' test metrics')
    # 将模型保存
    model_path = '33类未清洗maxlen30_batchsize64_epoch5_数据随机改_keras=2.2.4.h5'
    model.save(model_path)

    # 模型加载
# =============================================================================
#     from keras_bert import get_custom_objects
#     from keras.models import load_model
#     custom_objects = get_custom_objects()
#     my_objects = {'acc_top2': acc_top2}
#     custom_objects.update(my_objects)
#     model = load_model(model_path, custom_objects=custom_objects)
# =============================================================================

# =============================================================================
#     text_ = ['离火车站非常近，很方便','不推荐','床铺不干净，服务态度很一般',
#              '服务实在太差!7月9日入住的,晚上11店打1168总机拜托第二天早上645分叫醒服务,房号:2319(便于查询与确认)。竟然没有做到!!这是我这个长期出差者第一次碰到的事情,因为10日早上830分有会议才提前一天住酒店的,还好这个酒店的床板硬得不能好好入睡,才"救"了我。特此评价。']
# =============================================================================
# =============================================================================
#     text_=['异径四通永享','异接弯头芬陀利华','Ｑ平管桥美亚','龙头座骏宝','高压管件恒通']
#     # 单独评估一个本来分类
#     for _ in text_:
#         DATA_text = []
#         DATA_text.append((_, to_categorical(0, 5)))
#         DATA_text = np.array(DATA_text)
#         text= data_generator(DATA_text, shuffle=False)
#         test_model_pred  = model.predict_generator(text.__iter__(), steps=len(text), verbose=1)
#
#         print('预测结果',test_model_pred)
#         print(np.argmax(test_model_pred))
# =============================================================================
# =============================================================================
#         if np.argmax(test_model_pred) == 0:
#             print("'{}'的情感分类是：消极".format(_))
#         else:
#             print("'{}'的情感分类是：积极".format(_))
# =============================================================================


# =============================================================================
#     del model # 删除模型减少缓存
#     gc.collect()  # 清理内存
#     K.clear_session()  # clear_session就是清除一个session
# =============================================================================


```
