# AlBERT论文阅读笔记
> > https://www.cnblogs.com/dogecheng/p/11617940.html

## 1 前言

## 1.1  摘要
&emsp;&emsp;预训练自然语言模型时，通常增加模型大小会提升模型在下游任务中的性能。但是经验表明，一味的增加模型大小可能会导致：
- GPU/TPU内存不足
- 模型训练时间更长
- 模型效果退化

&emsp;&emsp;这篇论文的目的旨在通过提出两种参数精简技术减少内存消耗，加快BERT的训练速度，并且通过引入自监督损失(self-supervised loss)对句子相关性(inter-sentence coherence)建模，验证了该损失函数能够提升在多句子输入的情况下的下游任务的性能。

&emsp;&emsp;本文提出的AlBERT算法在GLUE,RACE,SQuAD三个基准上都取得了新的SOTA(SOTA也就是state-of-the-art，算法（模型）的性能在当前是最优的)结果，且参数量远小于BERT-large。

## 1.2 介绍
&emsp;&emsp;过往的研究者们在诸多NLP任务上的实验已经表明，模型规模在取得SOTA结果上至关重要。在应用场景中通常是预训练一个大规模的模型，再对其进行蒸馏萃取出一个更小的模型。上述问题首要解决的便是
1. 内存限制。

&emsp;&emsp; 当下的各种SOTA模型动辄数亿甚至数十亿个参数，倘若要扩大模型规模，对内存的要求极高，这无疑是限制算法投入实际使用的一大问题。
2. 训练速度过慢。

&emsp;&emsp; 由于训练速度正比于模型参数，模型越大训练速度也就越慢。并且再进一步扩大模型大小，简单的增加隐藏层单元数的效果反而适得其反。由下图

<center>

![BERT增加隐层单元比较](https://img-blog.csdnimg.cn/20190929113900594.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2xqcDE5MTk=,size_16,color_FFFFFF,t_70)
</center>

&emsp;&emsp;可以看出当增加了BERT-large隐层一倍的单元数时，模型性能反而大幅度下降。

<center>

![BERT增加隐层单元比较](https://img-blog.csdnimg.cn/20190929113930111.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2xqcDE5MTk=,size_16,color_FFFFFF,t_70)
</center>

&emsp;&emsp;BERT-large 和 BERT-xlarge 的训练损失（左）和 dev mask 的 LM 准确率（右）。模型增大之后，其mask LM准确率降低了， 同时没有出现明显的过拟合迹象

&emsp;&emsp;为解决上述问题，已有先贤们做了一些研究，比如模型并行化、智能内存管理等。这些解决方案只解决了内存受限问题，而没有考虑模型通信开销过大和模型退化问题。为此，本文提出A Lite BERT(ALBERT)模型以解决上述三个问题。 该模型的参数量远远少于传统的 BERT 架构。

&emsp;&emsp;ALBERT主要提出了这两种参数精简技术，克服了未来扩展预训练模型面临的主要障碍。

**1. 嵌入层参数的因式分解。**

&emsp;&emsp;将一个大的词嵌入矩阵通过因式分解成两个小矩阵，从而将隐藏层的大小与词汇嵌入的大小分离开来。这种分离便于后续隐藏层单元数量的增加。

**2. 跨层参数共享**

&emsp;&emsp;这一技术可以避免参数量随着网络深度的增加而增加。

&emsp;&emsp;这两种技术都显著降低了BERT的参数量，同时不显著损害其性能，从而提升了参数效率。ALBERT的配置类似于 BERT-large，但参数量仅为后者的1/18，训练速度却是后者的1.7 倍。这些参数精简技术还可以充当某种形式的正则化，可以使训练更加稳定，且有利于泛化。

&emsp;&emsp;本文还引入了一个自监督损失函数

**3. 自监督损失函数**

&emsp;&emsp;为了进一步提升ALBERT的性能，ALBERT引入了一个自监督损失函数用于句子顺序预测（SOP，sentence-order prediction）。SOP主要聚焦于句间连贯，用于解决原版 BERT中下一句预测（NSP）损失的低效问题。

&emsp;&emsp;基于上述的这3个设计，ALBERT 能够扩展为更大的版本，在参数量仍然小于 BERT-large的同时，性能可以显著提升。本文在GLUE、SQuAD 和 RACE 这3个自然语言理解基准测试上都刷新了记录：在 RACE 上的准确率为 89.4%，在 GLUE 上的得分为 89.4，在 SQuAD 2.0 上的 F1 得分为 92.2。

<center>

![GLUE榜单](https://img-blog.csdnimg.cn/20190929114241784.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2xqcDE5MTk=,size_16,color_FFFFFF,t_70)
</center>
<center>

![SQuAD榜单](https://img-blog.csdnimg.cn/20190929114335676.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2xqcDE5MTk=,size_16,color_FFFFFF,t_70)
</center>

#  2  主要改进点详解
##  2.1  词嵌入矩阵因式分解

&emsp;&emsp;在BERT、XLNet、RoBERTa等模型中，由于模型结构的限制，WordePiece embedding的大小`$E$` 总是与隐层大小`$H$`相同，即`$E \equiv H$` 。从建模的角度分析，词嵌入学习的是单词与上下文无关的表示，而隐层则是学习与上下文相关的表示。显然后者更加复杂，需要更多的参数，也就是说模型应当增大隐层大小`$H$`，或者说满足`$H >> E$`，但实际上词汇表的大小`$V$`通常非常大，如果`$E=H$`，那么增加隐层大小`$H$`后词嵌入矩阵的维度`$V \times E$`将会非常巨大。

&emsp;&emsp;因此本文想要打破`$H$`与 `$E$` 之间的绑定关系，从而减小模型的参数量，同时提升模型表现。具体做法是将embedding matrix分解为两个大小分别为 `$V \times E$` 和 `$E \times H$` 矩阵，也就是说先将单词投影到一个低维的embedding空间 `$E$` ，再将其投影到高维的隐藏空间 `$H$` 。这使得embedding matrix的维度从 `$O(V \times H)$` 减小到 `$O(V \times E + E \times H)$` 。当 `$H >> E$` 时，参数量减少非常明显。在实现时，随机初始化`$V \times E$`和`$ E \times H$` 的矩阵，计算某个单词的表示需用一个单词的one-hot向量乘以`$V \times E$` 维的矩阵（也就是lookup），再用得到的结果乘`$E \times H$` 维的矩阵即可。两个矩阵的参数通过模型学习。

&emsp;&emsp;简单来说，BERT的情况就是，`$E \equiv H$`；ALBERT的方案是，将`$E$`降低，在词嵌入和隐藏层之间加入一个project层，连接两个层。我们来分析一下，两种情况嵌入层的参数量。

1、BERT
```math
ParameterNumBERT = E \times V = H \times V
```
&emsp;&emsp;通常情况下V很大，BERT中文模型V约为30000，BERT_base中H = 1024。则
```math
ParameterNumBERT=30000*1024
```
2.ALBERT
```math
ParameterNumAL = (V +H)*E
```
&emsp;&emsp;ALBERT中，E=128；H=1024，那么
```math
ParameterNumAL=30000*128+128*1024
```
```math
ParameterNumAL/ParameterNumAL =7.7
```
从上面的分析可以看出，通过嵌入层的参数因式分解，成功将嵌入层的参数缩小为原来的1/8。

##  2.2  隐藏层参数共享
<center>

![SQuAD榜单](https://pics5.baidu.com/feed/ac345982b2b7d0a277629cea5e8bc50c4a369a8c.jpeg?token=c77a64817cdcbfc328abdae073a86e41&s=4BAC386291D749EB567030CC0300A0A0)
</center>

&emsp;&emsp;上图是BERT的结构示意图，BERT_base中，包含12层中间的隐藏层；BERT_large中，包含24层中间的隐藏层；各层之间的参数均不共享。

&emsp;&emsp;本文提出的另一个减少参数量的方法就是层之间的参数共享，即多个层使用相同的参数。参数共享可以显著减少参数数量，参数共享可以分为全连接层、注意力层的参数共享；在ALBERT中，全连接层、注意力层的参数均是共享的，也就是ALBERT依然有多层的深度连接，但是各层之间的参数是一样的。很明显的，通过这种方式，ALBERT中隐藏层的参数量变为原来的1/12或者1/24。

&emsp;&emsp;如下图所示，实验表明加入参数共享之后，每一层的输入embedding和输出embedding的L2距离和余弦相似度都比BERT稳定了很多。这证明参数共享能够使模型参数更加稳定。
<center>

![SQuAD榜单](https://pic1.zhimg.com/80/v2-498b971f8c4cb04a79d99f852ca3f724_720w.jpg)
</center>

##  2.3  语句顺序预测

&emsp;&emsp;在BERT中，句子间关系的任务是next sentence predict(NSP)，即向模型输入两个句子，预测第二个句子是不是第一个句子的下一句，NSP任务的正例是文章中连续的两个句子，而负例则是从两篇文档中各选一个句子构造而成。在先前的研究中，已经证明NSP是并不是一个合适的预训练任务。本文推测其原因是模型在判断两个句子的关系时不仅考虑了两个句子之间的连贯性（coherence），还会考虑到两个句子的话题（topic）。而两篇文档的话题通常不同，模型会更多的通过话题去分析两个句子的关系，而不是句子间的连贯性，这使得NSP任务变成了一个相对简单的任务。

&emsp;&emsp;在ALBERT中，句子间关系的任务是sentence-order prediction(SOP)，即句子间顺序预测，也就是给模型两个句子，让模型去预测两个句子的前后顺序。具体来说，其正例与NSP相同，但负例是通过选择一篇文档中的两个连续的句子并将它们的顺序交换构造的。这样两个句子就会有相同的话题，模型学习到的就更多是句子间的连贯性。

#  3  ALBERT实验结果
## 3.1  实验设置

&emsp;&emsp;为了进行更公平的对比，本文一方面使用与原始 BERT相同的配置训练试验模型，另一方面采用 BOOKCORPUS 和 English Wikipedia 共计 16GB 的纯文本作为预训练任务的数据。与BERT一样，使用的词典大小是30，000；此外还借鉴了XLNet中使用的SentencePiece。在MLM目标函数上使用`$n-gram$`的masking，随机选用`$n-gram$`的mask遮蔽输入。预测生成的`$n-gram$`的概率：

```math
    p(n) = \frac{1/n}{\sum_{k=1}^{n}1/k}
```
本文设置的n-gram最大长度为3，即MLM目标最多由3个全词组成。

## 3.2  BERT VS ALBERT

<center>

![SQuAD榜单](https://img-blog.csdnimg.cn/20190929114549771.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2xqcDE5MTk=,size_16,color_FFFFFF,t_70)
</center>

从上图中可以看出，，相比于BERT，**ALBERT能够在不损失模型性能的情况下，显著的减少参数量**。ALBERT_xxlarge模型有233M的参数量，其在各个数据集上的表现却能够全面优于有1270M参数的BERT_xlarge模型。此外，还观察到BERT-xlarge在全部的指标上全面溃败于BERT-base。这说明形如BERT-xlarge的大参数模型相较于更小参数量的模型是更难训练的。

## 3.3  嵌入向量的因式分解
<center>

![嵌入向量的因式分解](https://img-blog.csdnimg.cn/20190929114615757.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2xqcDE5MTk=,size_16,color_FFFFFF,t_70)
</center>

&emsp;&emsp;对于non-shared下(BERT-style)，更大的嵌入尺寸能够取得更好的结果，但是提升的幅度其实不大。对于all-shared(ALBERT-style)，嵌入大小128是最好的。基于上述这些结果，本文在后续的实验中的嵌入大小统一选用`$E = 128$`。

## 3.4  跨层参数共享
&emsp;&emsp;下图展示了不同跨层参数共享的效果，同样使用 ALBERT-base 作为示例模型，此外还增加了嵌入大小为768的结果。对比了所有all-shared策略(ALBERT-style)、not-shared 策略(BERT-style)及其介于二者之间的中间策略(仅注意力参数共享，FNN不共享；仅FNN参数共享，注意力参数不共享)。
<center>

![跨层参数共享](https://img-blog.csdnimg.cn/20190929114653489.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2xqcDE5MTk=,size_16,color_FFFFFF,t_70)
</center>

&emsp;&emsp;从上述结果可以看出，all-shared型策略在E=768和E=128上都会一定程度上降低性能。但是，需要说明的是，下降幅度较小，对于E=128，平均下降1.5；对于E=768，平均下降2.5。再细看，共享FFN层的参数，应该是罪魁祸首；而注意力机制的参数共享带来的影响不能一概而论，对于E=128反而在平均性能上提升了0.1，对于E=768平均性能下降0.7。

## 3.5  句子次序预测(SOP)
<center>

![句子次序预测(SOP)](https://img-blog.csdnimg.cn/20190929114712400.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2xqcDE5MTk=,size_16,color_FFFFFF,t_70)
</center>
&emsp;&emsp;本文这里对比了3种策略：没有句子间损失(比如XLNet和RoBERTa)、NSP(比如BERT)、SOP(ALBERT)。这里采用的ALBERT也是ALBERT-base。对比过程，一方面对比自身任务中的准确率，另一方面是下游任务的性能表现。在自身任务这一维度，可以看出NSP损失对于SOP几乎是没有任何益处，NSP训练后，在SOP上的表现只有52%，这跟瞎猜差不了多少。据此，可以得出结论：NSP建模止步于主题识别。反观SOP损失，确实一定程度上能够解决NSP任务，其准确率为78.9%，而自身的准确率为86.5%。更为重要的是，在下游任务上SOP损失统统起到促进作用，具体表现在SQuAD1.1提升1%，SQuAD 2.0提升2%，RACE提升1.7%。

## 3.5  相同训练时长下的对比

<center>

![句子次序预测(SOP)](https://img-blog.csdnimg.cn/20190929114754252.png)
</center>
&emsp;&emsp;在训练了差不多相同的时间之后，ALBERT-xxlarge 明显优于 BERT-large

## 3.6 引入额外训练集和Dropout的影响
&emsp;&emsp;上述实验都是在 Wikipedia 和 BOOKCORPUS 数据集上进行的，那么，如果增加额外的数据会对结果产生怎样的影响？这里采用的额外数据与XLNet和RoBERTa中的相同。
<center>

![引入额外训练集](https://img-blog.csdnimg.cn/20190929114820944.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2xqcDE5MTk=,size_16,color_FFFFFF,t_70)
</center>
Figure 3a 表明，添加额外数据后，模型的开发集 MLM 准确率显著提升。
&emsp;&emsp;添加额外数据后模型在下游任务中的性能情况如下图所示
<center>

![引入额外训练集](https://img-blog.csdnimg.cn/20190929114850824.png)
</center>
&emsp;&emsp;我们还注意到，即使在训练了100万步之后，最大的模型仍然没有过拟合。因此，尝试删除dropout，以进一步提高模型能力。如Figure 3b 所示，去掉dropout可以显著提高MLM准确度。去掉dropout后在下游任务上的表现如下图所示：
<center>

![引入额外训练集](https://img-blog.csdnimg.cn/20190929114909630.png)
</center>

## 3.7 当下SOTA模型在NLU任务上的对比
&emsp;&emsp;除了上述实验之外，ALBERT 在 GLUE、SQuAD 和 RACE 基准测试中都取得了 SOTA 结果，如Figure 10、11 所示：
<center>

![引入额外训练集](https://img-blog.csdnimg.cn/20190929114933675.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2xqcDE5MTk=,size_16,color_FFFFFF,t_70)
</center>
<center>

![引入额外训练集](https://img-blog.csdnimg.cn/20190929115007415.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2xqcDE5MTk=,size_16,color_FFFFFF,t_70)
</center>
可以看出ALBERT在各个NLU任务上的表现，几乎都达到了state-of-the-art的表现。

# 4  总结
ALBERT模型的提出其主要贡献如下：
- 提出Factorized embedding和层之间参数共享两种削减参数量的方式，在大家都想着把模型做大的时候给大家指出了另一条可行的路，意义重大。但本文提出的两种方法实际上都带来了模型效果的下降，也就是说本文似乎也还没有找到BERT中真正的冗余参数，减少模型参数量这方面还需要更多的研究。
- 本文提出了SOP，很好地替换了NSP作为预训练任务，给模型表现带来了明显提升。
- 本文的削减参数使模型表现下降，结果更好主要是靠SOP、更大的 `$H$`、更多的数据、去除dropout。那么如果不削减参数的话再使用SOP、加更多的数据、去除dropout呢？
- 本文的削减参数量参数量带来了模型训练速度的提升，但是ALBERT-xxlarge比BERT-xlarge参数量少了约1000M，而训练速度并没有太大的提升（只有1.2倍）。原因应该是更少的参数量的确能带来速度上的提升，但是本文提出的Factorized embedding引入了额外的矩阵运算，并且同时ALBERT-xxlarge大幅增加了 `$H$` ，实际上增加了模型的计算量。
- 本文还有两个小细节可以学习，一个是在模型不会过拟合的情况下不使用dropout；另一个则是warm-start，即在训练深层网络（例如12层）时，可以先训练浅层网络（例如6层），再在其基础上做fine-tune，这样可以加快深层模型的收敛。