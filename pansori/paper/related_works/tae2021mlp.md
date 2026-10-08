---
title: "MLP Singer: Towards Rapid Parallel Korean Singing Voice Synthesis"
authors: "Jaesung Tae; Hyeongju Kim; Younggun Lee"
venue: "2021 IEEE 31st International Workshop on Machine Learning for Signal Processing (MLSP)"
year: 2021
bib_key: null
source_url: "https://arxiv.org/pdf/2106.07886"
version: "arXiv v3 (published at IEEE MLSP 2021)"
cited_in_paper: false
relevance: "Efficient Korean non-autoregressive SVS baseline relevant to practical, low-compute model training and inference."
---
## Why it matters

Efficient Korean non-autoregressive SVS baseline relevant to practical, low-compute model training and inference.


2021 IEEE INTERNATIONAL WORKSHOP ON MACHINE LEARNING FOR SIGNAL PROCESSING, OCT. 25–28, 2021, GOLD COAST, AUSTRALIA

# MLP SINGER: TOWARDS RAPID PARALLEL KOREAN SINGING VOICE SYNTHESIS

_Jaesung Tae_<sup>1</sup><sup>_∗_</sup> _Hyeongju Kim_<sup>2</sup> _Younggun Lee_<sup>2</sup>

1 Yale University, New Haven, CT, USA

2 Neosapience, Inc., Seoul, Republic of Korea

## ABSTRACT

Recent developments in deep learning have significantly improved the quality of synthesized singing voice audio. However, prominent neural singing voice synthesis systems suffer from slow inference speed due to their autoregressive design. Inspired by MLP-Mixer, a novel architecture introduced in the vision literature for attention-free image classification, we propose MLP Singer, a parallel Korean singing voice synthesis system. To the best of our knowledge, this is the first work that uses an entirely MLP-based architecture for voice synthesis. Listening tests demonstrate that MLP Singer outperforms a larger autoregressive GAN-based system, both in terms of audio quality and synthesis speed. In particular, MLP Singer achieves a real-time factor of up to 200 and 3400 on CPUs and GPUs respectively, enabling order of magnitude faster generation on both environments.<sup>1</sup>

**_Index Terms_ —** singing voice synthesis, parallel text-tospeech, multi-layer perceptrons

# 1. INTRODUCTION

Singing voice synthesis (SVS) is a generative task in which a model learns to map musical scores containing linguistic, temporal, and melodic information to acoustic features. While SVS bears similarity to text-to-speech (TTS), it is distinct in that both pitch and temporal information are already provided in the input. A typical TTS model has to learn prosodies as well as temporal alignments between linguistic and acoustic features. Contrastingly in SVS, it is apparent at every time step which syllable should be voiced at what pitch given a musical score. Therefore, the objective of an SVS model is to learn to project time-aligned phoneme and pitch representations to acoustic features while interpolating details not immediately present in the input, such as timbre, breath, and vocalization methods. Some systems relax this task by assuming continuous _f_ 0 values as given [1, 2]; however, the problem we consider here is one in which only low-resolution

> _∗_ Work performed while interning at Neosapience.

> 1Source code available at https://github.com/neosapience/ mlp-singer.

pitch sequences are available, and the model has to learn how to dequantize that information to produce smooth acoustic features.

Traditional SVS systems rely on concatenation [3] or statistical parametric methods such as hidden Markov models [4], which require large amounts of data or complex pipelines for high quality synthesis. On the other hand, deep neural networks have proven to be remarkably effective in speech synthesis due to their ability to approximate complex nonlinear functions. Building on top of recent advancements in TTS architectures, modern SVS systems employ deep neural networks to autoregressively generate vocoder features from melody and lyrics. NPSS [5] uses causal dilated convolutions to produce WORLD vocoder features [6] conditioned on musical scores. Lee et al. [7] proposes an adversarial training scheme and phonetic enhancement masks to improve pronunciation, while Choi et al. proposes an autoregressive model trained on the boundary equilibrium GAN objective [8].

While autoregressive sampling can generate smooth acoustic features, it has a number of drawbacks. Autoregressive models are prone to exposure bias caused by traintest discrepancy; during training, the model is teacher-forced ground-truth data, which are not available during inference [9]. Moreover, generation can be time-consuming, especially for long sequences. Autoregressive sampling requires that each output be contingent on the output from the previous time step. This introduces temporal overhead since each input can only be processed after previous chunks have fully been produced. Blaauw et al. [2] avoids these issues by using a feedforward transformer [10] to generate WORLD vocoder features from alignments produced by a duration predictor. However, the computational complexity of self-attention [11] highlights the need for a light-weight baseline.

We propose MLP Singer, a parallel Korean singing voice synthesis system exclusively composed of multi-layer perceptrons (MLPs). MLP Singer is based on MLP-Mixer [12], an architecture introduced in the computer vision literature that demonstrates the potential of MLPs as a competitive replacement for transformers and convolutional neural networks. MLPs scale linearly with respect to input, and their simple structure benefits from hardware and software accel-

978-1-7281-6338-3/21/$31.00 ©2021 IEEE

erations highly optimized for matrix multiplication. Through listening tests, we show that the proposed system serves as a strong baseline, achieving superior performance to a larger autoregressive conditional GAN-based system [8] both in terms of audio quality and inference latency. We also propose an overlapped batch segmentation method as a means of reducing discontinuous frame artifacts that typically affect non-autoregressive models.

# 2. ARCHITECTURE

An overview of MLP Singer is illustrated in Figure 1. The model receives lyrics text and a sequence of MIDI notes as input. Exploiting the fact that text and pitch sequences are time-aligned, we convert each sequence to embeddings and concatenate them to feed into the model. The first layer is a fully-connected layer that projects the concatenated embeddings to a latent space, from which subsequent Mixer blocks gradually apply transformations to map intermediate representations to acoustic features. Finally, the output is projected to the mel-spectrogram space through a fully-connected layer. In the sections that follow, we expound the components of the model in greater detail.

## 2.1. Data Representation

## _2.1.1. Input Processing_

Given a total sequence length _T_ , let **m** = ( _m_ 1 _, m_ 2 _, . . . , mT_ ) denote the sequence of MIDI events, and **c** = ( _c_ 1 _, c_ 2 _, . . . , cT_ ) denote the sequence of text as syllables. Each MIDI event _mi_ can be decomposed into ( _pi, si, ei_ ), where _pi_ denotes the pitch of the _i_ -th event as MIDI note numbers, _si_ the starting time of the event, and _ei_ the ending time. _mi_ and _ci_ combined informs the duration of the _i_ -th syllable and its pitch.

Because _T_ does not equal the number of mel-spectrogram frames _L_ , **p** = ( _p_ 1 _, p_ 2 _, . . . , pT_ ) and **c** are elongated to match _L_ . This expansion is possible because each mel-spectrogram frame can be attributed to a single index in [1 : _T_ ] by observing the starting times and ending times given from **m** . The resulting sequences are **p**<sup>_′_</sup> and **c**<sup>_′_</sup> , which represent spectrogramaligned pitch and phoneme sequences, respectively.

## _2.1.2. Embedding_

In Korean, every syllable can be subdivided into at most three phonemes: an onset, nucleus, and an optional coda. Exploiting the fact that vowels consume the majority of extended vocalizations, Lee et al. [7] allocates one frame for the onset and coda, populating the rest with the nucleus. We follow a similar approach adopted by Choi et al. [8] and introduce consonant emphasis factor _k_ as a hyperparameter. The distribution scheme, then, is to allot _k_ frames for the onset and coda, and _n −_ 2 _k_ for the nucleus. Empirically, a _k_ value of 3 produced desirable results. Although this is not an exact alignment, we

found that the model is capable of generating plausible acoustic features from this initial approximation given its sufficient receptive field.

The expanded sequence **c**<sup>_′_</sup> is embedded through a lookup table into **Ec** _∈_ R<sup>_L×Dc_</sup> , where _Dc_ refers to the dimensionality of the phoneme embedding. By the same token, **p**<sup>_′_</sup> is transformed into **Ep** through an embedding layer. The phoneme and pitch embeddings are then concatenated to **E** = ( **Ec** _,_ **Ep** ) _∈_ R<sup>_L×D_</sup> , where _D_ = _Dc_ + _Dp_ and _Dp_ denotes the cardinality of the pitch embedding space. **E** is then passed through a single fully-connected layer to produce a latent representation that is a linear combination of the phoneme and pitch embeddings.

## 2.2. Mixer Block

We adopt the Mixer block from MLP-Mixer [12] with minimal modifications. A Mixer block comprises pre-layer normalization [13] and feedforward layers with residual skip connections. A single feedforward layer is defined by two affine transformations with an intermediate activation. The structure is schematically shown below.


_σ_ ( _·_ ) represents a non-linearity, in this case Gaussian Error Linear Units (GELU) [14]. The dimensions of **W** 1<sup>_i_and</sup><sup>**W**</sup> 2<sup>_i_</sup> are R<sup>_Di×Dh_</sup> and R<sup>_Dh×Di_</sup> , respectively, where _Dh_ denotes the size of the hidden dimension and _Di_ denotes the input dimension of the _i_ -th feedforward layer. Dropout and bias terms are omitted for brevity.

A Mixer block consists of two such feedforward layers, except that the second layer is applied on the transposed axis.


where ˜ _·_ := layernorm( _·_ ) and **X** _l_ denotes the input to the _l_ -th Mixer block.

We follow the nomenclature in MLP-Mixer and refer to the feedforward layer in Equation 2 as the Channel Mixer and 3 as the Token Mixer. The Channel Mixer is applied to each embedding, whereas the Token Mixer is applied in the transposed axis across different time steps. In MLP-Mixer, the Token Mixer precedes the Channel Mixer; in our experiments, we found that exchanging the order of these components yielded some benefits. We hypothesize that this is due to the importance of sufficiently diffusing text and pitch information across embedding channels prior to passing through the first Token Mixer layer.

At the core of this architecture is the Token Mixer. Unlike convolutional neural networks, which gradually enlarge their receptive fields via dilations or deep stacking, the Token Mixer gains immediate access to the entire input chunk


Transpose<br>Dropout<br>Fully-connected<br>Feedforward (Token Mixer)<br>Fully-connected<br>N     Mixer blocks Transpose<br>Dropout<br>LayerNorm<br>Fully-connected<br>GELU<br>ㅎ ㅏ ㅏ ㅏ ㅏ ㅂ ㅅ ㅓ ㅓ ㅓ ㅓ ㅇ Feedforward (Channel Mixer)<br>Fully-connected<br>C C C C C C D D D D D D<br>LayerNorm<br>성<br>합<br>

**Fig. 1** . Overview of MLP Singer

by accepting latent representations in their transposed form. The wide context window afforded to the model starting from the very first Mixer block helps MLP Singer generate latent features while being aware of neighboring frames within the input window.

The resulting output from _N_ Mixer blocks is projected by a final fully-connected layer that sends **X** _N_ +1 _∈_ R<sup>_L×D_</sup> to R<sup>_L×D_mel</sup> , where _D_ mel denotes the mel-spectrogram frequency bin. The model is trained on L1 mel-spectrogram loss.

## 2.3. Overlapped Batch Segmentation

One corollary of transposed matrix multiplication is the constraint that inputs be padded or truncated to length _L_ . Since a typical song is much longer than _L_ , in practice, generating audio for an entire song requires padding and reshaping the input into batches of segments that are _L_ in length. Batching inputs in this fashion is possible only because MLP Singer generates acoustic features in parallel; autoregressive models can only process one segment at a time, and thus require iterative loops. MLP Singer fully enjoys the benefit of hardware and software accelerations optimized for batch processing.

A potential downside of simple batching is that the model may produce audible artifacts around segment boundaries. Choi et al. [8] found that autoregressive models generated coherent spectrograms, whereas models that were not provided any previous context produced spectrograms with abrupt interruptions in harmonic patterns. Motivated by a similar concern, we also consider an improved scheme in which the input is batched into overlapping chunks, where the size of the overlap window is given by _w_ . The first and last _w_ frames of each chunk are then dropped to remove redundant frames. We experiment with both batching schemes to explore the importance of overlapping windows in parallel generation.

# 3. EXPERIMENT

## 3.1. Dataset

We used the Children’s Song Dataset (CSD) [15], an open dataset composed of English and Korean children songs sung by a professional female singer. Each song is sung twice in two different keys. We used 50 Korean songs, which totals approximately two hours in length excluding silence intervals. Each song is accompanied by MIDI and text annotations. We used a rule-based Korean grapheme-to-phoneme method [16] to preprocess raw text. 45 songs were used for training, one song for validation, and four songs for testing. To generate ground-truth mel-spectrograms, we downsampled the recordings to 16 kHz, applied a pre-emphasis factor of 0.97, then used a filter length of 1024, hop size of 200, window length of 800, and _D_ mel of 120 for STFT.

## 3.2. Setup

We used 16 Mixer blocks as the backbone of MLP Singer. The text embedding dimension _Dc_ was set to 256; _Dp_ , 32. _Dp_ was kept deliberately small since pitch embeddings only have to express a total of 25 notes including a silence token. The maximum sequence length _L_ was set to 200 frames, which translates to 2.5 seconds. If _L_ is excessively large, the model would unnecessarily attempt to learn noisy correlations between frames that are too distant, whereas the converse would limit the receptive field of the model. Dropout probability was set to 0.5 for all layers. We expect the model to perform better with more exhaustive hyperparameter search.

MLP Singer was trained on a single NVIDIA RTX 6000 GPU with a batch size of 384, using the Adam optimizer [17] with _β_ 1 = 0 _._ 9, _β_ 2 = 0 _._ 999. The learning rate was set to 0.001 and adjusted with linear warmup followed by linear decay.

## 3.3. Evaluation

We compared MLP Singer with BEGANSing, an autoregressive conditional GAN-based model proposed by Choi et al. [8]. While we also considered including Adversarial SVS [7], we were unable to reproduce the results of the paper, nor were there open source resources. The official implementation of BEGANSing was publicly available.

For the vocoder, we used a variant of HiFi-GAN [18], which was fine-tuned on CSD to account for distributional differences between mel-spectrograms produced from speech and singing voice recordings. Since BEGANSing outputs magnitude spectrograms, we applied a mel-basis projection to convert synthesized outputs to mel-spectrograms. We evaluated MLP Singer under two configurations: naive batching and overlapped batch segmentation as delineated in Section 2.3, with _w_ set to 30. Lastly, ground-truth recordings and their vocoder reconstructions were included as points of reference to estimate vocoder-induced bottleneck. We evaluated the SVS systems on both audio quality and inference speed.

## _3.3.1. Audio Quality_

8 audio segments were randomly selected from the test set. We conducted a mean opinion score (MOS) evaluation with 10 participants.<sup>2</sup> Each sample was rated from a scale of 1 to 5, in 0.5 increments. MLP Singer and overlapped batch segmentation are abbreviated as “MLPS” and “OB”, respectively.

|Model|Params. (M)_↓_|MOS_↑_|
|---|---|---|
|BEGANSing [8]|42|2_._325_±_0_._144|
|MLPS (Ours)|**8**|2_._875_±_0_._149|
|MLPS + OB (Ours)|**8**|**3**_._**169**_±_**0**_._**153**|
|GT|-|4_._269_±_0_._105|
|Reconst.|-|3_._575_±_0_._153|


**Table 1** . Parameter count and MOS

Overall, MLP Singer received higher ratings than the baseline model. The result also suggests that overlapped batch segmentation positively impacts perceived audio quality. A paired sample _t_ -test on the MOS of MLP Singer with and without overlapped batching indicated a statistically significant difference at a 95% confidence interval given a _p_ -value of 3 _._ 525 _×_ 10<sup>_−_5</sup> , suggesting that overlapped batch segmentation helped produce better results.

Vocoder reconstructions received considerably lower scores compared to ground-truth recordings. We suspect that this was due to noisy artifacts that the vocoder produced at high frequency intervals. Given the non-negligible vocoder bottleneck, the MOS for ground-truth reconstructions can be viewed as the upper bound for an SVS system.

2Audio samples available at https://mlpsinger.github.io.


10000<br>BEGANSing<br>8000 MLPS<br>MLPS + OB<br>6000<br>4000<br>2000<br>0<br>450<br>400<br>350<br>300<br>250<br>200<br>150<br>100<br>50<br>0<br>1000 2000 3000 4000 5000 6000 7000 8000 9000 10000<br>Number of Frames<br>CPU Latency (ms)<br>GPU Latency (ms)<br>

**Fig. 2** . Inference latency comparison on CPU and GPU

## _3.3.2. Inference Speed_

Inference speed tests were conducted on an Intel Core i9 CPU and NVIDIA RTX 6000 GPU. We measured the time it took for each model to produce a given number of mel-spectrogram frames, including data operations such as tensor reshaping and slicing. Each measurement was taken 20 times to produce reliable estimates.

As shown in Figure 2, MLP Singer consistently outperforms the autoregressive baseline on both GPU and CPU. As the number of predicted mel-spectrogram frames increases, BEGANSing’s inference latency grows conspicuously, whereas MLP Singer slopes only slightly, even with the added overhead caused by overlapped batch segmentation. We also found a substantial gap in the real-time factor (RTF), which was computed by dividing the total duration of synthesized speech by inference latency. The average RTF of MLP Singer on CPU and GPU were 203 and 3401 (145 and 2514 with overlapped batch segmentation), whereas BEGANSing achieved approximately 14 and 301, respectively. MLP Singer’s CPU latency is thus comparable to that of BEGANSing on GPU runtime, which demonstrates the scalability and efficiency of the proposed system.

# 4. DISCUSSION

## 4.1. Shift Invariance

Initially, the Token Mixer appears inappropriate for processing sequential data since features in each time step get multiplied with a different set of weights. This could be unde-


**Fig. 3** . Heatmaps of learned Toeplitz-like transformations

sirable since we expect the model to generate roughly identical acoustic features regardless of where a particular pitchsyllable combination appears in the input chunk.

To further investigate this intuition, we conducted an ablation study in which we removed Token Mixers from MLP Singer and constructed a model purely composed of Channel Mixers. This ablated model is essentially a stack of fullyconnected layers with a receptive field that is limited to a single time step. To ensure that model performance is not affected by differences in parameter count, we configured the ablated model to have 24 layers, which amounts to 8M parameters. Despite this adjustment, the model suffered a noticeable drop in performance. We quantitatively evaluated this degradation by measuring the L1 mel-spectrogram loss on the holdout set. We also computed mel-cepstral distortion (MCD) to introduce an additional metric unused during training.

|Model|L1 Loss_↓_|MCD_↓_|
|---|---|---|
|MLP Singer|**0**_._**0725**|**1**_._**87**|
|Channel Mixer-only|0_._0837|2_._26|


**Table 2** . Token Mixer ablation results

The relatively high L1 loss and MCD values indicate that the ablated model failed to learn quality transformations that approximate ground-truth spectrograms. Given that Channel Mixers and Token Mixers operate on orthogonal dimensions, the result suggests that transformations on one axis cannot substitute the effect of that on the transposed dimension.

By examining the trained weights of the model, we found that Token Mixers learned Toeplitz-like transformations that reflect shift invariance. Since a Token Mixer is composed of two fully-connected layers, we fed an identity matrix I _L_ and plotted the output to visualize their aggregate effect. Figure 3 displays the output produced by 8th and 11th Token Mixers. Previous work that explored similar MLP-based models [19] found that masked language modeling [20] encouraged models to learn shift invariance via Toeplitz transformations, since any offset of the input should not affect the task of predicting a masked token. Similarly in SVS, moving the text and MIDI inputs by some number of frames should not result in drastically different mel-spectrograms. Despite the lack


**Fig. 4** . Segment boundaries in synthesized mel-spectrograms (without overlapped batch segmentation)

of an explicit inductive bias, MLP Singer becomes robust to temporal shifts by learning Toeplitz-like matrices, which are characterized by their constancy along the descending diagonal. Thus, regardless of where a particular feature appears in an input segment, it is multiplied by a similar set of weights and undergoes a shift-invariant transformation.

## 4.2. Frame Congruence

Non-autoregressive models are known to produce poorly connected spectrogram frames since each chunk is synthesized independently of one another [5, 8]. Generally, we found that MLP Singer is capable of generating continuous frames on intervals with smooth harmonic patterns. Since the model has learned shift-invariant transformations, it can produce roughly similar outputs regardless of how the input is chunked. The top row of Figure 4 illustrates an instance of felicitous generation; the bottom row displays a counterexample that contains jagged artifacts near segment edges. The two outputs only differ by how the input was chunked. Segment boundaries are indicated as boxes.

Empirically, we found that visible junctures occur when segment boundaries coincide with transitions, such as pitch changes, rapid modulations, or movement from voiced to unvoiced intervals. We speculate that this is due to the limitation of Toeplitz transformations. Given a Toeplitz-like matrix with 2 _r_ diagonally constant components, shift invariance does not apply to the first and last _r_ frames in the input segment. In other words, embeddings in the first few time steps are only mixed with features in the future, whereas the last few representations are only mixed with those from the past. In contrast, the Token Mixer can produce rich representations for frames that are located in the the middle of the chunk between indices [ _r, L − r_ ], since it has sufficient bidirectional access to features _±r_ frames away from the current time step. This analysis is coherent with the observed average per-location mel-spectrogram loss on the test set, which is highest at each ends of the model’s receptive field as shown in Figure 5. The overlapping batch segmentation method mitigates this problem by enabling the model to look _w_ steps into the past and

0 50 100 150 200


**Fig. 5** . Average per-location mel-spectrogram loss

future at segment boundaries via overlaps. The theoretically optimal value of _w_ would then equal _r_ .

Transitions and segment boundaries rarely coincide by chance, which is why MLP Singer can reliably produce felicitous outputs. Experiments also show that overlapping batch segmentation leads to better perceived audio quality. Nonetheless, improved batching is an inference time sampling technique rather than a fundamental architectural improvement. We leave this area for investigation in future work.

# 5. CONCLUSION

We present MLP Singer, an all-MLP parallel Korean singing voice synthesis system. MLP Singer demonstrates competitive edge over an existing autoregressive GAN baseline both in terms of synthesis quality and inference latency, achieving order of magnitude faster real-time factors on both CPU and GPU by leveraging parallel generation. We analyze the effectiveness of MLPs in terms of their ability to learn shift invariance as Toeplitz-like transformations. We also propose a batch segmentation method that uses overlapping windows to ensure the model sufficient bidirectional receptive field, thus preventing it from generating discontinuous frame artifacts that typically affect non-autoregressive systems.

- [5] Merlijn Blaauw and Jordi Bonada, “A neural parametric singing synthesizer modeling timbre and expression from natural songs,” _Applied Sciences_ , vol. 7, 2017.

- [6] Masanori Morise, Fumiya Yokomori, and Kenji Ozawa, “World: A vocoder-based high-quality speech synthesis system for real-time applications,” _IEICE Transactions on Information and Systems_ , vol. E99.D, no. 7, 2016.

- [7] Juheon Lee, Hyeong-Seok Choi, Chang-Bin Jeon, Junghyun Koo, and Kyogu Lee, “Adversarially trained end-to-end korean singing voice synthesis system,” in _INTERSPEECH_ , 2019.

- [8] Soonbeom Choi, Wonil Kim, Saebyul Park, Sangeon Yong, and Juhan Nam, “Korean singing voice synthesis based on auto-regressive boundary equilibrium gan,” in _ICASSP_ , 2020.

- [9] Florian Schmidt, “Generalization in generation: A closer look at exposure bias,” in _WNGT@EMNLPIJCNLP_ , 2019.

- [10] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin, “Attention is all you need,” in _NIPS_ , 2017.

- [11] Nikita Kitaev, Lukasz Kaiser, and Anselm Levskaya, “Reformer: The efficient transformer,” in _ICLR_ , 2020.

- [12] Ilya Tolstikhin, Neil Houlsby, Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Thomas Unterthiner, Jessica Yung, Andreas Steiner, Daniel Keysers, Jakob Uszkoreit, Mario Lucic, and Alexey Dosovitskiy, “Mlpmixer: An all-mlp architecture for vision,” 2021.

- [13] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton, “Layer normalization,” 2016.

- [14] Dan Hendrycks and Kevin Gimpel, “Gaussian error linear units (gelus),” 2020.

# 6. ACKNOWLEDGEMENT

We express gratitude to Juhan Nam of KAIST Music and Audio Computing Lab for valuable discussions and feedback.

# 7. REFERENCES

- [1] Pritish Chandna, Merlijn Blaauw, Jordi Bonada, and Emilia G´omez, “Wgansing: A multi-voice singing voice synthesizer based on the wasserstein-gan,” in _EUSIPCO_ , 2019.

- [2] Merlijn Blaauw and Jordi Bonada, “Sequence-tosequence singing synthesis using the feed-forward transformer,” in _ICASSP_ , 2020.

- [3] J. Bonada, Alex Loscos, Oscar Mayor, and H. Kenmochi, “Sample-based singing voice synthesizer using spectral models and source-filter decomposition,” in _MAVEBA_ , 2003.

   - [15] Soonbeom Choi, Wonil Kim, Saebyul Park, Sangeon Yong, and Juhan Nam, “Children’s song dataset for singing voice research,” in _ISMIR_ , 2020.

   - [16] Yejin Cho, “Korean grapheme-to-phoneme analyzer (kog2p),” https://github.com/scarletcho/ KoG2P, 2017.

   - [17] Diederik P. Kingma and Jimmy Ba, “Adam: A method for stochastic optimization,” in _ICLR_ , 2015.

   - [18] Jungil Kong, Jaehyeon Kim, and Jaekyoung Bae, “Hifigan: Generative adversarial networks for efficient and high fidelity speech synthesis,” in _NeurIPS_ , 2020.

   - [19] Hanxiao Liu, Zihang Dai, David R. So, and Quoc V. Le, “Pay attention to mlps,” 2021.

   - [20] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova, “BERT: Pre-training of deep bidirectional transformers for language understanding,” in _NAACL_ , 2019.

- [4] Keijiro Saino, Heiga Zen, Yoshihiko Nankaku, Akinobu Lee, and Keiichi Tokuda, “An hmm-based singing voice synthesis system,” in _ICSLP_ , 2006.
