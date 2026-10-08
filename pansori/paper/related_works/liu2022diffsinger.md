---
title: "DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism"
authors: "Jinglin Liu; Chengxi Li; Yi Ren; Feiyang Chen; Zhou Zhao"
venue: "Proceedings of the AAAI Conference on Artificial Intelligence"
year: 2022
bib_key: "liu2022diffsinger"
source_url: "https://cdn.aaai.org/ojs/21350/21350-13-25363-1-2-20220628.pdf"
version: "published"
cited_in_paper: true
---
> Extraction note: 42 replacement glyph(s) remain, mainly in equations or figure text; the source PDF is required when an equation is ambiguous.


# DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism

**Jinglin Liu**<sup>*</sup> **, Chengxi Li**<sup>*</sup> **, Yi Ren**<sup>*</sup> **, Feiyang Chen, Zhou Zhao**<sup>†</sup>

Zhejiang University _{_ jinglinliu,chengxili,rayeren,zhaozhou _}_ @zju.edu.cn, chenfeiyangai@gmail.com

## Abstract

Singing voice synthesis (SVS) systems are built to synthesize high-quality and expressive singing voice, in which the acoustic model generates the acoustic features ( _e_ . _g_ ., mel-spectrogram) given a music score. Previous singing acoustic models adopt a simple loss ( _e_ . _g_ ., L1 and L2) or generative adversarial network (GAN) to reconstruct the acoustic features, while they suffer from over-smoothing and unstable training issues respectively, which hinder the naturalness of synthesized singing. In this work, we propose DiffSinger, an acoustic model for SVS based on the diffusion probabilistic model. DiffSinger is a parameterized Markov chain that iteratively converts the noise into mel-spectrogram conditioned on the music score. By implicitly optimizing variational bound, DiffSinger can be stably trained and generate realistic outputs. To further improve the voice quality and speed up inference, we introduce a shallow diffusion mechanism to make better use of the prior knowledge learned by the simple loss. Specifically, DiffSinger starts generation at a shallow step smaller than the total number of diffusion steps, according to the intersection of the diffusion trajectories of the ground-truth mel-spectrogram and the one predicted by a simple mel-spectrogram decoder. Besides, we propose boundary prediction methods to locate the intersection and determine the shallow step adaptively. The evaluations conducted on a Chinese singing dataset demonstrate that DiffSinger outperforms state-of-the-art SVS work. Extensional experiments also prove the generalization of our methods on text-to-speech task (DiffSpeech). Audio samples: https://diffsinger.github.io. Codes: https://github.com/ MoonInTheRiver/DiffSinger.

# 1 Introduction

Singing voice synthesis (SVS) which aims to synthesize natural and expressive singing voice from musical score (Wu and Luan 2020), increasingly draws attention from the research community and entertainment industries (Zhang et al. 2020). The pipeline of SVS usually consists of an acoustic model to generate the acoustic features ( _e_ . _g_ ., mel-spectrogram) conditioned on a music score, and a vocoder to convert the acoustic features to waveform (Nakamura et al.

*Equal contribution.

†Corresponding author


2019; Lee et al. 2019; Blaauw and Bonada 2020; Ren et al. 2020; Chen et al. 2020)<sup>1</sup> .

Previous singing acoustic models mainly utilize simple loss ( _e_ . _g_ ., L1 or L2) to reconstruct the acoustic features. However, this optimization is based on the incorrect unimodal distribution assumptions, leading to blurry and over-smoothing outputs. Although existing methods endeavor to solve this problem by generative adversarial network (GAN) (Lee et al. 2019; Chen et al. 2020), training an effective GAN may occasionally fail due to the unstable discriminator. These issues hinder the naturalness of synthesized singing.

Recently, a highly flexible and tractable generative model, diffusion probabilistic model (a.k.a. diffusion model) (SohlDickstein et al. 2015; Ho, Jain, and Abbeel 2020; Song, Meng, and Ermon 2021) emerges. Diffusion model consists of two processes: diffusion process and reverse process (also called denoising process). The diffusion process is a Markov chain with fixed parameters (when using the certain parameterization in (Ho, Jain, and Abbeel 2020)), which converts the complicated data into isotropic Gaussian distribution by adding the Gaussian noise gradually; while the reverse process is a Markov chain implemented by a neural network, which learns to restore the origin data from Gaussian white noise iteratively. Diffusion model can be stably trained by implicitly optimizing variational lower bound (ELBO) on the data likelihood. It has been demonstrated that diffusion model can produce promising results in image generation (Ho, Jain, and Abbeel 2020; Song, Meng, and Ermon 2021) and neural vocoder (Chen et al. 2021; Kong et al. 2021) fields.

In this work, we propose DiffSinger, an acoustic model for SVS based on diffusion model, which converts the noise into mel-spectrogram conditioned on the music score. DiffSinger can be efficiently trained by optimizing ELBO, without adversarial feedback, and generates realistic mel-spectrograms strongly matching the ground truth distribution.

To further improve the voice quality and speed up inference, we introduce a shallow diffusion mechanism to make better use of the prior knowledge learned by the simple loss. Specifically, we find that there is an intersection of the diffu-

> 1A music score consists of lyrics, pitch and duration.

11020

sion trajectories of the ground-truth mel-spectrogram _M_ and the one predicted by a simple mel-spectrogram decoder _M_<sup>�2</sup> : sending _M_ and _M_<sup>�</sup> into the diffusion process could result in similar distorted mel-spectrograms, when the diffusion step is big enough (but not reaches the deep step where the distorted mel-spectrograms become Gaussian white noise). Thus, in the inference stage we 1) leverage the simple mel-spectrogram decoder to generate _M_<sup>�</sup> ; 2) calculate the sample at a shallow step _k_ through the diffusion process: _M_<sup>�</sup> _k_<sup>3</sup> ; and 3) start reverse process from _M_<sup>�</sup> _k_ rather than Gaussian white noise, and complete the process by _k_ iteration denoising steps (Vincent 2011; Song and Ermon 2019; Ho, Jain, and Abbeel 2020). Besides, we train a boundary prediction network to locate this intersection and determine the _k_ adaptively. The shallow diffusion mechanism provides a better start point than Gaussian white noise and alleviates the burden of the reverse process, which improves the quality of synthesized audio and accelerates inference.

Finally, since the pipeline of SVS resembles that of text-to-speech (TTS) task, we also build DiffSpeech adjusting from DiffSinger for generalization. The evaluations conducted on a Chinese singing dataset demonstrate the superiority of DiffSinger (0.11 MOS gains compared with a state-of-the-art acoustic model for SVS (Wu and Luan 2020)), and the effectiveness of our novel mechanism (0.14 MOS gains, 0.5 CMOS gains and 45.1% speedup with shallow diffusion mechanism). The extensional experiments of DiffSpeech on TTS task prove the generalization of our methods (0.24/0.23 MOS gains compared with FastSpeech 2 (Ren et al. 2021) and Glow-TTS (Kim et al. 2020) respectively). The contributions of this work can be summarized as follows:

- We propose DiffSinger, which is the first acoustic model for SVS based on diffusion probabilistic model. DiffSinger addresses the over-smoothing and unstable training issues in previous works.

- We propose a shallow diffusion mechanism to further improve the voice quality, and accelerate the inference.

- The extensional experiments on TTS task (DiffSpeech) prove the generalization of our methods.

# 2 Diffusion Model

In this section, we introduce the theory of diffusion probabilistic model (Sohl-Dickstein et al. 2015; Ho, Jain, and Abbeel 2020). The full proof can be found in previous works (Ho, Jain, and Abbeel 2020; Kong et al. 2021; Song, Meng, and Ermon 2021). A diffusion probabilistic model converts the raw data into Gaussian distribution gradually by a diffusion process, and then learns the reverse process to restore the data from Gaussian white noise (Sohl-Dickstein et al. 2015). These processes are shown in Figure 1.

> 2Here we use a traditional acoustic model based on feedforward Transformer (Ren et al. 2021; Blaauw and Bonada 2020), which is trained by L1 loss to reconstruct mel-spectrogram.

> 3 _K < T_ , where T is the total number of diffusion steps. _M_ � _k_ can be calculated in closed form time (Ho, Jain, and Abbeel 2020).


diffusion process<br>𝑝𝜃 (𝑦𝑡−1|𝑦𝑡)<br>reverse process<br>𝑦0 𝑦t−1 𝑦t 𝑦T<br>𝑞(𝑦𝑡|𝑦𝑡−1)<br>

Figure 1: The directed graph for diffusion model.

**Diffusion Process** Define the data distribution as _q_ ( **y** 0), and sample **y** 0 _∼ q_ ( **y** 0). The diffusion process is a Markov chain with fixed parameters (Ho, Jain, and Abbeel 2020), which converts **y** 0 into the latent **y** _T_ in _T_ steps:


At each diffusion step _t ∈_ [1 _, T_ ], a tiny Gaussian noise is added to **y** _t−_ 1 to obtain **y** _t_ , according to a variance schedule _β_ = _{β_ 1 _, . . . , βT }_ :


If _β_ is well designed and _T_ is sufficiently large, then _q_ ( _yT_ ) is nearly an isotropic Gaussian distribution (Ho, Jain, and Abbeel 2020; Nichol and Dhariwal 2021). Besides, there is a special property of diffusion process that _q_ ( **y** _t|_ **y** 0) can be calculated in closed form in _O_ (1) time (Ho, Jain, and Abbeel 2020):


**Reverse Process** The reverse process is a Markov chain with learnable parameters _θ_ from **y** _T_ to **y** 0. Since the exact reverse transition distribution _q_ ( **y** _t−_ 1 _|_ **y** _t_ ) is intractable, we approximate it by a neural network with parameters _θ_ ( _θ_ is shared at every _t_ -th step):


**Training** To learn the parameters _θ_ , we minimize a variational bound of the negative log likelihood: E _q_ ( **y** 0)[ _−_ log _pθ_ ( **y** 0)] _≥_

E _q_ ( **y** 0 _,_ **y** 1 _,...,_ **y** _T_ ) [log _q_ ( **y** 1: _T |_ **y** 0) _−_ log _pθ_ ( **y** 0: _T_ )] =: L _._ Efficient training is optimizing a random term of L with stochastic gradient descent (Ho, Jain, and Abbeel 2020):


11021


outputoutput<br>L1 loss with L1 loss with 𝑀𝑀 𝑀𝑀 L1 loss with L1 loss with 𝜖𝜖 𝜖𝜖 𝑀𝑀� 𝑘𝑘𝑀𝑀� 𝑘𝑘 Loop Loop 𝑘𝑘𝑘𝑘/T/T times times<br>Denoiser𝜖𝜖𝜃𝜃Denoiser(·,𝜖𝜖 𝑡𝑡𝜃𝜃(·,)  𝑡𝑡) Alg 2, Line #9<br>𝑀𝑀� Alg 2, Line #9<br>𝑀𝑀�<br>Aux DecoderAux Decoder Em Et Denoiser EEmm<br>Em Et Denoiser<br>E t<br>Et<br>Encoder Step Embedding 𝑀𝑀t 𝜖𝜖,𝑘𝑘,𝛽𝛽<br>Encoder Step Embedding 𝑀𝑀t 𝜖𝜖,𝑘𝑘,𝛽𝛽<br>t<br>t 𝑀𝑀�<br>𝑀𝑀�<br>(a) The training procedure of DiffSinger.𝑥𝑥 (b) The inference procedure of DiffSinger.<br>𝑥𝑥 𝑀𝑀� 𝑘𝑘 𝑀𝑀𝑇𝑇<br>𝑀𝑀� 𝑘𝑘 𝑀𝑀𝑇𝑇<br>

Figure 2: The overview of DiffSinger (with shallow diffusion mechanism in the dotted line boxes). In subfigure (a), _x_ is the music score; _t_ is the step number; _M_ means the ground truth mel-spectrogram; _M_<sup>�</sup> means the blurry mel-spectrogram generated by the auxiliary decoder trained with L1 loss; _Mt_ is _M_ at the _t_ -th step in the diffusion process. In subfigure (b), _MT_ means the _M_ at _T_ -th diffusion step (Gaussian white noise); _k_ is the predicted intersection boundary; there is a switch to select _MT_ (naive version) or _M_<sup>�</sup> _k_ (with shallow diffusion) as the start point of the inference procedure.


where _C_ is a constant. And by reparameterizing Eq. (1) as **y** _t_ ( **y** 0 _,_ **_ϵ_** ) =<sup>_√_</sup> _<u>α</u>_ <u>¯</u> _t_ **y** 0 +<sup>_√_</sup> 1 _− α_ ¯ _t_ **_ϵ_** , and choosing the parameterization:


Eq. (4) can be simplified to:


Finally we set _σt_<sup>2to</sup><sup>_β_˜</sup><sup>_t_, sample</sup><sup>**_ϵ_**</sup><sup>_∼N_(</sup><sup>**0**</sup><sup>_,_</sup><sup>**I**) and</sup><sup>**_ϵ_**</sup><sup>_θ_(</sup><sup>_·_) is the</sup> outputs of the neural network.

**Sampling** Sample **y** _T_ from _p_ ( **y** _T_ ) _∼N_ ( **0** _,_ **I** ) and run the reverse process to obtain a data sample.

# 3 DiffSinger

As illustrated in Figure 2, DiffSinger is built on the diffusion model. Since SVS task models the conditional distribution _pθ_ ( _M_ 0 _|x_ ), where _M_ is the mel-spectrogram and _x_ is the music score corresponding to _M_ , we add _x_ to the diffusion denoiser as the condition in the reverse process. In this section, we first describe a naive version of DiffSinger (Section 3.1); then we introduce a novel shallow diffusion mechanism to improve the model performance and efficiency (Section 3.2); finally, we describe the boundary prediction network which can adaptively find the intersection boundary required in shallow diffusion mechanism (Section 3.3).

## 3.1 Naive Version of DiffSinger

In the naive version of DiffSinger (without dotted line boxes in Figure 2): In the training procedure (shown in Figure 2a), DiffSinger takes in the mel-spectrogram at _t_ -th step _Mt_ in the diffusion process and predicts the random noise **_ϵ_** _θ_ ( _·_ ) in Eq. (6), conditioned on _t_ and the music score _x_ . The inference procedure (shown in Figure 2b) starts at the Gaussian white noise sampled from _N_ ( **0** _,_ **I** ), as the previous diffusion models do (Ho, Jain, and Abbeel 2020; Kong et al. 2021). Then the procedure iterates for _T_ times to repeatedly denoise the intermediate samples with two steps: 1) predict the **_ϵ_** _θ_ ( _·_ ) using the denoiser; 2) obtain _Mt−_ 1 from _Mt_ using the predicted **_ϵ_** _θ_ ( _·_ ), according to Eq. (2) and Eq. (5):


where _z ∼N_ ( **0** _,_ **I** ) when _t >_ 1, and _z_ = 0 when _t_ = 1. Finally, a mel-spectrogram _M_ corresponding to _x_ could be generated.

## 3.2 Shallow Diffusion Mechanism

Although the previous acoustic model trained by the simple loss has intractable drawbacks, it still generates samples showing strong connection<sup>4</sup> to the ground-truth data distribution, which could provide plenty of prior knowledge to DiffSinger. To explore this connection and find a way to make better use of the prior knowledge, we conduct the empirical observation leveraging the diffusion process (shown in Figure 3): 1) when _t_ = 0, _M_ has rich details between the neighboring harmonics, which can influence the naturalness of the synthesized singing voice, but _M_<sup>�</sup> is over-smoothing as we introduced in Section 1; 2) as _t_ increases, samples

> 4The samples fail to maintain the variable aperiodic parameters, but they usually have a clear “skeleton” (harmonics) matching the ground truth.

11022

�𝑀𝑀 𝑀𝑀 Figure 3: The mel-spectrograms at different steps in the dif-𝑡𝑡 = 0 𝑡𝑡 = 25 𝑡𝑡 =50 𝑡𝑡 = 75

Figure 3: The mel-spectrograms at different steps in the dif-𝑡𝑡 fusion process. The first line shows the diffusion process of mel-spectorgrams _M_<sup>�</sup> generated by a simple decoder trained with L1 loss; the second line shows that of ground truth mel-spectrograms.

of two process become indistinguishable. We illustrate this observation in Figure 4: the trajectory from _M_<sup>�</sup> manifold to Gaussian noise manifold and the trajectory from _M_ to Gaussian noise manifold intersect when the diffusion step is big enough.

Inspired by this observation, we propose the shallow diffusion mechanism: instead of starting with the Gaussian white noise, the reverse process starts at the intersection of two trajectories shown in Figure 4. Thus the burden of the reverse process could be distinctly alleviated<sup>5</sup> . Specifically, in the inference stage we 1) leverage an auxiliary decoder to generate _M_<sup>�</sup> , which is trained with L1 conditioned on the music score encoder outputs, as shown in the dotted line box in Figure 2a; 2) generate the intermediate sample at a shallow step _k_ through the diffusion process, as shown in the dotted line box in Figure 2b according to Eq. (1):


where **_ϵ_** _∼N_ ( **0** _,_ **I** ), _α_ ¯ _k_ :=<sup>�</sup><sup>_k_</sup> _s_ =1<sup>_αs_,</sup><sup>_αk_:=1</sup><sup>_−βk_. If the</sup> intersection boundary _k_ is properly chosen, it can be considered that _M_<sup>�</sup> _k_ and _Mk_ come from the same distribution; 3) start reverse process from _M_<sup>�</sup> _k_ , and complete the process by _k_ iterations denoising. The training and inference procedures with shallow diffusion mechanism are described in Algorithm 1 and 2 respectively. The theoretical proof of the intersection of two trajectories can be found in the supplement.


𝒕� T<br>Gaussian Noise Manifold<br>𝒕�𝒌<br>𝒕�𝟎<br>𝑀 Manifold 𝒕�𝟎<br>𝑴 � Manifold<br>

Figure 4: The diffusion trajectories of _M_ and _M_<sup>�</sup> . Two distributions _q_ ( _Mt|M_ 0) and _q_ ( _M_<sup>�</sup> _t|M_<sup>�</sup> 0) become closer as _t_ increases.

written as:


where _Y_ is the training set of mel-spectrograms. When BP have been trained, we determine _k_ using the predicted value of BP, which indicates the probability of a sample classified to be 1. For all _M ∈Y_ , we find the earliest step _k_<sup>_′_</sup> where the 95% steps _t_ in [ _k_<sup>_′_</sup> _, T_ ] satisfies: the margin between BP( _Mt_ , _t_ ) and BP( _M_<sup>�</sup> _t_ , _t_ ) is under the threshold. Then we choose the average of _k_<sup>_′_</sup> as the intersection boundary _k_ .

**Algorithm 1:** Training <u>procedure of DiffSinger.</u>

**Input:** The denoiser **_ϵ_** _θ_ ; the intersection boundary _k_ ; the training set ( _X , Y_ ). **1 repeat**

**2** Sample ( _x, M_ ) from ( _X , Y_ ); **3** **_ϵ_** _∼N_ ( **0** _,_ **I** ); **4** _t ∼_ Uniform( _{_ 1 _, . . . , k}_ ); **5** Take gradient descent step on **6** _∇θ_ �� **_ϵ_** _−_ **_ϵ_** _θ_ ( _√α_ <u>¯</u> _tM_ +<sup>_√_</sup> 1 _− α_ ¯ _t_ **_ϵ_** _, x, t_ )��2 **7 until** _convergence_ ;

We also propose an easier trick for boundary prediction in the supplement by comparing the KL-divergence. Note that the boundary prediction can be considered as a step of dataset preprocessing to choose the hyperparameter _k_ for the whole dataset. _k_ actually can be chosen manually by bruteforce searching on validation set.

## 3.3 Boundary Prediction

We propose a boundary predictor (BP) to locate the intersection in Figure 4 and determine _k_ adaptively. Concretely, BP consists of a classifier and a module for adding noise to mel-spectrograms according to Eq. (1). Given the step number _t ∈_ [0 _, T_ ], we label the _Mt_ as 1 and _M_<sup>�</sup> _t_ as 0, and use cross-entropy loss to train the boundary predictor to judge whether the input mel-spectrogram at _t_ step in diffusion process comes from _M_ or _M_<sup>�</sup> . The training loss L _BP_ can be

> 5Converting _Mk_ into _M_ 0 is easier than converting _MT_ (Gaussion white noise) into _M_ 0 ( _k < T_ ). Thus the former could improve the quality of synthesized audio and accelerates inference.

## 3.4 Model Structures

**Encoder** The encoder encodes the music score into the condition sequence, which consists of 1) a lyrics encoder to map the phoneme ID into embedding sequence, and a series of Transformer blocks (Vaswani et al. 2017) to convert this sequence into linguistic hidden sequence; 2) a length regulator to expand the linguistic hidden sequence to the length of mel-spectrograms according to the duration information; and 3) a pitch encoder to map the pitch ID into pitch embedding sequence. Finally, the encoder adds linguistic sequence and pitch sequence together as the music condition sequence _Em_ following (Ren et al. 2020).

11023

#### **Algorithm 2:** Inference procedure of DiffSinger.


**Step Embedding** The diffusion step _t_ is another conditional input for denoiser **_ϵ_** _θ_ , as shown in Eq. (6). To convert the discrete step _t_ to continuous hidden, we use the sinusoidal position embedding (Vaswani et al. 2017) followed by two linear layers to obtain step embedding _Et_ with _C_ channels.

**Auxiliary Decoder** We introduce a simple mel-spectrogram decoder called the auxiliary decoder, which is composed of stacked feed-forward Transformer (FFT) blocks and generates _M_<sup>�</sup> as the final outputs, the same as the mel-spectrogram decoder in FastSpeech 2 (Ren et al. 2021).

**Denoiser** Denoiser **_ϵ_** _θ_ takes in _Mt_ as input to predict **_ϵ_** added in diffusion process conditioned on the step embedding _Et_ and music condition sequence _Em_ . Since diffusion model imposes no architectural constraints (Sohl-Dickstein et al. 2015; Kong et al. 2021), the design of denoiser has multiple choices. We adopt a non-causal WaveNet (Oord et al. 2016) architecture proposed by (Rethage, Pons, and Serra 2018; Kong et al. 2021) as our denoiser. The denoiser is composed of a 1 _×_ 1 convolution layer to project _Mt_ with _Hm_ channels to the input hidden sequence _H_ with _C_ channels and _N_ convolution blocks with residual connections. Each convolution block consists of 1) an element-wise adding operation which adds _Et_ to _H_ ; 2) a non-causal convolution network which converts _H_ from _C_ to 2 _C_ channels; 3) a 1 _×_ 1 convolution layer which converts the _Em_ to 2 _C_ channels; 4) a gate unit to merge the information of input and conditions; and 5) a residual block to split the merged hidden into two branches with _C_ channels (the residual as the following _H_ and the “skip hidden” to be collected as the final results), which enables the denoiser to incorporate features at several hierarchical levels for final prediction.

**Boundary Predictor** The classifier in the boundary predictor is composed of 1) a step embedding to provide _Et_ ; 2) a ResNet (He et al. 2016) with stacked convolutional layers and a linear layer, which takes in the mel-spectrograms at _t_ -th step and _Et_ to classify _Mt_ and _M_<sup>�</sup> _t_ . More details of model structure and configurations are

shown in the supplement.

# 4 Experiments

In this section, we first describe the experimental setup, and then provide the main results on SVS with analysis. Finally, we conduct the extensional experiments on TTS.

## 4.1 Experimental Setup

**Dataset** Since there is no publicly available high-quality unaccompanied singing dataset, we collect and annotate a Chinese Mandarin pop songs dataset: PopCS, to evaluate our methods. PopCS contains 117 Chinese pop songs (total _∼_ 5.89 hours with lyrics) collected from a qualified female vocalist. All the audio files are recorded in a recording studio. Every song is sampled at 24kHz with 16-bit quantization. To obtain more accurate music scores corresponding to the songs (Lee et al. 2019), we 1) split each whole song into sentence pieces following DeepSinger (Ren et al. 2020) and train a Montreal Forced Aligner tool (MFA) (McAuliffe et al. 2017) model on those sentence-level pairs to obtain the phoneme-level alignments between song piece and its corresponding lyrics; 2) extract _F_ 0 (fundamental frequency) as pitch information from the raw waveform using Parselmouth, following (Wu and Luan 2020; Blaauw and Bonada 2020; Ren et al. 2020). We randomly choose 2 songs for validation and testing. To release a high-quality dataset, after the paper is accepted, we clean and re-segment these songs, resulting in 1,651 song pieces, which mostly last 10 _∼_ 13 seconds. The codes accompanied with the access to PopCS are in https://github.com/MoonInTheRiver/DiffSinger. In this repository, we also add the extra codes out of interest, for MIDI-to-Mel, including the MIDI-to-Mel without F0 prediction/condition.

**Implementation Details** We convert Chinese lyrics into phonemes by pypinyin following (Ren et al. 2020); and extract the mel-spectrogram (Shen et al. 2018) from the raw waveform; and set the hop size and frame size to 128 and 512 in respect of the sample rate 24kHz. The size of phoneme vocabulary is 61. The number of mel bins _Hm_ is 80. The mel-spectrograms are linearly scaled to the range [- 1, 1], and F0 is normalized to have zero mean and unit variance. In the lyrics encoder, the dimension of phoneme embeddings is 256 and the Transformer blocks have the same setting as that in FastSpeech 2 (Ren et al. 2021). In the pitch encoder, the size of the lookup table and encoded pitch embedding are set to 300 and 256. The channel size _C_ mentioned before is set to 256. In the denoiser, the number of convolution layers N is 20 with the kernel size 3, and we set the dilation to 1 (without dilation) at each layer<sup>6</sup> . We set _T_ to 100 and _β_ to constants increasing linearly from _β_ 1 = 10<sup>_−_4</sup> to _βT_ = 0 _._ 06. The auxiliary decoder has the same setting as the mel-spectrogram decoder in FastSpeech 2. In the boundary predictor, the number of convolutional layers is 5, and the threshold is set to 0.4 empirically.

6You can consider setting a bigger dilation number to increase the receptive field of the denoiser. See our Github repository.

11024


(a)  GT (b)  Diffsinger<br>


(c)  GAN-singer (d)  FFT-Singer<br>

Figure 5: Visualizations of mel-spectrograms in four systems: GT, DiffSinger, GAN-Singer and FFT-Singer.

**Training and Inference** The training has two stages: 1) warmup stage: separately train the auxiliary decoder for 160k steps with the music score encoder, and then leverage the auxiliary decoder to train the boundary predictor for 30k steps to obtain _k_ ; 2) main stage: training DiffSinger as Algorithm 1 describes for 160k steps until convergence. In the inference stage, for all SVS experiments, we uniformly use a pretrained Parallel WaveGAN (PWG) (Yamamoto, Song, and Kim 2020)<sup>7</sup> as vocoder to transform the generated mel-spectrograms into waveforms (audio samples).

## 4.2 Main Results and Analysis

**Audio Performance** To evaluate the perceptual audio quality, we conduct the MOS (mean opinion score) evaluation on the test set. Eighteen qualified listeners are asked to make judgments about the synthesized song samples. We compare MOS of the song samples generated by DiffSinger with the following systems: 1) _GT_ , the ground truth singing audio; 2) _GT (Mel + PWG)_ , where we first convert the ground truth singing audio to the ground truth mel-spectrograms, and then convert these mel-spectrograms back to audio using PWG vocoder described in Section 4.1; 3) _FFT-NPSS (Blaauw and Bonada 2020) (WORLD)_ , the SVS system which generates WORLD vocoder features (Morise, Yokomori, and Ozawa 2016) through feedforward Transformer (FFT) and uses WORLD vocoder to synthesize audio; 4) _FFT-Singer (Mel + PWG)_ the SVS system which generates mel-spectrograms through FFT network and uses PWG vocoder to synthesize audio; 5) _GANSinger (Wu and Luan 2020) (Mel + PWG)_ , the SVS system with adversarial training using multiple random window discriminators.

The results are shown in Table 1. The quality of _GT (MEL + PWG)_ (4.04 _±_ 0.11) is the upper limit of the acoustic model for SVS. _DiffSinger_ outperforms the baseline system with simple training loss ( _FFT-Singer_ ) by a large margin, and shows the superiority compared with the state-of-theart GAN-based method ( _GAN-Singer_ (Wu and Luan 2020)), which demonstrate the effectiveness of our method.

As shown in Figure 5, we compare the ground truth, the generated mel-spectrograms from _Diffsinger_ , _GAN-singer_ and _FFT-Singer_ with the same music score. It can be seen that both Figure 5c and Figure 5b contain more delicate details between harmonics than Figure 5d does. Moreover, the

> 7We adjust PWG to take in F0 driven source excitation (Wang and Yamagishi 2020) as additional condition, similar to that in (Chen et al. 2020).

|Method|MOS|
|---|---|
|_GT_|4.30_±_0.09|
|_GT (Mel + PWG)_|4.04_±_0.11|
|_FFT-NPSS (WORLD)_|1.75_±_0.17|
|_FFT-Singer (Mel + PWG)_|3.67_±_0.11|
|_GAN-Singer (Mel + PWG)_|3.74_±_0.12|
|_DiffSinger Naive (Mel + PWG)_|3.71_±_0.10|
|_DiffSinger (Mel + PWG)_|**3.85**_±_0.11|


Table 1: The MOS with 95% confidence intervals of song samples. DiffSinger Naive means the naive version of DiffSinger without shallow diffusion mechanism.

performance of _Diffsinger_ in the region of mid or low frequency is more competitive than that of _GAN-singer_ while maintaining similar quality of the high-frequency region.

In the meanwhile, the shallow diffusion mechanism accelerates inference of naive diffusion model by 45.1% (RTF 0.191 vs. 0.348, RTF is the real-time factor, that is the seconds it takes to generate one second of audio).

**Ablation Studies** We conduct ablation studies to demonstrate the effectiveness of our proposed methods and some hyper-parameters studies to seek the best model configurations. We conduct CMOS evaluation for these experiments. The results of variations on _DiffSinger_ are listed in Table 2. It can be seen that: 1) removing the shallow diffusion mechanism results in quality drop (-0.500 CMOS), which is consistent with the MOS test results and verifies the effectiveness of our shallow diffusion mechanism (row 1 vs. row 2); 2) adopting other _k_ (row 1 vs. row 3) rather than the one predicted by our boundary predictor causes quality drop, which verifies that our boundary prediction network can predict a proper _k_ for shallow diffusion mechanism; and 3) the model with configurations _C_ = 256 and _L_ = 20 produces the best results (row 1 vs. row 4,5,6,7), indicating that our model capacity is sufficient.

## 4.3 Extensional Experiments on TTS

To verify the generalization of our methods on TTS task, we conduct the extensional experiments on LJSpeech dataset (Ito and Johnson 2017), which contains 13,100 English audio clips (total _∼_ 24 hours) with corresponding transcripts. We follow the train-val-test dataset splits, the pre-processing of mel-spectrograms, and the grapheme-tophoneme tool in FastSpeech 2. To build _DiffSpeech_ , we 1)

11025

|No.|_C_|_L_|w/ shallow|_k_|CMOS|
|---|---|---|---|---|---|
|1|256|20|✓|54|0.000|
|2|256|20|_×_|-|-0.500|
|3|256|20|✓|25|-0.053|
|4|128|20|✓|54|-0.071|
|5|512|20|✓|54|-0.044|
|6|256|10|✓|54|-0.293|
|7|256|30|✓|54|-0.445|


Table 2: Variations on the _DiffSinger_ . _T_ in all the experiments is set to 100. _C_ is channel size; _L_ is the number of layers in denoiser; w/ shallow means “with shallow diffusion mechanism”; _k_ = 54 is our predicted intersection boundary.

|Method|MOS|
|---|---|
|_GT_|4.22_±_0.07|
|_GT (Mel + HiFi-GAN)_|4.15_±_0.07|
|_Tacotron 2 (Mel + HiFi-GAN)_|3.54_±_0.05|
|_BVAE-TTS (Mel + HiFi-GAN)_|3.48_±_0.06|
|_FastSpeech 2 (Mel + HiFi-GAN)_|3.68_±_0.06|
|_Glow-TTS (Mel + HiFi-GAN)_|3.69_±_0.07|
|_DiffSpeech Naive (Mel + HiFi-GAN)_|3.69_±_0.05|
|_DiffSpeech (Mel + HiFi-GAN)_|**3.92**_±_0.06|


Table 3: The MOS of speech samples with 95% confidence intervals.

add a pitch predictor and a duration predictor to _DiffSinger_ as those in FastSpeech 2; 2) adopt _k_ = 70 for shallow diffusion mechanism.

We use Amazon Mechanical Turk (ten testers) to make subjective evaluation and the results are shown in Table 3. All the systems adopt HiFi-GAN (Kong, Kim, and Bae 2020) as vocoder. _DiffSpeech_ outperforms FastSpeech 2 and Glow-TTS, which demonstrates the generalization. Besides, the last two rows in Table 3 also show the effectiveness of shallow diffusion mechanism (with 29.2% speedup, RTF 0.121 vs. 0.171).

# 5 Related Work

## 5.1 Singing Voice Synthesis

Initial works of singing voice synthesis generate the sounds using concatenated (Macon et al. 1997; Kenmochi and Ohshita 2007) or HMM-based parametric (Saino et al. 2006; Oura et al. 2010) methods, which are kind of cumbersome and lack flexibility and harmony. Thanks to the rapid evolution of deep learning, several SVS systems based on deep neural networks have been proposed in the past few years. Nishimura et al. (2016); Blaauw and Bonada (2017); Kim et al. (2018); Nakamura et al. (2019); Gu et al. (2020) utilize neural networks to map the contextual features to acoustic features. Ren et al. (2020) build the SVS system from scratch using singing data mined from music websites. Blaauw and Bonada (2020) propose a feed-forward Trans-

former SVS model for fast inference and avoiding exposure bias issues caused by autoregressive models. Besides, with the help of adversarial training, Lee et al. (2019) propose an end-to-end framework which directly generates linear-spectrograms. Wu and Luan (2020) present a multi-singer SVS system with limited available recordings and improve the voice quality by adding multiple random window discriminators. Chen et al. (2020) introduce multi-scale adversarial training to synthesize singing with a high sampling rate (48kHz). The voice naturalness and diversity of SVS system have been continuously improved in recent years.

## 5.2 Denoising Diffusion Probabilistic Models

A diffusion probabilistic model is a parameterized Markov chain trained by optimizing variational lower bound, which generates samples matching the data distribution in constant steps (Ho, Jain, and Abbeel 2020). Diffusion model is first proposed by Sohl-Dickstein et al. (2015). Ho, Jain, and Abbeel (2020) make progress of diffusion model to generate high-quality images using a certain parameterization and reveal an equivalence between diffusion model and denoising score matching (Song and Ermon 2019; Song et al. 2021). Recently, Kong et al. (2021) and Chen et al. (2021) apply the diffusion model to neural vocoders, which generate high-fidelity waveform conditioned on mel-spectrogram. Chen et al. (2021) also propose a continuous noise schedule to reduce the inference iterations while maintaining synthesis quality. Song, Meng, and Ermon (2021) extend diffusion model by providing a faster sampling mechanism, and a way to interpolate between samples meaningfully. Diffusion model is a fresh and developing technique, which has been applied in the fields of unconditional image generation, conditional spectrogram-to-waveform generation (neural vocoder). And in our work, we propose a diffusion model for the acoustic model which generates mel-spectrogram given music scores (or text). There is a concurrent work (Jeong et al. 2021) at the submission time of our preprint which adopts a diffusion model as the acoustic model for TTS task.

# 6 Conclusion

In this work, we proposed DiffSinger, an acoustic model for SVS based on diffusion probabilistic model. To improve the voice quality and speed up inference, we proposed a shallow diffusion mechanism. Specifically, we found that the diffusion trajectories of _M_ and _M_<sup>�</sup> converge together when the diffusion step is big enough. Inspired by this, we started the reverse process at the intersection (step _k_ ) of two trajectories rather than at the very deep diffusion step _T_ . Thus the burden of the reverse process could be distinctly alleviated, which improves the quality of synthesized audio and accelerates inference. The experiments conducted on PopCS demonstrate the superiority of DiffSinger compared with previous works, and the effectiveness of our novel shallow diffusion mechanism. The extensional experiments conducted on LJSpeech dataset prove the effectiveness of DiffSpeech on TTS task. The directly synthesis without vocoder will be future work.

11026

## Acknowledgments

This work was supported in part by the National Key R&D Program of China under Grant No.2020YFC0832505, No.62072397, Zhejiang Natural Science Foundation under Grant LR19F020006. Thanks participants of the listening test for the valuable evaluations.

# References

Blaauw, M.; and Bonada, J. 2017. A neural parametric singing synthesizer modeling timbre and expression from natural songs. _Applied Sciences_ , 7(12): 1313.

Blaauw, M.; and Bonada, J. 2020. Sequence-to-sequence singing synthesis using the feed-forward transformer. In _ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , 7229– 7233. IEEE.

Chen, J.; Tan, X.; Luan, J.; Qin, T.; and Liu, T.-Y. 2020. HiFiSinger: Towards High-Fidelity Neural Singing Voice Synthesis. _arXiv preprint arXiv:2009.01776_ .

Chen, N.; Zhang, Y.; Zen, H.; Weiss, R. J.; Norouzi, M.; and Chan, W. 2021. WaveGrad: Estimating Gradients for Waveform Generation. In _International Conference on Learning Representations_ .

Gu, Y.; Yin, X.; Rao, Y.; Wan, Y.; Tang, B.; Zhang, Y.; Chen, J.; Wang, Y.; and Ma, Z. 2020. ByteSing: A Chinese Singing Voice Synthesis System Using Duration Allocated EncoderDecoder Acoustic Models and WaveRNN Vocoders. _arXiv preprint arXiv:2004.11012_ .

He, K.; Zhang, X.; Ren, S.; and Sun, J. 2016. Deep residual learning for image recognition. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , 770–778.

Ho, J.; Jain, A.; and Abbeel, P. 2020. Denoising Diffusion Probabilistic Models. In Larochelle, H.; Ranzato, M.; Hadsell, R.; Balcan, M. F.; and Lin, H., eds., _Advances in Neural Information Processing Systems_ , volume 33, 6832–6843. Curran Associates, Inc.

Ito, K.; and Johnson, L. 2017. The LJ Speech Dataset. https: //keithito.com/LJ-Speech-Dataset/. Accessed: 2019-10-12.

Jeong, M.; Kim, H.; Cheon, S. J.; Choi, B. J.; and Kim, N. S. 2021. Diff-tts: A denoising diffusion model for text-to-speech. _arXiv preprint arXiv:2104.01409_ .

Kenmochi, H.; and Ohshita, H. 2007. Vocaloid-commercial singing synthesizer based on sample concatenation. In _Eighth Annual Conference of the International Speech Communication Association_ .

Kim, J.; Choi, H.; Park, J.; Kim, S.; Kim, J.; and Hahn, M. 2018. Korean Singing Voice Synthesis System based on an LSTM Recurrent Neural Network. In _INTERSPEECH 2018_ . ISCA.

Kim, J.; Kim, S.; Kong, J.; and Yoon, S. 2020. Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search. _Advances in Neural Information Processing Systems_ , 33.

Kong, J.; Kim, J.; and Bae, J. 2020. HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis. _Advances in Neural Information Processing Systems_ , 33.

Kong, Z.; Ping, W.; Huang, J.; Zhao, K.; and Catanzaro, B. 2021. DiffWave: A Versatile Diffusion Model for Audio Synthesis. In _International Conference on Learning Representations_ .

Lee, J.; Choi, H.-S.; Jeon, C.-B.; Koo, J.; and Lee, K. 2019. Adversarially Trained End-to-End Korean Singing Voice Synthesis System. _Proc. Interspeech 2019_ , 2588–2592.

Macon, M.; Jensen-Link, L.; George, E. B.; Oliverio, J.; and Clements, M. 1997. Concatenation-based MIDI-to-singing voice synthesis. In _Audio Engineering Society Convention 103_ . Audio Engineering Society.

McAuliffe, M.; Socolof, M.; Mihuc, S.; Wagner, M.; and Sonderegger, M. 2017. Montreal Forced Aligner: Trainable Text-Speech Alignment Using Kaldi. In _Interspeech_ , 498– 502.

Morise, M.; Yokomori, F.; and Ozawa, K. 2016. WORLD: a vocoder-based high-quality speech synthesis system for real-time applications. _IEICE TRANSACTIONS on Information and Systems_ , 99(7): 1877–1884.

Nakamura, K.; Hashimoto, K.; Oura, K.; Nankaku, Y.; and Tokuda, K. 2019. Singing voice synthesis based on convolutional neural networks. _arXiv preprint arXiv:1904.06868_ . Nichol, A. Q.; and Dhariwal, P. 2021. Improved denoising diffusion probabilistic models. In _International Conference on Machine Learning_ , 8162–8171. PMLR.

Nishimura, M.; Hashimoto, K.; Oura, K.; Nankaku, Y.; and Tokuda, K. 2016. Singing Voice Synthesis Based on Deep Neural Networks. In _Interspeech_ , 2478–2482.

Oord, A. v. d.; Dieleman, S.; Zen, H.; Simonyan, K.; Vinyals, O.; Graves, A.; Kalchbrenner, N.; Senior, A.; and Kavukcuoglu, K. 2016. WaveNet: A Generative Model for Raw Audio. In _9th ISCA Speech Synthesis Workshop_ , 125– 125.

Oura, K.; Mase, A.; Yamada, T.; Muto, S.; Nankaku, Y.; and Tokuda, K. 2010. Recent development of the HMM-based singing voice synthesis system—Sinsy. In _Seventh ISCA Workshop on Speech Synthesis_ .

Ren, Y.; Hu, C.; Tan, X.; Qin, T.; Zhao, S.; Zhao, Z.; and Liu, T.-Y. 2021. FastSpeech 2: Fast and High-Quality End-toEnd Text to Speech. In _International Conference on Learning Representations_ .

Ren, Y.; Tan, X.; Qin, T.; Luan, J.; Zhao, Z.; and Liu, T.-Y. 2020. Deepsinger: Singing voice synthesis with data mined from the web. In _Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining_ , 1979–1989.

Rethage, D.; Pons, J.; and Serra, X. 2018. A wavenet for speech denoising. In _2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , 5069–5073. IEEE.

Saino, K.; Zen, H.; Nankaku, Y.; Lee, A.; and Tokuda, K. 2006. An HMM-based singing voice synthesis system. In

11027

_Ninth International Conference on Spoken Language Processing_ .

Shen, J.; Pang, R.; Weiss, R. J.; Schuster, M.; Jaitly, N.; Yang, Z.; Chen, Z.; Zhang, Y.; Wang, Y.; Skerrv-Ryan, R.; et al. 2018. Natural tts synthesis by conditioning wavenet on mel spectrogram predictions. In _ICASSP 2018_ , 4779–4783. IEEE.

Sohl-Dickstein, J.; Weiss, E.; Maheswaranathan, N.; and Ganguli, S. 2015. Deep Unsupervised Learning using Nonequilibrium Thermodynamics. In _International Conference on Machine Learning_ , 2256–2265. Song, J.; Meng, C.; and Ermon, S. 2021. Denoising Diffusion Implicit Models. In _International Conference on Learning Representations_ .

Song, Y.; and Ermon, S. 2019. Generative Modeling by Estimating Gradients of the Data Distribution. In _Proceedings of the 33rd Annual Conference on Neural Information Processing Systems_ .

Song, Y.; Sohl-Dickstein, J.; Kingma, D. P.; Kumar, A.; Ermon, S.; and Poole, B. 2021. Score-Based Generative Modeling through Stochastic Differential Equations. In _International Conference on Learning Representations_ .

Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A. N.; Kaiser, Ł.; and Polosukhin, I. 2017. Attention is all you need. In _Advances in Neural Information Processing Systems_ , 5998–6008.

Vincent, P. 2011. A Connection Between Score Matching and Denoising Autoencoders. In _Neural Computation_ . Wang, X.; and Yamagishi, J. 2020. Using Cyclic Noise as the Source Signal for Neural Source-Filter-Based Speech Waveform Model. _Proc. Interspeech 2020_ , 1992–1996.

Wu, J.; and Luan, J. 2020. Adversarially Trained MultiSinger Sequence-to-Sequence Singing Synthesizer. _Proc. Interspeech 2020_ , 1296–1300.

Yamamoto, R.; Song, E.; and Kim, J.-M. 2020. Parallel WaveGAN: A fast waveform generation model based on generative adversarial networks with multi-resolution spectrogram. In _ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , 6199–6203. IEEE.

Zhang, L.; Yu, C.; Lu, H.; Weng, C.; Zhang, C.; Wu, Y.; Xie, X.; Li, Z.; and Yu, D. 2020. DurIAN-SC: Duration Informed Attention Network Based Singing Voice Conversion System. _Proc. Interspeech 2020_ , 1231–1235.

11028
