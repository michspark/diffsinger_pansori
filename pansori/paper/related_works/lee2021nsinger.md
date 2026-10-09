---
title: "N-Singer: A Non-Autoregressive Korean Singing Voice Synthesis System for Pronunciation Enhancement"
authors: "Gyeong-Hoon Lee; Tae-Woo Kim; Hanbin Bae; Min-Ji Lee; Young-Ik Kim; Hoon-Young Cho"
venue: "Proc. Interspeech"
year: 2021
bib_key: lee2021nsinger
source_url: "https://www.isca-archive.org/interspeech_2021/lee21b_interspeech.pdf"
version: "published"
cited_in_paper: true
relevance: "Korean parallel SVS with explicit linguistic/musical separation and voicing-aware discriminators."
---
## Why it matters

Korean parallel SVS with explicit linguistic/musical separation and voicing-aware discriminators.


_INTERSPEECH 2021_ 30 August – 3 September, 2021, Brno, Czechia


# N-Singer: A Non-Autoregressive Korean Singing Voice Synthesis System for Pronunciation Enhancement

_Gyeong-Hoon Lee, Tae-Woo Kim, Hanbin Bae, Min-Ji Lee, Young-Ik Kim, Hoon-Young Cho_

Speech AI Lab., AI Center, NCSOFT, Republic of Korea

_{_ ghlee0304, ktw0114, bhb0722, leeminxji, youngik, hycho _}_ @ncsoft.com

## Abstract

Recently, end-to-end Korean singing voice systems have been designed to generate realistic singing voices. However, these systems still suffer from a lack of robustness in terms of pronunciation accuracy. In this paper, we propose N-Singer, a nonautoregressive Korean singing voice system, to synthesize accurate and pronounced Korean singing voices in parallel. N- Singer consists of a Transformer-based mel-generator, a convolutional network-based postnet, and voicing-aware discriminators. It can contribute in the following ways. First, for accurate pronunciation, N-Singer separately models linguistic and pitch information without other acoustic features. Second, to achieve improved mel-spectrograms, N-Singer uses a combination of Transformer-based modules and convolutional network-based modules. Third, in adversarial training, voicing-aware conditional discriminators are used to capture the harmonic features of voiced segments and noise components of unvoiced segments. The experimental results prove that N-Singer can synthesize a natural singing voice in parallel with a more accurate pronunciation than the baseline model. **Index Terms** : Singing voice synthesis, Non-autoregressive, Transformer, FastSpeech, Parallel WaveGAN

# 1. Introduction

The singing voice synthesis (SVS) system is a generative model that synthesizes singing voice from text and musical information such as lyrics, beat, note pitch and length. Recently, Korean SVS systems [1–3] have attracted significant attention with the increase of Korean pop music (K-pop) popularity. To minimize the difference between real and synthesized singing voices, SVS systems should consider that synthesized singing voice must have not only accurate pronunciation, but also musical factors of the corresponding musical score. However, it is difficult to modeling linguistic and musical features simultaneously because singing voices generally have a longer vowel duration and a wider range of pitch than speech [4].

Previous deep-learning-based SVS systems adopted the architecture of autoregressive speech synthesis systems [5–7] to synthesize a more realistic singing voice. Autoregressive neural SVS systems are designed to model text and musical information using auxiliary modules, such as pitch encoder [2] and duration module [8], or feature composing method between linguistic and musical features [1]. Although autoregressive SVS systems outperform conventional parametric models [9–11], these systems lack robustness; i.e., there are word skipping and repetition problems in synthesized singing voices because of incorrect attention alignment.

The adversarially trained end-to-end Korean SVS system (ATK) [2] has shown the best performance among Korean SVS systems [1–3]. ATK is an autoregressive SVS system based on convolution [7]. To generate a more accurate and pronounced

singing voice, the mel-generator of ATK is designed to independently model the pronunciation information from text in an unsupervised manner using the phonetic enhancement mask decoder. In addition, ATK uses a conditional generative adversarial network [12] with a projection discriminator [13] to achieve cleaner linear spectrograms. However, it was observed that the ATK’s synthesized singing voice pronunciation was inaccurate, and short phonemes were often skipped because convolution, which can only capture local features, caused incorrect attention alignment. Additionally, because ATK predicts the linear spectrogram, the complexity of the generator is too high, and the synthesis speed is slow.

To overcome the limitations of autoregressive generative models, non-autoregressive SVS systems have been proposed to synthesize singing in parallel [4, 14–16]. In the non-autoregressive SVS systems, acoustic models can synthesize acoustic features such as mel-spectrogram faster and in a more robust manner than autoregressive systems using a Transformer-based on self-attention network, which has the advantage of capturing long-term dependencies in the input sequence [17, 18]. A length regulator and duration predictor were adopted so that each phoneme sequence matched the time length of the mel-spectrogram. However, in our preliminary experiments on Korean singing, we found that the pronunciation of singing voice generated by a previous SVS model is not accurate, given a pitch sequence with dynamic changes.

In this paper, we propose N-Singer, a non-autoregressive Korean SVS system for synthesizing singing voices with more accurate pronunciation. Our contributions are listed below. First, for a singing voice that articulates pronunciation better, we independently model the linguistic features and musical features in mel-spectrograms by using two feed-forward Transformer (FFT)-based encoder-decoder modules. Second, to improve the quality of the synthesized mel-spectrograms, we jointly train the convolutional neural network (CNN)-based postnet and Transformer-based generator. Third, we use the GAN framework with voicing-aware conditional discriminators using pitch, voiced and unvoiced (V/UV) flag to efficiently train the acoustic model. Two discriminators deal with each voiced and unvoiced segment of a singing voice, effectively improving the performance of the generator. N-Singer uses Parallel WaveGAN (PWG) [19] as a vocoder to synthesize a singing voice waveform in parallel. Experimental results show that N-Singer can synthesize mel-spectrograms with higher pronunciation accuracy and better sound quality, and is 36.7 times faster than the ATK on an NVIDIA V100 GPU.

# 2. Proposed Method

Our proposed Korean SVS system, N-Singer, consists of a melgenerator and postnet as shown in Figure 1. The mel-generator is composed of two encoder–decoder modules based on a Trans-


**http://dx.doi.org/10.21437/Interspeech.2021-239**


Figure 1: _(a) is a whole structure of N-Singer, (b) is a postnet of N-Singer, and (c) is a voicing-aware discriminator of N-Singer_

former, and each module uses lyrics and note pitch information independently. The CNN-based postnet generates an enhanced mel-spectrogram from a coarse mel-spectrogram using phoneme and pitch encoder outputs as local conditions.

## 2.1. Input representation

We converted the grapheme sequences in Korean lyrics to phoneme sequences using the Korean grapheme-to-phoneme algorithm. We rearranged the pitch sequence to manually match one Korean syllable with a single note pitch. Next, we expanded the phoneme and pitch sequences to be the same time length as the mel-spectrogram _M ∈_ R<sup>_N×L_</sup> by using the note duration in the musical score, where _N_ is the number of bins and _L_ is the number of frames of the mel-spectrogram. Generally, Korean syllables are composed of three components: onset, nucleus, and coda. In the aligned phoneme sequence _T ∈_ R<sup>1</sup><sup>_×L_</sup> , we assigned onset and coda to a maximum of three frames and assigned the remaining frames to the nucleus because singing voices have a long vowel duration and the nucleus of a Korean syllable corresponds to the vowel. We converted each note pitch in the musical score into a pitch ID according to the standard musical instrument digital interface (MIDI) standard [20]. In the aligned pitch sequence _P ∈_ R<sup>1</sup><sup>_×L_</sup> , the pitch corresponding to each syllable was repeatedly assigned to the frames to fit the syllable length. We referred to [2] for this method.

## 2.2. Mel-generator

The N-Singer’s mel-generator consists of two encoder–decoder modules based on FastSpeech [18]. In the encoder, the input sequence is embedded into a dense vector, added with positional encoding, and then passed through the FFT blocks. The FFT block consists of a self-attention network with multihead attention and a 1D convolutional network [15, 18]. In the decoder, positional encoding is added to the encoded sequence and passed through the FFT block, passing through a linear layer with sigmoid activation. During training, the output _DT ∈_ R<sup>_N×L_</sup> of the phoneme encoder-decoder contained the linguistic features related to pronunciation from the phoneme sequence _T ∈_ R<sup>1</sup><sup>_×L_</sup> , while the pitch encoder–decoder’s output _DP ∈_ R<sup>_N×L_</sup> contained features related to F0 and harmonics from the pitch sequence _P ∈_ R<sup>1</sup><sup>_×L_</sup> . Because the mel-

generator can model pronunciation and harmonics separately, the phoneme encoder–decoder is not distorted by any input pitch sequence and can generate a mel-spectrogram containing more accurate pronunciation information. The output mel-spectrogram _MD ∈_ R<sup>_N×L_</sup> is generated by the element-wise product of _DT_ and _DP_ as follows [2]:


where _MG_ ( _·_ ) indicate mel-generator. The mel-generator was trained with _L_ 1 and binary divergence loss _Lbd_ [2, 7] between the ground truth mel-spectrogram _M ∈_ R<sup>_N×L_</sup> and the predicted mel-spectrogram _MD_ . Additionally, we used an auxiliary loss to extract the text and pitch information included in the mel-spectrograms. The auxiliary loss is the sum of _L_ 1 losses between _M_ and _DP_ and between _M_ and _DT_ . We experimentally found that the mel-generator could not model by separating linguistic features and harmonic features without auxiliary loss, and this loss had an effect at the beginning of training. Therefore, we reduced the weight of this auxiliary loss by multiplying the decay rate _λinit_ during training. The total loss _Lmg_ for the mel-generator can be calculated as follows:

_Lmd_ ( _M, MD_ ) = _L_ 1 ( _M, MD_ ) + _Lbd_ ( _M, MD_ ) _Linit_ ( _M, DP , DT_ ) = ( _L_ 1 ( _M, DP_ ) + _L_ 1 ( _M, DT_ )) _/_ 2 (2) _Lmg_ = _Lmd_ + _Linit × λ_<sup>(training stpes)</sup> _init_

## 2.3. Postnet

Recent studies have shown that models with a combination of Transformer and CNN modules outperformed models that use them individually [21, 22]. A Transformer-based on selfattention captures the global context well, but it lacks the ability to extract local features. CNN-based models can efficiently capture local feature patterns, but many layers are needed to obtain global information. Because the adjacent frames are more related in the mel-spectrogram, we designed a CNN-based postnet to capture the local features and generate a more realistic mel-spectrogram from the coarse mel-spectrogram. Figure 2(a) depicts the architecture of the proposed postnet. The postnet of N-Singer, inspired by the ATK’s super-resolution network [2], consists of a 1 _×_ 1 convolutional network, dropout [23], and highway network block that consists of highway networks [24],


convolutions, and exponential linear units (ELU) [25]. To reflect the input text and pitch information to the generated mel-spectrogram, the postnet uses encoded phoneme _ET_ and pitch sequence _EP_ of the intermediate outputs of encoders as local condition inputs using the local conditioning method as follows [2]:


where, _U∗, V∗, W∗_ are convolutional layers, _φ_ is ELU activation function and _σ_ is sigmoid activation function. We used _L_ 1 and _Lbd_ between the ground-truth mel-spectrogram _M_ , and generated mel-spectrogram _MP_ . We jointly trained the Transformerbased mel-generator and CNN-based postnet using _Lmp_ in (4) to take advantage of the Transformer and CNN.


## 2.4. Discriminator

Because pronunciation is related to voiced segments in singing voices, it is important that the mel-generator predicts the voiced and unvoiced segments in the mel-spectrogram. The voiced segments are periodic and have clean harmonic components that change slowly; while the unvoiced segments are aperiodic and have noise components that change rapidly. Because the voiced and unvoiced segments’ characteristics in singing voice are different, we adopted a voicing-aware conditional GAN with projection discriminators [26]. Our discriminators consisted of 1D and 2D convolutions with residual connections. To capture the harmonic structure in the voiced region of the mel-spectrogram, the first discriminator had residual convolutional layers with a large kernel size to increase the receptive field. Note that the model needs large receptive fields to deal with the long-term temporal dependencies of the harmonic structure in a voiced segment. The second discriminator for the unvoiced region was composed of residual convolutional layers with a small kernel size to capture the noise components. Figure 1(c) shows that to provide conditional information into the model, our discriminator took the inner product between the embedded acoustic feature vector converted by 1D convolutional layers and the intermediate output generated from the input mel-spectrogram. Additionally, a sigmoid activation was placed at the end to prevent extreme increases in output and to train the model stably. We took the V/UV mask from the final output such that the value corresponding to the voiced or unvoiced segment was the output. We use loss functions for adversarial training as follows:


where _D_<sup>_v_</sup> and _D_<sup>_uv_</sup> denote the voiced and unvoiced discriminators, _G_ indicate parameters of the generator, **x** is the input of generator, **y** is a mel-spectrogram from real data, and **c** denotes the conditional acoustic feature. Our final training objectives _LG_ and _LD_ for generator and discriminator are follows:


where _λp_ and _λadv_ are hyperparameters that need to be adjusted in the experiment.

# 3. Experiments

## 3.1. Experimental Setup

### _3.1.1. Datasets_

Our singing voice dataset consisted of 50 Korean pop songs and MIDI files collected from a professional female singer. All recordings were sampled at 48 kHz with 16-bit quantization and then down-sampled to 24 kHz. All songs were split into 1,083 clips, each being 5 to 10 seconds long. We randomly chose 47 songs with 983 clips for training and the remaining three songs with 100 clips for the test. For acoustic features, we extracted F0 sequences and a 1D V/UV (0-1 value) flags from real audio samples using an open-source WORLD vocoder. Mel-spectrograms with 80 bins were extracted by short-time Fourier transformation (STFT) using FFT size, window size, and frame shifts of 2,048, 1,200, and 240, respectively.

### _3.1.2. Model Configuration_

The proposed N-Singer system has two encoder and decoder modules. The encoder and decoder consist of six FFT blocks. The dimensions of the phoneme and pitch embeddings, the selfattention’s hidden size, and 1D convolution in the FFT block are all set to 256. The number of attention heads is set to 2. The kernel sizes of the two 1D convolution layers in the FFT block are both set to 13, with an input/output size of 256/1024 for the first layer and 1024/256 for the second layer. The output linear layer converts the 256-dimensional output into an 80-dimensional mel-spectrogram. The postnet in N-Singer consisted of a 1D convolutional layer and a highway network block. The highway network block consisted of four highway network layers with hidden size/kernel size set to 256/3, dropout with a rate of 0.05, and a 1D convolutional layer with ELU activation.

The ATK’s super-resolution network and discriminator were modified to predict the mel-spectrograms. The output channel of the last convolutional layer of the super-resolution network was set to 80, and the kernel sizes of the two 2D convolutional layers in the discriminator were set to 7 _×_ 1/3 _×_ 1.

Parallel WaveGAN with voicing-aware conditional discriminators [26] is used to synthesize the singing voice waveform as a vocoder. Particularly, because the voiced segment length of the singing voice is longer than that of speech, the kernel size of each 1D convolutional layer in the voiced discriminator is set to 13 to have a large receptive field.

### _3.1.3. Training_

The three modules of N-Singer, mel-generator, postnet, and discriminator, were trained jointly, while the vocoder was trained separately. The N-Singer model was trained for 100,000 steps with a minibatch size of 8 using the Adam optimizer [27] with _β_ 1 = 0 _._ 9, _β_ 2 = 0 _._ 999, and _ϵ_ = 10<sup>_−_6</sup> . The initial learning rate was set to 2 _×_ 10<sup>_−_4</sup> and was reduced by half for every 50,000 steps. The decay rate in (2) was set to 0 _._ 999. To warm up the generator, we set the hyperparameters in (6) as follows [2]:


The generator of the ATK model was trained for 100,000 steps with a minibatch size of 8 using the Adam optimizer with the same hyperparameters and learning rate decay method as in [2]. The vocoder was trained for 600,000 steps using a RAdam [28] optimizer with the same hyperparameters and training method as in [26] on NVIDIA V100 GPU.


Table 1: _MOS scores_

||**ATK**|**N-Singer**|**GT**|
|---|---|---|---|
|Pronun.acc|3.72_±_0.16|4.23_±_0.14|4.40_±_0.14|
|Sound.quality|3.80_±_0.16|3.85_±_0.16|3.85_±_0.16|


Table 2: _Quantitative evaluation results_

|**Metric**|**ATK**|**N-Singer**|
|---|---|---|
|Accuracy|0.886|**0.889**|
|Precision|0.905|**0.914**|
|Recall|0.884|0.884|
|F1-score|0.887|**0.893**|
|F0 CORR|0.984|**0.985**|
|F0 RMSE|8.959|**8.870**|
|V/UV Error (%)|10.231|**9.483**|


## 3.2. Evaluation

In this experiment, we compared the performance of our proposed model N-Singer, quantitatively and qualitatively, with that of the baseline ATK. Some samples are available online<sup>1</sup> .

### _3.2.1. Qualitative evaluation_

We conducted a mean opinion score (MOS) listening test as qualitative evaluation. We randomly selected 20 lyrics and MIDI pairs. Thirty native Korean participants evaluated synthesized singing voices based on the criteria for pronunciation accuracy and sound quality. For accurate pronunciation evaluation, participants first listened to the audio samples, looked at the lyrics, and then evaluated the samples. The MOS results, presented in Table 1, show that N-Singer can synthesize a singing voice with a more accurate pronunciation and higher sound quality than ATK. To prove that N-Singer is superior to ATK, we conducted the Wilcoxon [29] signed-rank test and win-draw-loss (w-d-l) comparisons. The w-d-l comparisons counted the number of test samples that N-Singer achieved, in terms of better, equal, or worse pronunciation accuracy as compared to ATK. In the Wilcoxon signed-rank test, the p-value was less than 0.05, and the w-d-l records were obtained as 29-1-0. These results indicate that the difference in the MOS results between the two models is statistically significant. Although MOS sound quality is dependent on the performance of the neural vocoder, it shows that the quality of the N-Singer’s mel-spectrograms is similar to that of the real mel-spectrograms.

### _3.2.2. Quantitative evaluation_

We conducted an evaluation to measure whether the model reflected the input pitch of the musical score to synthesize singing. We extracted F0 sequences from the synthesized samples using the WORLD [30] vocoder. The F0 sequences were converted to note pitch sequences and then compared with the true note pitch sequences (see Table 2 for the results). For all metrics, the N- Singer’s value was higher than that of ATK. This result shows that N-Singer synthesizes singing by accurately reflecting the given MIDI note. To further evaluate the synthesized singing voice quality, we calculated the F0 root mean square error (F0 RMSE), F0 correlation (F0 CORR), and voiced/unvoiced error rate (V/UV error) between the ground-truth and the synthesized singing voices. Table 2 shows that N-Singer achieves a higher


Figure 2: _F0 contour comparison for N-Singer and ATK_


Figure 3: _Intermediate outputs of N-Singer (left), ATK (right)_

performance than ATK. It was observed that the V/UV error of N-Singer was significantly less than that of ATK. To compare the V/UV decision accuracy, we plotted the F0 contour generated by the two models in Figure 2. Figure 2 shows that N- Singer better predicts voiced and unvoiced segments of groundtruth than ATK. Because the V/UV decision affects singing intelligibility, the results suggest that our voicing-aware discriminators of N-Singer are effective in improving the pronunciation of the singing synthesized by the mel-generator.

### _3.2.3. Ablation studies_

Here, we show that the N-Singer can model text and pitch information individually. In Figure 3, _DP_ and _DT_ are generated by Transformer-based encoders from text and pitch information, respectively. _DT_ consists of non-periodic components with a structure that determines the pronunciation, whereas _DP_ is composed of a frequency component. The coarse mel-spectrogram _MD_ is generated by multiplying _DT_ and _DP_ . _Mask_ , _DM_ , and _M_<sup>ˆ</sup> are the intermediate outputs of the ATK. Because ATK uses 1/4-length mel-spectrograms, the intermediate outputs in Figure 3 show pronunciation and pitch information loss. However, because the N-Singer uses the whole mel-spectrogram, it can synthesize a more accurate singing voice.

# 4. Conclusion

In this paper, we proposed N-Singer, which can synthesize Korean singing voice with more accurate pronunciation by individually modeling linguistic features of lyrics and musical features of pitches from mel-spectrograms in a non-autoregressive manner. Experimental results show that N-Singer’s voicing-aware discriminators are effective in improving pronunciation.

> 1https://nc-ai.github.io/speech/publications/nsinger/


# 5. References

- [1] J. Kim, H. Choi, J. Park, M. Hahn, S.-J. Kim, and J.-J. Kim, “Korean singing voice synthesis based on an lstm recurrent neural network.” in _Proceedings INTERSPEECH 2018 International Speech Communication Association_ , 2018, p. 1551–1555.

- [2] J. Lee, H. S. Choi, C. B. Jeon, J. Koo, and K. Lee, “Adversarially trained end-to-end korean singing voice synthesis system,” in _Proceedings INTERSPEECH 2019 International Speech Communication Association_ , 2019, pp. 803–806.

- [3] S. Choi, W. Kim, S. Park, S. Yong, and J. Nam, “Korean singing voice synthesis based on auto-regressive boundary equilibrium gan,” in _Proceedings ICASSP 2020 IEEE International Conference on Acoustics, Speech and Signal Processing_ , 2020, pp. 7234– 7238.

- [4] J. Chen, X. Tan, J. Luan, T. Qin, and T. Y. Liu, “Hifisinger: Towards high-fidelity neural singing voice synthesis,” in _arXiv preprint arXiv:2009.01776_ , 2021.

- [5] Y. Wang, R. J. Skerry-Ryan, D. Stanton, Y. Wu, R. J. Weiss, N. Jaitly, Z. Yang, Y. Xiao, Z. Chen, S. Bengio, Q. Le, Y. Agiomyrgiannakis, R. Clark, and R. A. Saurous, “Tacotron: Towards end-to-end speech synthesis,” in _Proceedings INTERSPEECH 2017 International Speech Communication Association_ , 2017, p. 4006–4010.

- [6] J. Shen, R. Pang, R. J. Weiss, M. Schuster, N. Jaitly, Z. Yang, Z. Chen, Y. Zhang, Y. Wang, and R. S.-R. et al, “Natural tts synthesis by conditioning wavenet on mel spectrogram predictions,” in _Proceedings ICASSP 2018 IEEE International Conference on Acoustics, Speech and Signal Processing_ , 2018, pp. 4779–4783.

- [7] H. Tachibana, K. Uenoyama, and S. Aihara, “Efficiently trainable text-to-speech system based on deep convolutional networks with guided attention,” in _Proceedings ICASSP 2018 IEEE International Conference on Acoustics, Speech and Signal Processing_ , 2018, pp. 4784–4788.

- [8] Y. Gu, X. Yin, Y. Rao, Y. Wan, B. Tang, and Y. Zhang, “Bytesing: A chinese singing voice synthesis system using duration allocated encoder-decoder acoustic models and wavernn vocoders,” in _Proceedings ISCSLP 12th International Symposium on Chinese Spoken Language Processing_ , 2021, pp. 1–5.

- [9] H. Y. Gu and J. K. He, “Singing-voice synthesis using demisyllable unit selection,” in _Proceedings ICMLC 2016 International Conference on Machine Learnign and Cybernetics_ , 2016, pp. 654–659.

- [10] M. Nishimura, K. Hashimoto, K. Oura, Y. Nankaku, and K. Tokuda, “Singing voice synthesis based on deep neural networks,” in _Proceedings INTERSPEECH 2016 International Speech Communication Association_ , 2016, pp. 2478–2482.

- [11] M. Blaauw and J. Bonada, “A neural parametric singing synthesizer modeling timbre and expression from natural songs,” _Applied Sciences_ , vol. 7, no. 12, p. 1333, 2017.

- [12] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. WardeFarley, S. Ozair, A. Courville, and Y. Bengio, “Generative adversarial nets,” in _Proceedings NIPS Neural Information Processing Systems_ , 2014, pp. 2672–2680.

   - [17] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, “Attention is all you need,” in _Proceedings NIPS Neural Information Processing Systems 30_ , 2017, pp. 6000–6010.

   - [18] Y. Ren, Y. Ruan., X. Tan, T. Qin, S. Zhao, Z. Zhao, and T. Y. Liu, “Fastspeech: Fast, robust and controllable text to speech,” in _Proceedings NIPS Neural Information Processing Systems 32_ , 2019, pp. 3165–3174.

   - [19] R. Yamamoto, E. Song, and J. M. Kim, “Parallel wavegan: A fast waveform generation model based on generative adversarial networks with multi-resolution spectrogram,” in _Proceedings ICASSP 2020 IEEE International Conference on Acoustics, Speech and Signal Processing_ , 2021, pp. 6199–6203.

   - [20] M. M. Association, “Midi manufacturers association,” in _https://www.midi.org._

   - [21] I. Bello, B. Zoph, A. Vaswani, J. Shlens, and Q. V. Le, “Attention augmented convolutional networks,” in _Proceedings ICCV 2019 IEEE International Conference on Computer Vision_ , 2019, pp. 3286–3295.

   - [22] A. Gulati, J. Qin, C. Chiu, and e. a. N. Parmar, Y. Zhang, “Conformer: Convolution-augmented transformer for speech recognition,” in _Proceedings INTERSPEECH 2020 International Speech Communication Association_ , 2020, pp. 5036–5040.

   - [23] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, “Dropout: a simple way to prevent neural networks from overfitting,” _The Journal of Machine Learning Research_ , vol. 15, no. 1, pp. 1929–1958, 2014.

   - [24] R. K. Srivastava, K. Greff, and J. Schmidhuber, “Highway networks,” in _arXiv preprint arXiv:1505.00387_ , 2015.

   - [25] D. A. Clevert, T. Unterthiner, and S. Hochreiter, “Fast and accurate deep network learning by exponential linear units (elus),” in _Proceedings ICLR International Conference on Learning Representations_ , 2016.

   - [26] R. Yamamoto, E. Song, M. J. Hwang, and J. M. Kim, “Parallel waveform synthesis based on generative adversarial networks with voicing-aware conditional discriminators,” in _Proceedings ICASSP 2021 IEEE International Conference on Acoustics, Speech and Signal Processing_ , 2021, pp. 6039–6043.

   - [27] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” in _arXiv preprint arXiv:1412.6980_ , 2014.

   - [28] L. Liu, H. Jiang, P. He, W. Chen, X. Liu, J. Gao, and J. Han, “On the variance of the adaptive learning rate and beyond,” in _arXiv preprint arXiv:1908.03265_ , 2019.

   - [29] W. Frank, “Individual comparisons by ranking methods,” _Biometrics Bulletin_ , vol. 1, no. 6, pp. 80–83, 1945.

   - [30] M. Morise, F. Yokomori, and K. Ozawa, “World: a vocoder-based high-quality speech synthesis system for real-time applications,” _IEICE TRANSACTIONS on Information and Systems_ , vol. 99, no. 7, pp. 1877–1884, 2016.

- [13] T. Miyato and M. Koyama, “cgans with projection discriminator,” in _Proceedings ICLR 2018 International Conference on Learning Representations_ , 2018.

- [14] M. Blaauw and J. Bonada, “Sequence-to-sequence singing synthesis using the feed-forward transformer,” in _Proceedings ICASSP 2020 IEEE International Conference on Acoustics, Speech and Signal Processing_ , 2020, pp. 7229–7233.

- [15] P. Lu, J. Wu, J. Luan, X. Tan, and L. Zhou, “Xiaoicesing: A highquality and integrated singing voice synthesis system,” in _Proceedings INTERSPEECH 2020 International Speech Communication Association_ , 2020, pp. 1306–1310.

- [16] J. Wu and J. Luan, “Adversarially trained multi-singer sequenceto-sequence singing synthesizer,” in _Proceedings INTERSPEECH 2020 International Speech Communication Association_ , 2020, pp. 1296–1300.
