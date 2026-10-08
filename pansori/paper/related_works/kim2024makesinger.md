---
title: "MakeSinger: A Semi-Supervised Training Method for Data-Efficient Singing Voice Synthesis via Classifier-free Diffusion Guidance"
authors: "Semin Kim; Myeonghun Jeong; Hyeonseung Lee; Minchan Kim; Byoung Jin Choi; Nam Soo Kim"
venue: "Proc. Interspeech"
year: 2024
bib_key: null
source_url: "https://www.isca-archive.org/interspeech_2024/kim24i_interspeech.pdf"
version: "published"
cited_in_paper: false
relevance: "Data-efficient semi-supervised SVS is a direct comparator for exploiting unlabeled speech or singing."
---

## Why it matters

Data-efficient semi-supervised SVS is a direct comparator for exploiting unlabeled speech or singing.


Interspeech 2024 1-5 September 2024, Kos, Greece


# MakeSinger: A Semi-Supervised Training Method for Data-Efficient Singing Voice Synthesis via Classifier-free Diffusion Guidance

_Semin Kim_<sup>1</sup> _, Myeonghun Jeong_<sup>1</sup> _, Hyeonseung Lee_<sup>1</sup> _, Minchan Kim_<sup>1</sup> _, Byoung Jin Choi_<sup>1</sup> _, Nam Soo Kim_<sup>1</sup>

1Department of Electrical and Computer Engineering and INMC, Seoul National University, Seoul, South Korea

_{_ smkim21, mhjeong, hslee, bjchoi, mckim _}_ @hi.snu.ac.kr, nkim@snu.ac.kr

## Abstract

In this paper, we propose MakeSinger, a semi-supervised training method for singing voice synthesis (SVS) via classifierfree diffusion guidance. The challenge in SVS lies in the costly process of gathering aligned sets of text, pitch, and audio data. MakeSinger enables the training of the diffusion-based SVS model from any speech and singing voice data regardless of its labeling, thereby enhancing the quality of generated voices with large amount of unlabeled data. At inference, our novel dual guiding mechanism gives text and pitch guidance on the reverse diffusion step by estimating the score of masked input. Experimental results show that the model trained in a semisupervised manner outperforms other baselines trained only on the labeled data in terms of pronunciation, pitch accuracy and overall quality. Furthermore, we demonstrate that by adding Text-to-Speech (TTS) data in training, the model can synthesize the singing voices of TTS speakers even without their singing voices.

**Index Terms** : singing voice synthesis, semi-supervised training, diffusion generative model

# 1. Introduction

Singing voice synthesis (SVS) is a rapidly growing field of research adopting various deep learning techniques. Similar to Text-to-Speech (TTS), a typical two-stage pipeline of neural SVS consists of an acoustic model and a vocoder. The acoustic model [1, 2, 3, 4] generates acoustic features such as mel-spectrogram from the musical score, including text, tempo, and pitch. The vocoder [5, 6, 7, 8] then converts these acoustic features to a waveform. Recent advances include end-to-end models that aim to improve the sound quality and naturalness such that the generated audio results in a more human-like voice [9, 10, 11, 12].

However, training such models demands a high-quality dataset with precisely aligned ( _text_ , _pitch_ , _audio_ ) triplets. This makes collecting data for SVS challenging, as it involves professional singers who can accurately sing to the pitch and tempo of the musical score. Semi-supervised approaches have emerged to address this issue, leveraging extracted pseudo labels from unlabeled data. For instance, [13] employed Harvest [14] to generate pitch labels, and phoneme classifier for text labels. DeepSinger [15] gathered unlabeled data from publicly available music websites while extracting the pitch using Parselmouth<sup>1</sup> . Despite their effectiveness, these approaches come with a significant limitation: they are not entirely free from data labeling. They still require ( _text_ , _audio_ ) pair for the training data, or require an auxiliary classifier specifically tailored for

> 1https://github.com/YannickJadoul/Parselmouth

singing voices, which eventually demands a large amount of labeled data to prevent potential error propagation from inaccurate labeling.

Meanwhile, diffusion generative models [16, 17, 18] have shown significant breakthroughs in many areas of deep learning. Notably, studies such as [19, 20, 21] have demonstrated the remarkable effectiveness of diffusion models in TTS and SVS. Additionally, [22, 23] introduced a training method that capitalizes on the strengths of diffusion models, incorporating the training of a separate phoneme classifier to provide text guidance. To avoid building the classifier, [24] introduced classifierfree guidance, enabling a model to learn from both conditional and unconditional data and generate class-conditional samples by a single joint model. These guiding approaches facilitate efficient learning from unlabeled datasets.

Inspired by these works, we propose MakeSinger, a semisupervised training method for data-efficient singing voice synthesis that leverages any speech or singing voice, regardless of its labeling. MakeSinger employs classifier-free guidance, taking advantage of the joint model’s capability to learn from both labeled and unlabeled data. We also present a novel dual guiding mechanism that adjusts the balance between text and pitch guidance during reverse diffusion steps. Our experimental results indicate that incorporating a substantial amount of unlabeled singing voice data into training enhances the pronunciation and overall quality of the synthesized voices, outperforming baselines supervised with more labeled songs. Furthermore, by leveraging the multi-speaker TTS dataset, we demonstrate MakeSinger’s ability to generate the singing voices of diverse TTS speakers, even without specific singing voice data for those speakers. We present our samples generated by MakeSinger on our demo page<sup>2</sup> .

# 2. Proposed Method

In this section, we describe MakeSinger, our proposed semisupervised training method for SVS. As a base architecture, we use Grad-TTS [19], a score-based TTS model, with several modifications. In training procedure, our diffusion model learns from any audio without relying on its labeling, contributing the robustness of the system in the low-resource scenario with a limited availability of paired data. During inference, we apply a dual guiding mechanism to provide a balanced and appropriate level of pitch and text guidance. We explain detailed architecture of MakeSinger in Section 2.1, and our dual guiding mechanism in Section 2.2. The comprehensive overview of the MakeSinger is illustrated in Figure 1.

> 2https://makesinger.github.io/MakeSinger-demo


```
10.21437/Interspeech.2024-678
```


TEXT PITCH SPK TEXT PITCH SPK<br>Text Embedding Pitch Embedding Speaker  Text Embedding Pitch Embedding Speaker<br>Embedding Embedding<br>Text Encoder<br>Text Encoder<br>MASKING<br>𝑐 𝑚 ∅𝑐 𝑚<br>𝑐 𝑚 ∅𝑐 𝑚<br>Labeled data Pitch-only data<br>Pitch-only data Labeled data Pitch-only data<br>MASKING<br>𝑐 ∅m ∅𝑐 ∅m ∅𝑐 ∅m<br>Text-only data Unlabeled data Unlabeled data<br>CONCAT<br>CONCAT<br>Score Estimator<br>𝑋0 𝑠𝜃(𝑋𝑡, 𝑚, 𝑐, 𝑡)<br>Score Estimator 𝑠𝜃 𝑋𝑡, 𝑚, ∅𝑐, 𝑡−𝑠𝜃(𝑋𝑡, ∅m, ∅𝑐, 𝑡)<br>𝑠𝜃(𝑋𝑡, 𝑚, 𝑐, 𝑡) ,  𝑠𝜃(𝑋𝑡, 𝑚, ∅𝑐, 𝑡) ,  𝑠𝜃(𝑋𝑡, ∅m, 𝑐, 𝑡) ,   𝑠𝜃(𝑋𝑡, ∅m, ∅𝑐, 𝑡) 𝑋𝑡, 𝑡<br>𝑠𝜃 𝑋𝑡, 𝑚, 𝑐, 𝑡−𝑠𝜃 𝑋𝑡, 𝑚, ∅𝑐, 𝑡<br>𝑋𝑡−1<br>Pitch Guidance<br>Text Guidance<br>(a)  Training Procedure (b)  Inference Procedure<br>

Figure 1: _Overall procedure of MakeSinger._

## 2.1. Architecture

MakeSinger is designed to generate a mel-spectrogram corresponding to the given text, pitch, and speaker. When the model learns from labeled data, temporally aligned text and pitch sequence along with its mel-spectrogram becomes an input. In case where neither text nor pitch label exists, we use _⟨unknown⟩_ token sequence as substituting label. MakeSinger consists of two main parts: encoder and score estimator.

**Encoder** : The encoder part converts text, pitch, and speaker information into the feature sequence of mel-spectrogram frame length. Text encoder is consists of a pre-net, 6 Transformers [25], and a linear projection layer. All the feature sequences are concatenated together to give conditional information to the score estimator. As in Figure 1, we denote the feature sequences extracted from the text and pitch as _c_ and _m_ . We additionally denote the feature sequence extracted from _⟨unknown⟩_ text token as _∅c_ and _⟨unknown⟩_ pitch token as _∅m_ . Unlike in GradTTS, our encoder solely focuses on encoding the conditional information without estimating the mean and variance of the prior noise distribution, as the strong prior information can diminish the effect of the guiding process in Section 2.2.

**Score estimator** : The score estimator _sθ_ , which has U-Net [26] architecture, predicts the score at the given diffused mel-spectrogram _Xt_ and the concatenated feature sequence, where _t_ stands for the diffusion timestep. When training the model, we first pick uniformly random _t_ between 0 and 1. Then, we transform the ground-truth mel-spectrogram _X_ 0 to the diffused mel-spectrogram _Xt_ through a forward diffusion process. We leverage score-matching loss, an L2 loss between the predicted and computed scores of _Xt_ to train the score estimator.

## 2.2. Dual guiding in diffusion generative model

Original diffusion guiding mechanism [18] employs an additional classifier that demands enough labeled data to train the classifier as in Guided-TTS [22]. To address the low resource problem of the singing voice, we adopt a classifier-free guiding from [24] in a semi-supervised manner. This method involves providing guidance at each reverse diffusion step, facilitating class-conditional generation.

In our guiding scheme for generating singing voices, we recognize the necessity to manage underlying textual and pitch information. A simple approach is modeling the guidance for text and pitch with a single term. However, this approach hinders separate control over text and pitch, limiting the model’s flexibility. To ensure finer control, we opt for dual guidance, applying distinct guidance terms for text and pitch. We formulate the dual guidance for diffused mel-spectrogram _Xt_ , diffusion timestep _t_ , and conditional information _y_ . We start from a general diffusion guiding formulation from [18]:


We split _y_ to pitch condition _m_ and text condition _c_ . Then, applying Bayes’ rule, we can get classifier-free guidance for both labels:


Finally, we can derive the score using dual guiding with scaling factors:

- _st_ = _∇Xt_ log _pt_ ( _Xt|m, c_ ) + _w_ 1( _∇Xt_ log _pt_ ( _Xt|m, c_ ) _−∇Xt_ log _pt_ ( _Xt|m_ )) + _w_ 2( _∇Xt_ log _pt_ ( _Xt|m_ ) _−∇Xt_ log _pt_ ( _Xt_ )) _,_ (3)

where _st_ is a time-dependent total score for the reverse time process, _w_ 1 and _w_ 2 are scaling factors for text and pitch guidance. By a similar approach, we can also derive the following:

- _st_ = _∇Xt_ log _pt_ ( _Xt|m, c_ ) + _w_ 1( _∇Xt_ log _pt_ ( _Xt|m, c_ ) _−∇Xt_ log _pt_ ( _Xt|c_ )) + _w_ 2( _∇Xt_ log _pt_ ( _Xt|c_ ) _−∇Xt_ log _pt_ ( _Xt_ )) _._ (4)

Note that we provide guidance to the score estimated from conditional input since it generates better samples compared to giving guidance to the unconditional score. Actual implementation for Eq. 3 is expressed as follows:


We emphasize that a single joint model trained with both conditional and unconditional data estimates all the score terms. The model masks the original feature sequence to pitch-only and unlabeled data sequences to estimate each guidance, as depicted in Figure 1b. By employing norm-based guidance from [22], we optimize the control by scaling each norm of the guiding gradient in proportion to the norm of the score of the joint model in every diffusion timestep _t_ . _αt_<sup>1and</sup><sup>_α_</sup> _t_<sup>2in Eq.5 are normalizing</sup> factors of the norm-based guidance.

# 3. Experimental Settings

## 3.1. Datasets

We conducted multi-speaker SVS experiments on the the opensourced Korean multi-speaker singing voice dataset<sup>3</sup> . We employed 1,210 ballad songs sung by 12 singers. We spitted it into 973 songs for training and 277 songs for the test. We also conducted SVS experiments utilizing TTS data from open-sourced large-scale Korean multi-speaker dataset<sup>4</sup> , aiming to synthesis singing voices of TTS speakers. All audio files were resampled to 22,050 Hz and we computed the mel-spectrograms using a 1,024-point short-time Fourier transform with 256-point hop size and mel-scale filterbanks featuring 80 bins.

## 3.2. Input Processing

Our dataset is paired with musical instrument digital interface (MIDI) files that serve as labels. As MIDI file contains aligned pitch and text expressed in syllables according to the tempo, we extracted _m_ , _c_ , _f_<sup>_s_</sup> , and _f_<sup>_e_</sup> of length _I_ from each song. _m_ , _c_ denotes sequence of pitches and Korean syllables respectively, and _fi_<sup>_s_,</sup><sup>_f_</sup> _i_<sup>_e_correspond to start and end frame index</sup> of _i_ th pitch and syllable. _fi_<sup>_s_,</sup><sup>_f_</sup> _i_<sup>_e_is computed by considering the</sup> frequency of waves and mel-spectrogram frame length. Since a Korean syllable can be divided into three phonemes: an onset, a nucleus, and an optional coda, we allocated frames for each

phoneme. In Korean, most vocalization time spends on pronouncing vowels, which is always the nucleus. Following the approach of [4, 9, 27], we allocated three frames for the onset and coda, the rest for the nucleus while dividing the frames of a syllable.

For TTS data, We used Montreal Forced Aligner (MFA) [28], which automatically extracts speech-phoneme alignments, to extend syllables to mel-spectrogram length. In order to standardize the phoneme set for both the SVS and TTS, we applied the same allocation of onset, nucleus, and coda as we did for the singing voice data.

## 3.3. Model Configuration

We trained the following models for comparison in the multispeaker SVS experiments.

- **MLP-Singer** : MLP-Mixer [29] based Korean SVS model.

- **VISinger** : End-to-end SVS model based on VITS [30]. Given that our experimental data were pre-aligned, we removed the duration predictor and length regulator from the original model. Additionally, we substituted its encoder to ours for fair comparison.

- **MakeSinger without semi-supervision** : Our proposed model only supervised with labeled data.

- **MakeSinger with semi-supervision** : Our proposed model with semi-supervised training. Dual guiding was applied in the inference procedure.

To demonstrate the effectiveness of our proposed method and its correlation with the amount of labeled data, we varied the number of labeled songs across three different settings, maintaining constant 12 singers: 12, 36, and 108 songs. MLPSinger, VISinger, and MakeSinger without semi-supervision exclusively utilized labeled songs for training.

For the MakeSinger with semi-supervision, we extended the training set by including the remaining unlabeled songs. Specifically, we used [12, 961], [36, 937], [108, 865] labeled and unlabeled songs. During inference, we employed Eq. 3 from Section 2 for reverse diffusion steps. To improve the gradient estimation of _∇x_ log _pt_ ( _Xt|m_ ), we introduced pitch-only data in the training data created by masking text labels of the labeled data. We empirically configured the scaling factor for text guidance _w_ 1 = 0 _._ 2 and pitch guidance _w_ 2 = 0 _._ 02 in the dual guiding process.

Both MakeSinger models used 200 reverse diffusion steps and shared hyperparameter settings with Grad-TTS. Unlike [18, 19] that used an ordinary differential equation (ODE) without the stochastic term in the inference process, we opted for a stochastic differential equation (SDE) to better capture the vibrato commonly found at the end of sentences in ballad songs.

As MLP-Singer and MakeSinger generate mel-spectrogram, we used Hifi-GAN [7] to generate the waveform. We fine-tuned its universal pre-trained version<sup>5</sup> to our singing voice dataset since Hifi-GAN is originally trained to synthesis speech. MLP-Singer is trained on a single 2080Ti GPU. All the other models were trained up to 60 epochs on 4 2080Ti GPUs.

# 4. Results

## 4.1. Main Results

To verify above four systems in three different settings, we performed a subjective evaluation using the mean opinion

> 3https://bit.ly/3uZc7WM

> 4https://bit.ly/4c46sz6

> 5https://github.com/jik876/hifi-gan


score (MOS) test, and for objective evaluation, we utilized F0 root mean square error (F0-RMSE) and semitone accuracy (SACC). Results are presented in Table. 1

**Subjective Evaluation Metrics** : The MOS test involved 14 participants and was conducted with 95 % confidence intervals. We evaluated 24 randomly sampled song segments each 4 to 6 seconds long, focusing on pronunciation accuracy (MOSP) and overall quality (MOS-Q). For pronunciation accuracy, we instructed testers to evaluate how naturally the singer pronounced the text in terms of ballad singing. For overall quality, we instructed testers to evaluate the comprehensive quality of singing, including sound quality, pronunciation, pitch accuracy, and naturalness.

**Objective Evaluation Metrics** : We computed F0-RMSE and S-ACC against reference audios reconstructed by a vocoder. S- ACC measures how accurately a system can determine the pitch of a musical note in semitones. Randomly selected 350 segments of songs between 4 to 6 seconds were used to compute average score.

**Results** : As shown in Table. 1, even MakeSinger without semi-supervision outperformed other baseline systems for every metric when trained with 36 and 108 songs. Furthermore, MakeSinger with semi-supervision achieved best scores across all evaluations, demonstrating the effectiveness of dual guiding mechanism, especially in pitch accuracy. Remarkably, with just 12 or 36 labeled songs, the semi-supervised MakeSinger showed comparable or superior performance to other baseline models trained on 108 labeled songs in every metric, highlighting our model’s data efficiency. This efficiency was most pronounced with the smallest labeled dataset, where the semisupervised model showed significant improvement over its nonsemi-supervised counterpart. The performance gap between the two proposed models becomes smaller as the quantity of labeled songs increased, which aligns with the expectation that the importance of guiding decreases as more labeled data becomes available for the model to learn from.

## 4.2. Ablation study

To validate our dual guiding mechanism, we carried out an ablation study on the same semi-supervised MakeSinger model, comparing three different approaches during inference: dual guiding, single guiding, and no guiding. Single guiding approach models text and pitch guidance in a single term, which is expressed as follows:


We set _w_ = 0 _._ 2 in the single guiding to match the scale of guidance. The result in Table 2 shows that the single guiding mechanism degraded MakeSinger’s pronunciation and overall quality, whereas completely omitting the guiding mechanism led to further degradation.

## 4.3. Singing Voice Synthesis based on TTS data

Utilizing TTS data helps MakeSinger to generate singing voices in TTS speakers’ voices while maintaining the characteristics and timbre of the TTS speaker. Exploiting the high quality and high coverage of multi-speaker TTS data can also be a breakthrough to the insufficient number of singers and songs in the SVS dataset. Eq. 4 is more suitable for the task, as the model learns to estimate _∇x_ log _pt_ ( _Xt|c_ ) from the TTS data, which only has text labels.

Table 1: _Experimental results in terms of MOS-P, MOS-Q, F0RMSE, and S-ACC._

|**Method**|**MOS-P**|**MOS-Q**|**F0-RMSE**<br>**(**_↓_**)**|**S-ACC**<br>**(**_↑_**)**|
|---|---|---|---|---|
|Reconstruction|4.64_±_0.07|4.47_±_0.08|-|-|
|**12 labeled songs**|||||
|MLP-singer|2.79_±_0.11|2.52_±_0.10|0.2468|41.76|
|VISinger|2.20_±_0.11|2.19_±_0.09|0.3005|37.14|
|MakeSinger|||||
|_- w/o semi-sup._|2.37_±_0.11|2.54_±_0.11|0.2543|48.98|
|_- w/ semi-sup._|**3.30**_±_**0.12 **|**3.48**_±_**0.12**|**0.1736**|**54.29**|
|**36 labeled songs**|||||
|MLP-singer|3.14_±_0.11|2.71_±_0.09|0.2419|42.95|
|VISinger|2.74_±_0.11|2.59_±_0.09|0.2121|45.91|
|MakeSinger|||||
|_- w/o semi-sup._|3.41_±_0.11|3.36_±_0.11|0.1942|52.51|
|_- w/ semi-sup._|**3.61**_±_**0.12 **|**3.66**_±_**0.12**|**0.1720**|**54.25**|
|**108 labeled songs**|||||
|MLP-singer|3.45_±_0.11|3.01_±_0.09|0.2233|46.06|
|VISinger|3.51_±_0.11|3.33_±_0.10|0.2059|50.98|
|MakeSinger<br>_- w/o semi-sup._|3.95_±_0.10|3.76_±_0.09|0.1971|52.57|
|_- w/ semi-sup._|**4.03**_±_**0.09 **|**4.08**_±_**0.09**|**0.1702**|**56.05**|


Table 2: _Ablation study with MOS-P and MOS-Q._

|**Method**|**MOS-P**|**MOS-Q**|
|---|---|---|
|Reconstruction|4.54_±_0.10|4.49_±_0.10|
|**MakeSinger**|||
|_- no guiding_|3.80_±_0.10|3.77_±_0.07|
|_- single guiding_|3.87_±_0.10|3.84_±_0.07|
|_- dual guiding_|**3.95**_±_**0.10 **|**3.94**_±_**0.06**|


To show the feasibility of generating singing voices of TTS speakers, we incorporated multi-speaker TTS data into the training process. This involved using 487 labeled songs along with the remaining unlabeled songs and 3 hours of TTS data uttered by 80 speakers. Results show that MakeSinger is capable of generating singing voices from TTS speakers with dependable quality, as presented in our demo page.

# 5. Conclusion

We presented MakeSinger, a diffusion-based semi-supervised training method for singing voice synthesis. Exploiting the advantage of the classifier-free guidance, we could utilize any human speech and singing voice data regardless of its labeling. Furthermore, during the reverse diffusion steps, our novel dual guiding mechanism provides better controllability to the model regarding text and pitch. To the best of our knowledge, MakeSinger is the first SVS model to leverage classifier-free diffusion guiding in a semi-supervised manner. By applying our method, we showed that the model can generate singing voices with better pronunciation and sound quality. Also, we demonstrated that MakeSinger generates songs in the voice of TTS speakers by utilizing multi-speaker TTS data in training.


# 6. Acknowledgements

This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No.2021-0-00456, Development of Ultra-high Speech Quality Technology for Remote Multi-speaker Conference System)

# 7. References

- [1] M. Blaauw and J. Bonada, “Sequence-to-sequence singing synthesis using the feed-forward transformer,” in _ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ . IEEE, 2020, pp. 7229–7233.

- [2] G.-H. Lee, T.-W. Kim, H. Bae, M.-J. Lee, Y.-I. Kim, and H.Y. Cho, “N-singer: A non-autoregressive korean singing voice synthesis system for pronunciation enhancement,” _arXiv preprint arXiv:2106.15205_ , 2021.

- [3] Y. Hono, K. Hashimoto, K. Oura, Y. Nankaku, and K. Tokuda, “Sinsy: A deep neural network-based singing voice synthesis system,” _IEEE/ACM Transactions on Audio, Speech, and Language Processing_ , vol. 29, pp. 2803–2815, 2021.

- [4] J. Tae, H. Kim, and Y. Lee, “Mlp singer: Towards rapid parallel korean singing voice synthesis,” in _2021 IEEE 31st International Workshop on Machine Learning for Signal Processing (MLSP)_ . IEEE, 2021, pp. 1–6.

- [5] A. van den Oord, S. Dieleman, H. Zen, K. Simonyan, O. Vinyals, A. Graves, N. Kalchbrenner, A. Senior, and K. Kavukcuoglu, “Wavenet: A generative model for raw audio,” in _Arxiv_ , 2016. [Online]. Available: https://arxiv.org/abs/1609.03499

- [6] R. Yamamoto, E. Song, and J.-M. Kim, “Parallel wavegan: A fast waveform generation model based on generative adversarial networks with multi-resolution spectrogram,” in _ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ . IEEE, 2020, pp. 6199–6203.

- [7] J. Kong, J. Kim, and J. Bae, “Hifi-gan: Generative adversarial networks for efficient and high fidelity speech synthesis,” _Advances in Neural Information Processing Systems_ , vol. 33, pp. 17 022– 17 033, 2020.

- [8] R. Huang, F. Chen, Y. Ren, J. Liu, C. Cui, and Z. Zhao, “Multisinger: Fast multi-singer singing voice vocoder with a large-scale corpus,” in _Proceedings of the 29th ACM International Conference on Multimedia_ , 2021, pp. 3945–3954.

- [9] J. Lee, H.-S. Choi, C.-B. Jeon, J. Koo, and K. Lee, “Adversarially trained end-to-end korean singing voice synthesis system,” in _Interspeech_ , 2019.

- [10] J. Chen, X. Tan, J. Luan, T. Qin, and T.-Y. Liu, “Hifisinger: Towards high-fidelity neural singing voice synthesis,” _arXiv preprint arXiv:2009.01776_ , 2020.

- [11] Y. Zhang, J. Cong, H. Xue, L. Xie, P. Zhu, and M. Bi, “Visinger: Variational inference with adversarial learning for end-to-end singing voice synthesis,” in _ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ . IEEE, 2022, pp. 7237–7241.

- [12] Y. Zhang, H. Xue, H. Li, L. Xie, T. Guo, R. Zhang, and C. Gong, “Visinger 2: High-fidelity end-to-end singing voice synthesis enhanced by digital signal processing synthesizer,” _arXiv preprint arXiv:2211.02903_ , 2022.

- [13] S. Choi and J. Nam, “A melody-unsupervision model for singing voice synthesis,” in _ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ . IEEE, 2022, pp. 7242–7246.

- [14] M. Morise _et al._ , “Harvest: A high-performance fundamental frequency estimator from speech signals.” in _INTERSPEECH_ , 2017, pp. 2321–2325.

- [15] Y. Ren, X. Tan, T. Qin, J. Luan, Z. Zhao, and T.-Y. Liu, “Deepsinger: Singing voice synthesis with data mined from the web,” in _Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining_ , 2020, pp. 1979–1989.

- [16] B. D. Anderson, “Reverse-time diffusion equation models,” _Stochastic Processes and their Applications_ , vol. 12, no. 3, pp. 313–326, 1982.

- [17] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” _Advances in Neural Information Processing Systems_ , vol. 33, pp. 6840–6851, 2020.

- [18] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole, “Score-based generative modeling through stochastic differential equations,” in _International Conference on Learning Representations_ , 2020.

- [19] V. Popov, I. Vovk, V. Gogoryan, T. Sadekova, and M. Kudinov, “Grad-tts: A diffusion probabilistic model for text-to-speech,” in _International Conference on Machine Learning_ . PMLR, 2021, pp. 8599–8608.

- [20] J. Liu, C. Li, Y. Ren, F. Chen, and Z. Zhao, “Diffsinger: Singing voice synthesis via shallow diffusion mechanism,” in _Proceedings of the AAAI Conference on Artificial Intelligence_ , vol. 36, no. 10, 2022, pp. 11 020–11 028.

- [21] M. Jeong, H. Kim, S. J. Cheon, B. J. Choi, and N. S. Kim, “Diff-tts: A denoising diffusion model for text-to-speech,” _arXiv preprint arXiv:2104.01409_ , 2021.

- [22] H. Kim, S. Kim, and S. Yoon, “Guided-tts: A diffusion model for text-to-speech via classifier guidance,” in _International Conference on Machine Learning_ . PMLR, 2022, pp. 11 119–11 133.

- [23] S. Kim, H. Kim, and S. Yoon, “Guided-tts 2: A diffusion model for high-quality adaptive text-to-speech with untranscribed data,” _arXiv preprint arXiv:2205.15370_ , 2022.

- [24] J. Ho and T. Salimans, “Classifier-free diffusion guidance,” in _NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications_ , 2021.

- [25] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” _Advances in neural information processing systems_ , vol. 30, 2017.

- [26] O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image segmentation,” in _Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18_ . Springer, 2015, pp. 234–241.

- [27] S. Choi, W. Kim, S. Park, S. Yong, and J. Nam, “Korean singing voice synthesis based on auto-regressive boundary equilibrium gan,” in _ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ . IEEE, 2020, pp. 7234–7238.

- [28] M. McAuliffe, M. Socolof, S. Mihuc, M. Wagner, and M. Sonderegger, “Montreal forced aligner: Trainable text-speech alignment using kaldi.” in _Interspeech_ , vol. 2017, 2017, pp. 498–502.

- [29] I. O. Tolstikhin, N. Houlsby, A. Kolesnikov, L. Beyer, X. Zhai, T. Unterthiner, J. Yung, A. Steiner, D. Keysers, J. Uszkoreit _et al._ , “Mlp-mixer: An all-mlp architecture for vision,” _Advances in neural information processing systems_ , vol. 34, pp. 24 261–24 272, 2021.

- [30] J. Kim, J. Kong, and J. Son, “Conditional variational autoencoder with adversarial learning for end-to-end text-to-speech,” in _International Conference on Machine Learning_ . PMLR, 2021, pp. 5530–5540.
