---
title: "Pitch Contour Tokenization using VQ-VAE and Its Application on Korean Traditional Music Analysis"
authors: "Seonguk Ju, Seola Cho, Sooin Chung, Danbinaerin Han, Dasaem Jeong"
year: 2026
venue: "ISMIR 2026"
bib_key: ju2026contour
cited_in_paper: false
source_url: "https://arxiv.org/abs/2608.10979"
pdf_url: "https://arxiv.org/pdf/2608.10979.pdf"
code_url: "https://github.com/MALerLab/pitch-contour-tokenizer"
relevance_note: >
  This work learns discrete pitch-contour tokens that retain expressive Korean traditional vocal gestures. Its VQ-VAE representation can encode sigimsae as reusable units instead of flattening them into equal-tempered notes. Such tokens could be predicted after lyric syllable alignment and translated into DiffSinger pitch curves or note ornaments.
---

# **PITCH CONTOUR TOKENIZATION USING VQ-VAE AND ITS APPLICATION ON KOREAN TRADITIONAL MUSIC ANALYSIS** 

**Seonguk Ju**<sup>1</sup> **Seola Cho**<sup>1</sup> **Sooin Chung**<sup>1</sup> 

**Danbinaerin Han**<sup>2</sup> **Dasaem Jeong**<sup>1</sup> 

1 Music & Arts Learning (MALer) Lab, Sogang University, South Korea 

2 Graduate School of Culture Technology, KAIST, South Korea 

{jinin5113, dasaemj}@sogang.ac.kr 

## **ABSTRACT** 

Computational analysis of music often relies on discrete representations, yet many musical traditions are organized around continuous pitch movement that resists segmentation into note-like units. For such traditions, the discrete units that analysis would build on are not given in advance. We address this gap by learning a vocabulary of local pitch-contour patterns directly from unlabeled audio, using a VQ-VAE that quantizes fixed-length contour segments into a finite codebook. To make the learned tokens stable across segmentation positions and small variations in timing and pitch range, we train the model with a reconstruction objective evaluated under the best alignment among a set of candidate temporal and pitch-domain transformations. Applied to Korean traditional music, the learned tokens recover information about expert-defined _sigimsae_ categories without supervision, and in _pansori_ individual tokens align with the two principal modes, _Gyemyeonjo_ and _Ujo_ , supporting their use as units for corpuslevel analysis of contour-centric traditions. 

## **1. INTRODUCTION** 

Representing music as sequences of discrete units has been one of the central abstractions underlying computational musicology and symbolic music information retrieval. Its analytical value lies in the fact that transitions between musical events can be counted and modeled statistically. For example, corpus-based chord progression analysis [1] can construct transition matrices between chord symbols and examine their distributional properties, which is possible because harmonic events are represented as discrete categories. A similar principle underlies probabilistic models of melody: once a melody is represented as a sequence of symbolic events, the probability or information content of each note can be estimated from its musical context. Temperley’s theory of uniform information density in music builds on this idea, explaining musical organization in 

> © S. Ju, S. Cho, S. Chung, D. Han and D. Jeong. Licensed under a Creative Commons Attribution 4.0 International License (CC BY 4.0). **Attribution:** S. Ju, S. Cho, S. Chung, D. Han and D. Jeong, “Pitch Contour Tokenization using VQ-VAE and Its Application on Korean Traditional Music Analysis”, in _Proc. of the 27th Int. Society for Music Information Retrieval Conf.,_ Abu Dhabi, UAE, 2026. 



<!-- Start of picture text -->
Input Pitch Contour<br>Receptive Field<br>Subtract Median<br>Encoder<br>Discrete Token Sequence<br>Pattern<br>Codebook Token i Token j Token k Token l<br>Decoder<br>Transformation-Minimized Reconstruction<br>Receptive Field<br>Temporal Offset Temporal Scale<br>Token Decoded Contour<br>Pitch Shifting Pitch-Range Scale<br>Add Median & Combine Receptive Field<br>Reconstructed Pitch Contour<br><!-- End of picture text -->

**Figure 1** . Overview of the proposed model. 

terms of the probability and information density of musical events [2]. The same logic also extends to performance data: MIDI represents performances as note events with onset time, duration, pitch, and velocity, allowing timing and dynamics to be analyzed as event-level attributes [3]. In this sense, symbolic representations are not merely encodings of music, but interfaces that make musical structure available to computational analysis. 

The utility of discrete representations, however, is fundamentally contingent on the availability of appropriate discrete units. Automatic music transcription (AMT) has made substantial progress in converting audio into notelevel symbolic representations, yet existing systems are developed and evaluated primarily on Western music [4]. This limitation is not merely a matter of musical style 

or timbral diversity; it reflects an assumption that musically meaningful events can be segmented into note-like units with relatively well-defined pitch and duration. Many musical traditions do not fully conform to this premise. In such traditions, expressive content is often realized as continuous pitch movement rather than as a succession of clearly bounded note events, making conventional transcription pipelines ill-suited as a basis for analysis. 

One way to obtain discrete units in such traditions is to rely on human expertise. Some studies use expert-labeled events directly: for instance, a recent study on Carnatic music used expert-labeled svara and svara-form annotations to show that the svara-form performed for a given svara is strongly influenced by neighboring svaras [5]. Others define a vocabulary of intermediate units in advance and transcribe recordings into it: the IDTAP platform, for example, represents Hindustani melodic practice using manually specified pitch-contour trajectories as a middle-level representation between raw F0 curves and conventional notation [6]. Both directions show that discrete representations of contour-centric music can support analysis, but they share a reliance on predefined categories and manual labor, which is costly to scale and ill-suited to traditions where the relevant ornamental vocabulary is not formally codified. This motivates a different question: can we learn a discrete vocabulary of contour units directly from data? 

In this work, we investigate this question through VQVAE-based pitch contour tokenization. VQ-VAE provides a natural framework for this purpose because it learns a finite set of latent codes while preserving enough information to reconstruct the input contour. We use this mechanism not as a generative objective in itself, but as a way to induce a compact vocabulary of local contour patterns that can be used for musical analysis. We apply this approach to _sigimsae_ , the ornamental system of Korean traditional music, where expressive musical units are often realized as continuous pitch movements rather than conventional note events. 

Our contribution is threefold: (i) a VQ-VAE-based pitch-contour tokenization model that learns a discrete vocabulary of local contour patterns without labels; (ii) a transformation-minimized reconstruction loss that makes token learning robust to small temporal and pitch-domain variations; and (iii) an evaluation via segmentation-shift consistency, _sigimsae_ classification, and qualitative mode analysis in _pansori_ . Our code and a demo page with audio examples are publicly available.<sup>1</sup> 

## **2. RELATED WORK** 

level transcription. Ranjani et al. [7, 8] mapped each pitch value onto a music-theoretically defined grid of note positions and showed that quantizing only at contour critical points preserves r¯aga and gamaka characteristics. Shikarpur et al. [9] adopted finely quantized pitch contours (10-cent bins) as an intermediate representation for hierarchical generative modeling of Hindustani vocal music. Both discretize pitch _values_ onto a predefined grid; we instead learn a vocabulary over local pitch _patterns_ (fixedlength contour segments) directly from unlabeled data. 

A separate line of work mines recurring melodic patterns from raw contours via similarity search. Gulati et al. [10] used DTW-based similarity over a 365-hour Carnatic corpus, and Nuttall et al. [11] applied the matrix profile to find repeated motifs within a single recording. These methods identify a small set of variable-length, recurrencebased instances that highlight reuse, but they do not yield a reusable vocabulary that covers every frame. Our tokenizer, by contrast, aims to produce a dense token sequence over the entire contour using a fixed codebook, which makes the resulting representation directly amenable to corpus-level statistical analysis. 

## **2.2 VQ-VAE and Discrete Representation Learning** 

VQ-VAE [12] learns discrete latent codes through reconstruction-based training. Given an input **x** , an encoder _fθ_ produces a continuous embedding **z** _e_ = _fθ_ ( **x** ), which is replaced by its nearest entry in a learned codebook _E_ = _{_ **e** _k}_<sup>_K_</sup> _k_ =1<sup>via</sup><sup>**z**</sup><sup>_q_=</sup><sup>**e**</sup><sup>_k∗_,</sup><sup>_k∗_=arg min</sup><sup>_k ∥_</sup><sup>**z**</sup><sup>_e−_</sup><sup>**e**</sup><sup>_k∥_2,</sup> and a decoder _gϕ_ reconstructs **x** ˆ = _gϕ_ ( **z** _q_ ). The model is trained with 



where sg[ _·_ ] is the stop-gradient operator and gradients flow through the quantization step via the straight-through estimator. 

In music and audio, vector quantization has been used primarily for generation and compression of audio waveforms or symbolic sequences [13–16], where codes are optimized for fidelity or bitrate rather than interpretability. 

Our use of VQ-VAE differs from both lines: the input is a one-dimensional pitch contour **x** _∈_ R<sup>_L_</sup> , and the learned codes are intended as interpretable analytical units rather than for generation or compression. In speech, vectorquantized autoencoders applied to raw or spectral audio have been shown to discover units that align with phonetic content [17], supporting the use of VQ-VAE for unsupervised unit discovery from continuous signals. 

## **2.1 Pitch-Contour Abstractions for Ornamental Music** 

A notable precedent is found in Indian art music, where quantized pitch contours have served as an intermediate representation capturing ornamental detail beyond note- 

1 Codebase: https://github.com/SeongUkJu/pitch-con tour-tokenizer, Demo: https://seongukju.github.io/ pitch-contour-tokenizer-page/ 

## **3. VQ-VAE ON PITCH CONTOUR** 

The overall architecture is based on VQ-VAE, using a stack of 1D convolutional layers, with two key design choices to enable interpretable, segmentation-robust (position-invariant) tokenization of pitch contours, as illustrated in Figure 1. 

## **3.1 Receptive Field Separation** 

To ensure a 1:1 correspondence between each discrete token and a fixed-length pitch contour segment, the input is pre-segmented into non-overlapping windows of a fixed size before encoding. This prevents inter-token dependency during decoding and promotes interpretability of the learned codebook by constraining each token to represent a single localized contour segment. 

To direct the codebook toward capturing ornamental contour patterns rather than absolute pitch levels, we subtract the median pitch value of each receptive field segment before encoding. This makes the objective insensitive to absolute pitch offset and encourages each entry to represent local movements such as oscillations and slides. When needed for visualization of the full contour sequence, the decoded contour can be shifted back by adding the subtracted median. 

## **3.2 Transformation-Minimized Reconstruction Loss** 

Because the input contour is divided into fixed-size nonoverlapping segments before encoding, the assigned token can be sensitive to the arbitrary segmentation position. For example, if a vibrato pattern is shifted by only a few frames, the phase observed within a segment changes, although the underlying oscillatory pattern remains the same. This matters because the model is trained with a frame-level MSE reconstruction loss: even when two contours correspond to the same local ornamental pattern, small temporal misalignments can lead to large reconstruction errors. Without an alignment-aware objective, the model may encode incidental frame-level differences as separate entries rather than a stable representation of the underlying pattern. 

We address this issue by minimizing the reconstruction error over four transformation factors: **(i) temporal offset** , which accounts for arbitrary segmentation position; **(ii) temporal scale** , which accounts for small duration differences; **(iii) pitch offset** , which accounts for residual vertical mismatch between the target and reconstruction; and **(iv) pitch-range scale** , which accounts for amplitude or range differences in otherwise similar contour shapes. We define a set of candidate transformations consisting of temporal alignment, pitch shifting, and pitch-range scaling. 

Let **x** _∈_ R<sup>_L_</sup> be a median-normalized input segment and **x** ˆ _∈_ R<sup>_L′_</sup> be the decoder output, where _L_<sup>_′_</sup> _> L_ . 

We define a transformed reconstruction candidate as 



where _Rs_ ( _·_ ) resamples the decoded contour with temporal scale _s_ , _Cτ_ ( _·_ ) extracts a length- _L_ crop at temporal offset _τ_ , _a_ scales the pitch range, and _b_ shifts the pitch level. A length- _L_ crop is then extracted with temporal offset _τ ∈T_ : 



where _Cτ_ ( _·_ ) returns a contiguous segment of length _L_ starting at offset _τ_ . 

The reconstruction loss is computed as the minimum MSE over all candidate transformations: 



Only the transformation attaining the minimum error is used for backpropagation. 

This can be interpreted as fitting the model under the best alignment between the decoded contour and the target segment, rather than penalizing all frame-level discrepancies directly. This objective encourages the codebook to represent contour patterns that are stable under small temporal and pitch-domain variations. 

For example, vibrato contours with similar periodic structure can be mapped to the same token even when they differ in phase or amplitude, because the loss evaluates reconstruction after the best temporal and pitch-domain alignment. 

## **4. EXPERIMENTAL SETUP** 

## **4.1 Dataset** 

In this study, we apply VQ-VAE to pitch contours of Korean traditional music. Although the proposed method is applicable to any pitch contour-based musical tradition, such as Indian art music or pop vocal singing, we focus on Korean traditional music. This is because continuous pitch movement is not merely an expressive embellishment but a central, defining element of the tradition. Unlike Western staff notation, in which music is conceived as a sequence of discrete note events, Korean traditional music, particularly its ornamental system _sigimsae_ , is realized through fluid, continuous pitch trajectories in which the boundary between one “note” and the next is often ambiguous or undefined. This makes it an especially well-suited testbed for a model designed to discover meaningful structure from pitch contours without presupposing discrete note events. 

We selected an in-house _pansori_ recording dataset consisting of approximately 280 hours of audio recordings. _Pansori_ is a representative vocal genre of Korean traditional music, sung by a solo singer with barrel drum accompaniment. It is well known for its extensive use of melodic ornamentation, making it a particularly rich source of the ornamental patterns investigated in this study. For _pansori_ recordings, the vocal tracks are first isolated using HT-Demucs [18] to remove instrumental accompaniment. 

## _4.1.1 Pre-processing_ 

We extract the fundamental frequency (F0) with the pretrained CREPE [19], which estimates F0 values every 10 ms with a confidence score in [0 _,_ 1]. The extracted F0 values are converted to MIDI scale and normalized by median. Frames below a predefined confidence threshold are filtered out. Within each receptive field segment, the gaps of up to two consecutive filtered frames are filled via linear interpolation. 

Segments containing extensive unvoiced regions are discarded because our 1D CNN-based tokenizer assumes a dense continuous contour and cannot directly represent missing pitch values. Filling such regions with a constant padding value would introduce artificial patterns into the input, which could be absorbed by the codebook and hinder the interpretability of the learned representations in subsequent analysis. Addressing unvoiced regions more explicitly is left for future work. 

Before filtering, the data are split 8:1:1 into train, validation, and test sets. During training, a random start point is sampled from each filtered segment at every epoch, introducing variability in segmentation position, while the validation and test sets use fixed segments for consistent evaluation. 

## **4.2 Model Specification** 

The encoder and decoder each consist of 6 stacked onedimensional convolutional layers with a receptive field size of 128 frames and a codebook of 256 entries. The model is trained using the VQ-VAE objective [12], in which the standard reconstruction loss is replaced by the transformation-minimized reconstruction loss described in Section 3.2, combined with a vector quantization loss with commitment coefficient _β_ = 2 _._ 0. Training is conducted for 50K updates using the AdamW optimizer with an initial learning rate of 0.001 and a batch size of 1024. We select the checkpoint with the lowest validation reconstruction loss. 

The transformation parameters are set empirically. The decoder produces 2 _L_ frames, resampled by _Rs_ ( _·_ ) to lengths _Ls_ approximately _S_ = _{_ 1 _,_ 1 _._ 08 _,_ 1 _._ 2 _}_ times _L_ = (i.e., 128, 139, and 154 frames); all valid offsets _T {_ 0 _,_ 1 _, . . . , Ls − L}_ are considered as crops, where _Ls_ is the resampled length. Pitch-range scaling and pitch shifting use _A_ = _{_ 0 _._ 85 _,_ 1 _._ 0 _,_ 1 _._ 15 _}_ and _B_ = _{−_ 0 _._ 1 _,_ 0 _._ 0 _,_ 0 _._ 1 _}_ . 

For the comparison, we also trained an **Autoencoder Baseline** , adapting Nuttall et al. [20], who applied Uniform Manifold Approximation and Projection (UMAP) [21] and K-Means clustering to supervised GRU embeddings. Our unsupervised variant trains an autoencoder on the _pansori_ dataset and tokenizes its latent embeddings via z-score normalization, UMAP, and K-Means. 

## **5. SEGMENTATION SHIFT CONSISTENCY** 

We introduced the transformation-minimized reconstruction loss to reduce the sensitivity of token assignment to temporal and pitch-domain variations. Here, we isolate one of its intended effects: robustness to segmentation position. Specifically, we test whether the same long pitch contour yields similar token sequences when the starting position of the fixed-size segmentation window is shifted. 

## **5.1 Experimental Dataset Construction** 

From the _pansori_ dataset used for VQ-VAE training, we select contour segments that satisfy the filtering criteria, contain no unvoiced frames, and are at least three times 

|Method|KL|D_↓_|Ac|c._↑_|
|---|---|---|---|---|
||0–60|0–124|0–60|0–124|
|Autoencoder|1.747|2.085|0.100|0.115|
|VQ-VAE|0.657|0.859|0.462|0.488|
|+ temporal align|**0.531**|**0.735**|**0.519**|**0.561**|
|+ all transformations|0.652|0.881|0.456|0.495|



**Table 1** . Token consistency evaluation results. KLD denotes Token Distribution KLD, and Acc. denotes Token Matching Accuracy. Results are reported for two segmentation-offset ranges: 0–60 and 0–124 frames. “+ temporal align” applies temporal offset and temporal-scale transformations, while “+ all transformations” additionally includes pitch offset and pitch-range scaling. 

longer than the receptive field. For each segment, we generate multiple token sequences by shifting the segmentation start position in increments of 4 frames. 

We report results for two offset ranges: 0–60 frames and 0–124 frames. The former evaluates shifts smaller than half of the 128-frame receptive field, while the latter covers nearly all possible phases within one receptive field and thus provides a stricter test of segmentation robustness. 

## **5.2 Evaluation Metrics** 

We use two complementary metrics. **Token Distribution KLD** measures whether the overall codebook usage remains stable across shifted versions of the same contour. For each segment, we construct a reference token distribution by aggregating token counts across all shifted versions. We then compute the Kullback–Leibler divergence (KLD) between the token distribution of each shifted version and this reference distribution, and average the result over versions and segments. Lower KLD indicates more stable token usage. 

**Token Matching Accuracy** measures local token-level consistency. We use the offset-0 token sequence as the reference and compare each shifted token sequence against it. For offsets below half of the receptive field, tokens are compared at the same sequence index. For offsets of 64 frames or greater, each shifted token is compared with the succeeding reference token, since the shifted receptive field overlaps more strongly with the next reference window. Higher accuracy indicates more consistent token assignment across shifts. 

## **5.3 Results** 

As shown in Table 1, the autoencoder baseline is highly sensitive to segmentation shifts, whereas VQ-VAE substantially improves both distribution-level and local token consistency. 

Adding temporal alignment gives the best results on this segmentation-specific evaluation, supporting the intended effect of the transformation-minimized reconstruction loss. The full transformation setting performs worse than temporal alignment alone on this metric. This is expected because the additional pitch-domain transformations are not 

designed specifically for segmentation robustness; rather, they allow similar contour shapes to be matched after pitch shifting or pitch-range scaling. As a result, similar contours may be represented by different tokens whose decoded outputs can still be adjusted to fit the same target, which can reduce exact token consistency under shifted segmentations. Their benefit is therefore evaluated more directly in the downstream _sigimsae_ classification experiment. 

## **6. SIGIMSAE CLASSIFICATION** 

If the learned tokens are to serve as analytical units, their patterns should align with expert-defined ornament categories. We test this with a minimal probing setup, a pertoken MAP classifier — that isolates how much labeldiscriminative information is carried by individual tokens. 

## **6.1 Sigimsae Labeled Dataset** 

We employed the annotated Korean traditional music recordings from the NIA AI-Hub dataset [22]. The dataset consists of annotated _sigimsae_ segments from monophonic Korean traditional music recordings. Because it is highly class-imbalanced and contains both vocal and instrumental performances, we report results in two settings: the full setting, with 34.0 hours of audio and 42.3k annotated segments totaling 9.3 hours across six classes, and the vocalonly setting, with 10.0 hours of audio and 13.5k annotated segments totaling 2.4 hours across five classes. 

The data are split at the track level with per-class stratification into train, validation, and test sets (approximately 8:1:1), ensuring that segments from the same track do not appear in multiple splits. 

## **6.2 Token-to-label lookup classifier** 

For each annotated _sigimsae_ segment, the center portion is cropped to the receptive field size and encoded into a single token. Segments shorter than the receptive field are padded with surrounding context. 

We use a simple frequency-based classifier to test how much label-discriminative information is carried by individual tokens. This is not intended to build a strong supervised classifier. Rather, it serves as a probing task: if the tokens learned without _sigimsae_ labels are musically meaningful, their distribution should contain information that is predictive of expert-defined ornament categories. 

From the training set, we count how often each token occurs with each _sigimsae_ label and use these counts to estimate _P_ (token _|_ label). Classification is then performed using maximum a posteriori (MAP) estimation: 



where the temperature _τ ∈_ [0 _,_ 1] is selected on the validation set and applied to the test set. 

As a scale reference, we additionally report a supervised Encoder-Decoder Temporal Convolutional Network (EDTCN), which was employed for R¯aga ornamentation detection [23] trained directly on _sigimsae_ classification with 

|Method|_F_1_m_|_acro ↑_|_mA_|_P ↑_|
|---|---|---|---|---|
||All|Vocal|All|Vocal|
|Autoencoder|0.284|0.324|0.265|0.290|
|VQ-VAE|0.266|0.319|0.248|0.302|
|+ temporal align|0.285|0.320|0.272|0.317|
|+ all transformations|**0.285**|**0.331**|**0.278**|**0.324**|
|ED-TCN [23]|0.463|0.541|0.579|0.531|



**Table 2** . _Sigimsae_ Classification results on NIA AI-Hub dataset [22]. We report _F_ 1 _macro_ and _mAP_ for both all/vocal subset settings. 



<!-- Start of picture text -->
VQ-VAE Autoencoder<br>Token58  P=0.02 Token196  P=0.02<br>5.0<br>Southern-style 2.5<br>descending 2.5<br>flick 0.0<br>Token180  P=0.04 Token196  P=0.06<br>Southern-style 2 2.5<br>vibrato 0 0.0<br>Token135  P=0.07 Token16  P=0.04<br>2.5<br>Western-stylevibrato 0.0 0.0<br>2.5<br>Token4  P=0.13 Token196  P=0.03<br>1.5<br>2.5<br>Soft<br>vibrato 1.0 0.0<br>Token243  P=0.08 Token190  P=0.07<br>0 0<br>Upward<br>accent 5 5<br>Token178  P=0.07 Token196  P=0.03<br>1.5<br>2.5<br>Descending 1.0 0.0<br>slide 0.5<br>0 60 120 0 60 120<br>Timeframe (10ms)<br><!-- End of picture text -->

**Figure 2** . Decoded contours of the most representative token per _sigimsae_ category, selected by the highest _P_ ( _token | label_ ), for VQ-VAE (left) and Autoencoder baseline (right). 

median-normalized F0 contours. It uses full label supervision and is not directly comparable, but indicates the order of magnitude reachable with supervised contour-based classification. 

## **6.3 Results** 

As presented in Table 2, vanilla VQ-VAE alone does not outperform the autoencoder baseline in _F_ 1. Without alignment-aware training, similar contours are split across distinct codes, fragmenting per-class distributions. Adding the full transformation-minimized loss reverses this and yields the best probing scores across both settings, with a larger gain than temporal alignment alone. The gap to the supervised ED-TCN remains substantial, but a single token recovers roughly three-fifths of its _F_ 1 on the vocal setting without sequence modeling or label supervision during representation learning. 

Figure 2 visualizes the per-class prototype token (highest _P_ (token _|_ label)) for each model. For VQ-VAE, the six 



<!-- Start of picture text -->
   <br> <br><br>  <br> <br> <br> <br>  <br><br><br><!-- End of picture text -->

**Figure 3** . Decoded contours(black) of representative tokens for _Gyemyeonjo_ (top) and _Ujo_ , selected by the highest _P_ (mode _|_ token). Blue and red lines show the mediansubtracted original contours assigned to each token for _Gyemyeonjo_ and _Ujo_ , respectively. 

classes recruit six distinct tokens, each visually matching the canonical realization of its category: the upward-leapand-glide of _southern-style descending flick_ (Token 58), the monotonic descent of _descending slide_ (Token 178), and three vibrato classes mapped to tokens with clearly distinct oscillation profiles. For the autoencoder, a single token (Token 196) is selected as the prototype for four of the six classes. The autoencoder’s embeddings entangle phase, pitch level, range, and shape, so K-Means cuts across ornament types. The transformation-minimized loss removes these factors during training, leaving contour shape as the main basis for the codebook. 

## **7. TOKEN ANALYSIS IN PANSORI** 

As a case study, we explore whether the learned tokenizer can serve as a tool for data-driven musicological analysis by examining how individual tokens relate to the two principal modes of _pansori_ . 

**Dataset:** _Pansori_ has two representative modes: _Gyemyeonjo_ and _Ujo_ , each characterized by distinct melodic and ornamental patterns. We used a mode-annotated subset [24] of our in-house _pansori_ dataset, which was annotated by an expert. The dataset consists of 26.7 hours of _Gyemyeonjo_ annotations and 8.7 hours of _Ujo_ annotations. 

**Method:** A _pansori_ mode is largely defined by its scale, but each mode is also known to carry characteristic ornamental patterns. Because our tokens are mediannormalized and thus carry no absolute pitch information, any mode-specific token usage must arise from contour shape itself. We therefore count token occurrences within mode-labeled segments and identify the tokens with the highest _P_ (mode _|_ token) for each mode, treating these as candidate ornamental patterns characteristic of each mode. 

**~~Result~~** ~~: Figure 3 shows the contour of those represe~~ n- ~~<u>tative tokens together with examples of input contours a</u>~~ s- ~~<u>signed to each token. To provide musical context, one</u>~~ of ~~<u>the co-authors also transcribed a slightly extended windo</u>~~ w around each example into staff notation. 

~~<u>Inspecting these tokens conf</u>~~ i ~~<u>rms that each is indee</u>~~ d ~~<u>characteristic of its associated mode. Token 87 captures th</u>~~ e ~~<u>upper turn gesture, and Token 141 captures a downwa</u>~~ rd ~~<u>gliding contour; both are widely described in</u>~~ _~~<u>pansori</u>~~_ ~~<u>stu</u>~~ d- ~~<u>ies as def</u>~~ i ~~<u>ning ornamental features of</u>~~ _~~<u>Gyemyeonjo</u>~~_ ~~<u>(Sco</u>~~ re ~~<u>Excerpts 24 and 3 in [25]). Tokens 171, in contrast, corr</u>~~ e- sponds to melodic phrase frequently observed in _ujo_ . 

~~<u>Token 54 is particularly informative. The descendin</u>~~ g ~~<u>major-second motion to the tonic that it captures is</u>~~ a ~~<u>melodic move shared by both</u>~~ _~~<u>Gyemyeonjo</u>~~_ ~~<u>and</u>~~ _~~<u>Ujo</u>~~_ ~~<u>, y</u>~~ et this specific token occurs predominantly in _Ujo_ . The distinguishing factor is the wide vibrato applied to the upper note in _Ujo_ before it resolves to the tonic. The tokenizer, trained without any mode supervision, separates the two modes along this ornamental dimension, a distinction that note-level transcription would discard since the underlying pitch sequence is identical. 

Although limited to a single case, this result indicates that the learned tokens can serve as analytical units linking raw contours to higher-level musicological categories. The same codebook, queried against other label sets such as schools or performers, offers a path toward corpus-level study of ornamental practice without requiring predefined ornament categories or note-level transcription. 

## **8. CONCLUSION AND FUTURE WORK** 

We proposed a VQ-VAE-based tokenizer that learns a discrete vocabulary of local pitch-contour patterns without ornament labels, together with a transformation-minimized reconstruction loss that makes the learned tokens robust to small temporal and pitch-domain variations. Experiments on Korean traditional music show improved segmentationshift consistency and _sigimsae_ classification over baselines, and qualitative analysis on _pansori_ shows that individual tokens separate _Gyemyeonjo_ and _Ujo_ even when the underlying pitch sequence is shared, a distinction that note-level transcription would discard. 

Our approach has one main limitation: our 1D-CNN tokenizer assumes a dense continuous contour and cannot directly model unvoiced regions, so we restricted training and evaluation to segments with few unvoiced frames. Extending the model to handle unvoiced regions explicitly is necessary for tokenizing entire recordings end-to-end. 

Two directions remain open for future work. The most immediate is corpus-level musicological analysis: converting long recordings into token sequences and studying transition statistics, n-gram distributions, or conditional patterns of ornamental units across modes, schools, and performers — analyses that have been routine for symbolic Western music but are not yet available for contour-centric traditions. The same framework is also in principle applicable to other such traditions, including Indian art music and pop vocal singing. 

## **9. ACKNOWLEDGEMENT** 

This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (RS-2025-00560548). 

## **10. REFERENCES** 

- [1] F. C. Moss, M. Neuwirth, D. Harasim, and M. Rohrmeier, “Statistical characteristics of tonal harmony: A corpus study of beethoven’s string quartets,” _PLoS One_ , vol. 14, no. 6, p. e0217242, 2019. 

- [2] D. Temperley, “Information flow and repetition in music,” _Journal of Music Theory_ , vol. 58, no. 2, pp. 155– 178, 2014. 

- [3] G. Widmer, S. Dixon, W. Goebl, E. Pampalk, and A. Tobudic, “In search of the horowitz factor,” _AI Magazine_ , vol. 24, no. 3, pp. 111–111, 2003. 

- [4] S. Chang, E. Benetos, H. Kirchhoff, and S. Dixon, “Yourmt3+: Multi-instrument music transcription with enhanced transformer architectures and cross-dataset stem augmentation,” in _2024 IEEE 34th International Workshop on Machine Learning for Signal Processing (MLSP)_ , 2024, pp. 1–6. 

- [5] T. Nuttall, X. Serra, and L. Pearson, “Svara-forms in carnatic music: Contextual influences on the performance of svara,” in _IEEE International Conference on Acoustics, Speech, and Signal Processing, ICASSP 2025 - Workshops, Hyderabad, India, April 6-11, 2025_ , 2025, pp. 1–5. 

- [6] J. Myers and D. Neuman, “Beyond notation: A digital platform for transcribing and analyzing oral melodic traditions,” in _Proceedings of the 26th International Society for Music Information Retrieval Conference_ , 2025, pp. 421–429. 

- [7] H. G. Ranjani, D. Paramashivan, and T. V. Sreenivas, “Quantized melodic contours in indian art music perception: Application to transcription,” in _Proceedings of the 18th International Society for Music Information Retrieval Conference, ISMIR 2017, Suzhou, China, October 23-27, 2017_ , S. J. Cunningham, Z. Duan, X. Hu, and D. Turnbull, Eds., 2017, pp. 174–180. [Online]. Available: https://ismir2017.smcnus.org/w p-content/uploads/2017/10/61_Paper.pdf 

- [8] H. G. Ranjani, A. Srinivasamurthy, D. Paramashivan, and T. V. Sreenivas, “A compact pitch and time representation for melodic contours in indian art music,” _The Journal of the Acoustical Society of America_ , vol. 145, no. 1, pp. 597–603, 01 2019. 

- [9] N. N. Shikarpur, K. M. Dendukuri, Y. Wu, A. Caillon, and C. A. Huang, “Hierarchical generative modeling of melodic vocal contours in hindustani classical music,” in _Proceedings of the 25th International Society for Music Information Retrieval Conference, ISMIR 2024,_ 

_San Francisco, California, USA and Online, November 10-14, 2024_ , 2024, pp. 1020–1028. 

- [10] S. Gulati, J. Serrà, V. Ishwar, and X. Serra, “Mining melodic patterns in large audio collections of indian art music,” in _2014 Tenth International Conference on Signal-Image Technology and Internet-Based Systems_ , 2014, pp. 264–271. 

- [11] T. Nuttall, G. Plaja-Roglans, L. Pearson, and X. Serra, “The matrix profile for motif discovery in audio - an example application in carnatic music,” in _Music in the AI Era - 15th International Symposium, CMMR 2021, Virtual Event, November 15-19, 2021, Revised Selected Papers_ , 2021, pp. 228–237. 

- [12] A. van den Oord, O. Vinyals, and K. Kavukcuoglu, “Neural discrete representation learning,” in _Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA_ , 2017, pp. 6306–6315. 

- [13] S. Han, H. Ihm, M. Lee, and W. Lim, “Symbolic music loop generation with neural discrete representations,” in _Proceedings of the 23rd International Society for Music Information Retrieval Conference, ISMIR 2022, Bengaluru, India, December 4-8, 2022_ , 2022, pp. 403– 410. 

- [14] P. Dhariwal, H. Jun, C. Payne, J. W. Kim, A. Radford, and I. Sutskever, “Jukebox: A generative model for music,” _CoRR_ , vol. abs/2005.00341, 2020. 

- [15] N. Zeghidour, A. Luebs, A. Omran, J. Skoglund, and M. Tagliasacchi, “Soundstream: An end-to-end neural audio codec,” _IEEE ACM Trans. Audio Speech Lang. Process._ , vol. 30, pp. 495–507, 2022. 

- [16] A. Défossez, J. Copet, G. Synnaeve, and Y. Adi, “High fidelity neural audio compression,” _Trans. Mach. Learn. Res._ , vol. 2023, 2023. 

- [17] J. Chorowski, R. J. Weiss, S. Bengio, and A. van den Oord, “Unsupervised speech representation learning using wavenet autoencoders,” _IEEE ACM Trans. Audio Speech Lang. Process._ , vol. 27, no. 12, pp. 2041–2053, 2019. 

- [18] S. Rouard, F. Massa, and A. Défossez, “Hybrid transformers for music source separation,” in _IEEE International Conference on Acoustics, Speech and Signal Processing ICASSP 2023, Rhodes Island, Greece, June 4-10, 2023_ , 2023, pp. 1–5. 

- [19] J. W. Kim, J. Salamon, P. Li, and J. P. Bello, “Crepe: A convolutional representation for pitch estimation,” in _2018 IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2018, Calgary, AB, Canada, April 15-20, 2018_ , 2018, pp. 161–165. 

- [20] T. Nuttall, X. Serra, and L. Pearson, “Svara-forms and coarticulation in carnatic music: an investigation using deep clustering,” in _Proceedings of the 11th International Conference on Digital Libraries for Musicology, DLfM 2024, Stellenbosch, South Africa, 27 June 2024_ , 2024, pp. 15–22. 

- [21] L. McInnes and J. Healy, “UMAP: uniform manifold approximation and projection for dimension reduction,” _CoRR_ , vol. abs/1802.03426, 2018. 

- [22] The Open AI Dataset Project (AI-Hub, S. Korea), “Korean traditional music score and audio dataset,” AIHub, 2022, dataset No. 71470. Available at https://ai hub.or.kr/aihubdata/data/view.do?dataSetSn=71470. 

- [23] S. Kumar, P. Singh, and V. Arora, “Recognizing ornaments in vocal indian art music with active annotation,” _CoRR_ , vol. abs/2505.04419, 2025. 

- [24] Anonymous, “Frame-level pansori mode classification with complementary audio representations,” 2026, submitted to ISMIR 2026. 

- [25] J. seon Jo, “A study on the Shigimsae of Pansori Kyemyunjo,” Master’s thesis, Hanyang University, Seoul, Republic of Korea, 2000, original title in Korean: `판소 리 계면조 시김새에 관한 고찰` [ `조주선` ].
