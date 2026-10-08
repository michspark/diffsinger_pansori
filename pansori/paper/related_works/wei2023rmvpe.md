---
title: "RMVPE: A Robust Model for Vocal Pitch Estimation in Polyphonic Music"
authors: "Haojie Wei; Xueke Cao; Tangpeng Dan; Yueguo Chen"
venue: "Proc. Interspeech"
year: 2023
bib_key: "wei2023rmvpe"
source_url: "https://www.isca-archive.org/interspeech_2023/wei23b_interspeech.pdf"
version: "published"
cited_in_paper: true
---

INTERSPEECH 2023 20-24 August 2023, Dublin, Ireland


# RMVPE: A Robust Model for Vocal Pitch Estimation in Polyphonic Music

_Haojie Wei_<sup>1</sup> _, Xueke Cao_<sup>2</sup> _, Tangpeng Dan_<sup>1</sup> _, Yueguo Chen_<sup>1</sup><sup>_,∗_</sup>

1School of Information, Renmin University of China, Beijing, China

2Institute of Information Science, Beijing Jiaotong University, Beijing, China

weihaojie@ruc.edu.cn, caoxueke19@163.com, tangpengdan@ruc.edu.cn, chenyueguo@ruc.edu.cn

## Abstract

Vocal pitch is an important high-level feature in music audio processing. However, extracting vocal pitch in polyphonic music is more challenging due to the presence of accompaniment. To eliminate the influence of the accompaniment, most previous methods adopt music source separation models to obtain clean vocals from polyphonic music before predicting vocal pitches. As a result, the performance of vocal pitch estimation is affected by the music source separation models. To address this issue and directly extract vocal pitches from polyphonic music, we propose a robust model named RMVPE. This model can extract effective hidden features and accurately predict vocal pitches from polyphonic music. The experimental results demonstrate the superiority of RMVPE in terms of raw pitch accuracy (RPA) and raw chroma accuracy (RCA). Additionally, experiments conducted with different types of noise show that RMVPE is robust across all signal-to-noise ratio (SNR) levels. **Index Terms** : vocal pitch estimation, polyphonic music, robust with noise

# 1. Introduction

Pitch Estimation (PE), also known as pitch tracking or fundamental frequency ( _f_ 0) estimation, is important in music signal processing. It plays a crucial role in melody extraction, and _f_ 0 can reflect various features of music audio and speech [1]. As a result, it is widely used in music information retrieval (MIR) [2] and speech analysis (SA) [3]. A specific case of pitch estimation is vocal pitch estimation, which is extracting vocal pitches from clean vocals. It should be noted that clean vocals are a type of monophonic music. While for extracting vocal pitches in polyphonic music, the vocal pitch estimation task becomes more challenging due to the presence of accompaniment.

The pitch estimation task in music information retrieval has been researched for decades. Many algorithms have been proposed to solve this problem, primarily falling into two categories: traditional heuristic methods and data-driven based methods. Traditional methods, such as YIN [4], SWIPE [5], and pYIN [6], employ specific candidate-generating functions to produce pitch results. These methods can get a well pitch result from clean vocals or other monophonic music. Data-driven based methods, including CREPE [7], DeepF0 [8], and HARMOF0 [9], utilize supervised training of CNN-based models for pitch estimation. These methods outperform traditional methods when applied to clean vocals. Additionally, data-driven based methods have the ability to process low-level noise as they can capture deep-level features that are more suitable for pitch estimation, even in low-noise environments.

The aforementioned methods belong to single-pitch estimation (SPE) since they are designed for monophonic music and predict no more than one pitch per frame. Consequently, their

> _∗_ Corresponding Author

performance sharply decreases when extracting vocal pitches from polyphonic music due to the presence of accompaniment. However, the polyphonic music, which consists vocals and accompaniment (e.g., pop music), has a large proportion in music data. It is more universal than clean vocals in music audio. Moreover, vocal pitch is an important high-level feature in polyphonic music and is effective for music information retrieval. Therefore, extracting vocal pitches from polyphonic music is an important and challenging task in music information retrieval. It should be noted that, the vocal pitch estimation task also belongs to SPE since it only extracts vocal pitches in polyphonic music, and each frame has at most one vocal pitch.

Some existing work has also tried to extract vocal pitches in polyphonic music using two main methods: the pipeline method and the end-to-end method. The pipeline method uses existing models to extract vocal pitches in polyphonic music. Firstly, a music source separation model (e.g., Open-Unmix [10], Demucs [11], Spleeter [12], etc.) is applied to extract clean vocals. Then, a pitch estimation model (e.g., pYIN [6], CREPE [7], HARMOF0 [9], etc.) is used to extract vocal pitches based on the predicted vocals. It is evident that the performance of vocal pitch estimation is highly related to both music source separation models and pitch estimation models. The end-to-end method designs a robust model to eliminate the influence of accompaniment. For example, CRN-Raw [13] uses a deep convolutional residual network to directly extract vocal pitches from the raw waveform of polyphonic music. JDC [14] employs a joint model of singing voice detection and classification to process the accompaniment. However, neither of these methods can extract highly accurate vocal pitches in polyphonic music.

In this paper, we propose a **R** obust **M** odel for **V** ocal **P** itch **E** stimation (RMVPE) in polyphonic music. RMVPE is inspired by music source separation models, which can extract clean vocals from polyphonic music. Our model utilizes deep U-Net and GRU to directly extract vocal pitches from polyphonic music. By this way, RMVPE can not only extract vocal pitches from polyphonic music, but also has robustness to different types of noise. Additionally, RMVPE have the comparable performance on clean vocals or other monophonic music.

# 2. Method

The vocal pitch estimation task in this paper belongs to SPE. To solve this task, we propose a robust model called RMVPE. The overall structure of RMVPE and the loss function are introduced in Section 2.1 and Section 2.2, respectively.

## 2.1. Model Architecture

We use the log mel-spectrograms X _T ×F_ as the input features and the probability matrix _YT ×_ 360 to represent the predicted pitches. In this way, the pitch estimation task can be formally written as, _F_ : _XT ×F → YT ×_ 360, where _T_ represents frames of the audio and _F_ is the logarithmically-spaced frequency bins


```
10.21437/Interspeech.2023-528
```

of spectrogram. As shown in Figure 1, RMVPE mainly con-


Skip Hidden Feature Filters<br>RCB (16, 16)<br>RCB (32, 32)<br>RCB (64, 64)<br>RCB (128, 128)<br>RCB (256, 256)<br>Encoder Layers Decoder Layers<br>Intermediate Layers<br>BN<br>Wav Input REB (1, 16) REB (16, 32) REB (32, 64) REB (64, 128) REB (128, 256) RDB (512, 256) RDB (256, 128) RDB (128, 64) RDB (64, 32) RDB (32, 16) Conv2D (16, 3) BiGRU FC+Sigmoid Pitch Prediction<br>Log Mel-Spectrogram ICB (256, 512) ICB (512, 512) ICB (512, 512) ICB (512, 512)<br>

Figure 1: _The overall structure of our proposed model RMVPE._

sists of four components, encoder layers, skip hidden feature filters, intermediate layers and decoder layers. The details of each component are introduced as follows:

The input features firstly pass through a batch normalization (BN) [15] layer. After that, there are encoder layers, which contain 5 residual encoder blocks (REBs). Each REB contains 4 residual convolutional blocks (RCBs) followed by an average pooling layer with a kernel size 2 _×_ 2 to extract the hidden features as shown in Figure 2(a). The details of RCB are shown


RCB<br>BN+ReLU RDB<br>REB Conv2D (out, out, 3) RCB (out, out)<br>Avgpool (2, 2) Conv2D (in, out, 1) RCB (out, out)<br>RCB (out, out) BN+ReLU RCB (out, out)<br>RCB (out, out) RCB (out * 2, out)<br>Conv2D (in, out, 3)<br>RCB (out, out) BN+ReLU<br>RCB (in, out) ConvTranspose (in, out)<br>(a) (b) (c)<br>

Figure 2: _(a) Residual Encoder Block (REB), (b) Residual Convolutional Block (RCB), (c) Residual Decoder Block (RDB)_

in Figure 2(b). There are two convolutional layers with 3 _×_ 3 kernels. After each convolutional layer, there is a BN layer and a ReLU function. Additionally, there is a shortcut connection between the input and the output of a RCB.

After the encoder layers, there are intermediate layers and skip hidden feature filters. The intermediate layers take the output of the encoder layers as input and contain four intermediate convolutional blocks (ICBs). Each ICB has the same structure as an REB without the average pooling layer. The skip hidden feature filters consist of 5 RCBs and take the output of each REB before the average pooling layer as input. The output of each RCB in the skip hidden feature filters is used as the input for the corresponding RDB in the decoder layers.

The decoder layers are the final component of RMVPE, and they consist of 5 residual decoder blocks (RDBs). Each RDB includes a transposed convolutional layer with a kernel size of 3 _×_ 3 and stride of 2 _×_ 2 to upsample feature maps, followed by 4 RCBs, as shown in Figure 2(c). After the decoder layers, a convolutional layer with a kernel size of 3 _×_ 3, a bidirectional GRU [16] with 256 units, and a fully connected sigmoid layer with 360 outputs are applied.

## 2.2. The Loss Function

Each frame of the output _YT ×_ 360 is a 360-dimensional vector _y_ , which represents the logarithmic scale pitches measured in terms of cents. The cent is a logarithmic unit of measure used for musical intervals relative to a reference pitch _fref_ in Hz, de-

fined as a function of frequency _f_ in Hz:


where we set _fref_ as 10 Hz here followed by [7]. The output vector _y_ corresponds to the frequency bin, which ranges from _C_ 1 (32.7 Hz) to _B_ 6 (1975.5 Hz) with 20 cents of intervals. By this way, the output vector _y_ fully covers the range of most melodic instruments including the singing voice.

When calculating the pitch value from _y_ ˆ, we use the local weighted average of pitches closest to the frequency bins with the highest peak value, as shown in Eq. 2:


Then the predicted frequency is calculated by Eq. 3. The _confi_ in Eq. 3 represents the voicing confidence, which reflects the confidence in the presence of a vocal pitch. In this paper, we set the default threshold ( _th_ ) to 0.5.


When training the model, we transform the ground-truth pitch of each frame to a 360-dimensional one-hot vector _y_ . To <mark>address the c</mark> lass imbalance between positive and negative cat- <mark>egories, we u</mark> se the weighted cross entropy loss. The loss function is <mark>def</mark> i <mark>ned as follo</mark> ws:


where we set _ω_ as 5, as it achieved the best performance in this paper.

# 3. Experiment

In this section, we firstly evaluate our model on three public polyphonic music datasets to show our model is practical in real music scenarios. Then we evaluate RMVPE with different types of noise to show our model is robust to noise. Finally, the experimental results show that RMVPE is also general with monophonic music since it has the comparable performance on clean vocals or other monophonic music.

## 3.1. Datasets and Evaluation Metrics

To compare with previous models fairly, we use four public datasets MDB-stem-synth [17], MIR-1K [18], Cmedia<sup>1</sup> and MIR ST500 [19]. We use raw pitch accuracy (RPA), raw chroma accuracy (RCA) and overall accuracy (OA), which are computed by mir ~~e~~ val[20] library to evaluate different models.

**Datasets.** _MDB-stem-synth_ is a monophonic music dataset for pitch estimation, which contains 230 resynthesized music files corresponding perfect _f_ 0 annotation. _MIR-1K_ is a dataset designed for singing voice separation, which has accompaniment on right channels and clean vocals on left channel. It also has pitch labels of vocal parts. _MIR ST500_ and _Cmedia_ are polyphonic music datasets, which have the YouTube URL of pop music and the pitch label of vocal parts.

1https://www.music-ir.org/mirex/wiki/2020: Singing_Transcription_from_Polyphonic_Music


**Evaluation Metrics.** Raw pitch accuracy (RPA) computes the proportion of melody frames in the reference for which the predicted pitch is within _±_ 50 cents of the ground truth pitch. While raw chroma accuracy (RCA) measures the raw pitch accuracy ignoring the octave errors. Overall accuracy (OA) computes the proportion of all frames correctly estimated by the algorithm, including whether non-melody frames where labeled by the algorithm as non-melody.

6.49% at RCA, by 2.74% at OA. This is because the encoder layers and decoder layers can directly extract effective hidden features from polyphonic music for vocal pitch estimation. It should be noted that RMVPE _Poly_ means the model is trained on polyphonic music data. The above results show that our model is effective in real music scenarios since it achieves state-of-theart performance on different polyphonic music datasets.

## 3.4. Results with Different Types of Noise

## 3.2. Experimental setup

The music audio is sampled at 16 kHz and then transformed into mel-spectrograms with log amplitude which has 256 mel bins. The hop length of mel-spectrograms is 320 (20ms) and the Hann window size is 2048. We cut the frequency between 30Hz and 8000Hz to extract the mel-spectrograms. librosa [21] is adopted to finish the above audio processing.

Our model is trained using Adam optimizer [22]. The learning rate is initialized as 0.0005 and reduced by 0.98 of the previous learning rate every 10 epochs and the batch size is set as 16. For MDB-stem-synth, MIR-1K and Cmedia datasets, we randomly spilt these datasets (at song-level) into training (80%) set and testing (20%) set. The spilt way of MIR ~~S~~ T500 is introduced in [19]. For all datasets, we split the audio recordings into 2.56-second segments when training the model.

## 3.3. Results on Polyphonic Music

We firstly compare RMVPE with several previous models on polyphonic music datasets MIR-1K, MIR ~~S~~ T500 and Cmedia. For PYIN [6], CREPE [7] and HARMOF0 [9], they are proposed for monophonic music, thus we firstly get clean vocals by Spleeter [12] and evaluate these models based on predicted vocals. The above methods belong to the pipeline method as illustrated in Section 1. For JDC [14] and CRN-Raw [13], they belong to the end-to-end method. We use the open checkpoint to evaluate JDC, since JDC is a joint model, which can process the accompaniment. The CRN-Raw model is retrained by using the same datasets as the RMVPE.

Table 1: _Average RPA(%), RCA(%) and OA(%) on three different polyphonic music datasets._

|Methods|Metrics||Datasets<br><br>||
|---|---|---|---|---|
|||MIR-1K|MIR<br>~~S~~T500|Cmedia|
||RPA|77.29±9.05|63.85±8.49|59.69±9.17|
|PYIN [6]|RCA|77.86±9.52|65.01±8.14|60.92±9.83|
||OA|71.79±9.97|65.27±7.66|60.25±8.73|
||RPA|91.05±8.46|81.61±8.03|74.61±5.92|
|CREPE [7]|RCA|92.16±7.30|82.36±7.70|75.05±5.57|
||OA|88.71±7.44|76.59±6.35|75.19±5.23|
||RPA|88.63±8.12|83.00±7.47|79.08±6.37|
|HARMOF0 [9]|RCA|88.97±7.61|83.35±7.09|79.64±5.95|
||OA|87.64±7.55|81.80±5.94|81.83±5.56|
||RPA|82.75±9.56|76.64±8.52|75.88±8.06|
|CRN-Raw [13]|RCA|91.92±9.39|82.48±7.96|79.86±7.23|
||OA|85.53±7.99|80.81±8.09|78.85±7.85|
||RPA|82.28±7.86|80.09±9.24|79.00±7.83|
|JDC [14]|RCA|82.98±7.57|80.38±9.08|79.25±7.57|
||OA|78.61±6.84|82.86±5.72|82.64±6.01|
||RPA|**95.42±3.97**|**89.32±5.64**|**83.57±5.62**|
|RMVPE_Poly_|RCA|**95.84±3.51**|**89.84±5.11**|**84.04±5.28**|
||OA|**91.86±5.08**|**84.54±5.17**|**85.09±5.01**|


As shown in Table 1, we find that our model has the best performance at RPA, RCA and OA compared with other models. Specifically, on MIR ~~S~~ T500 dataset, RMVPE _Poly_ outperforms the second best model HARMOF0 by 6.32% at RPA, by

Our model can not only predict vocal pitches from polyphonic music, but also extract vocal pitches with different types of noise. To verify RMVPE is more robust with different types of noise than previous models, we use the clean vocals in MIR1K [18] adding different types of noise to evaluate previous models and ours. Specifically, we add different types of noise by using Audio Degradation Toolbox (ADT) [23]. In this paper, we use four types of noise provided by ADT, white noise, pink noise, brown noise and pub noise. The white noise is a random signal which has constant power over all frequencies and the pink noise is a signal with a power density which decreases with increasing frequency. The brown noise is similar to pink noise, it has the highest power spectral density in low frequencies. The pub noise is a real recording of the sound in a crowded pub. We set the signal-to-noise ratio (SNR) value as 0 dB in this experiment.

We retrain CREPE [7] and HARMOF0[9] using clean vocals of MIR-1K here. RMVPE _Vocal_ and RMVPE _Poly_ have the same model structure, but they are trained on different types of music datasets. RMVPE _Poly_ is trained on mixture (polyphonic) music of MIR-1K and RMVPE _Vocal_ is trained on clean vocals of MIR-1K. The results of GIO [24] are copied from the original paper, because our experimental settings are the same as that in GIO. All results on vocals with different types of noise are summarized in Table 2.

Table 2: _Comparison with previous work with different types of noise when signal-to-noise ratio (SNR) is 0 dB on MIR-1K dataset (%)._

|Methods|Metrics|white|pink|brown|pub|Avg.|
|---|---|---|---|---|---|---|
|PYIN [6]|RPA|33.94|57.83|79.36|40.80|52.98|
||RCA|34.21|58.30|79.80|48.64|55.24|
|CREPE [7]|RPA|92.15|89.01|91.47|61.22|81.16|
||RCA|95.61|90.07|93.75|71.24|87.67|
|HARMOF0 [9]|RPA|74.61|74.85|97.18|61.62|77.07|
||RCA|75.18|76.41|97.24|65.05|78.48|
|JDC 14|RPA|70.65|71.32|85.25|72.44|74.92|
|[]|RCA|72.01|82.24|85.45|73.14|77.96|
|GIO 24|RPA|94.20|89.10|97.10|68.60|87.25|
|[]|RCA|94.90|91.40|97.50|74.70|89.63|
|RMVPE|RPA|**96.47**|**93.87**|97.78|**86.26**|**93.60**|
|_Poly_|RCA|**96.96**|**95.13**|97.95|**87.44**|**94.37**|
|RMVPE|RPA|96.09|93.55|**98.25**|82.13|92.51|
|_Vocal_|RCA|96.33|95.02|**98.30**|84.39|93.51|


For RMVPE _Poly_ and RMVPE _Vocal_ , the performance of RPA and RCA is better than the previous models. With the white, pink and pub noise, RMVPE _Poly_ performs better than RMVPE _Vocal_ . Especially for pub noise, RMVPE _Poly_ gets an improvement of 4.13% at RPA and 3.05% at RCA compared with RMVPE _Vocal_ . This is because pub is a more complex noise than others and RMVPE trained on mixture music is more robust to noise. While for brown noise, RMVPE _Vocal_ outperforms RMVPE _Poly_ by 0.47% at RPA, by 0.35% at RCA. This is because brown noise has most of its energy at low frequency and affects less to the pitch estimation results. These results show


White Noise Pink Noise Brown Noise Pub Noise<br>100<br>80<br>PYIN<br>60 CREPE<br>HARMOF0<br>40 JDC<br>RMVPEPoly<br>20 RMVPEVocal<br>40 30 20 10 5 0 40 30 20 10 5 0 40 30 20 10 5 0 40 30 20 10 5 0<br>SNR (dB) SNR (dB) SNR (dB) SNR (dB)<br>RPA (%)<br>

Figure 3: _The RPA (%) performance under different noise levels._

that RMVPE is stable at different training conditions. Thus, RMVPE is easier to use in practice than other models because training our model requires music data with less limitation (both polyphonic music and clean vocals are OK).

After that, Table 2 also shows that RMVPE achieves the highest performance with all types of noise compared with previous models. With pink noise, RMVPE _Poly_ outperforms the second best model GIO by 4.77% at RPA, by 3.73% at RCA. With pub noise, RMVPE _Poly_ outperforms the second best model GIO by 17.66% at RPA, by 12.74% at RCA. And there is an average improvement of 6.35% at RPA, 4.74% at RCA. The RMVPE improves significantly at different types of noise, especially with the pub noise. These results show that our model is more robust with different types of noise.

## 3.5. Noise Robustness

In order to further evaluate the performance of different models with different levels of noise. We add the noise to clean vocals with different signal-to-noise ratio (SNRs). The details of different types of noise and the settings of previous models are the same as Section 3.4. In this experiment, we use six different SNR values: 40, 30, 20, 10, 5 and 0 dB.

In Figure 3, we find that RMVPE _Poly_ maintains the highest performance at RPA, and the performance of RMVPE _Vocal_ is almost the same as RMVPE _Poly_ . This is because both of them can extract effective hidden features for vocal pitch estimation and RMVPE _Poly_ trained on polyphonic music is more robust to noise. It is obviously that the performance of different models decreases sharply when SNR is under 10 dB. Especially on pub noise, since the pub noise is a complex environmental noise. Brown noise is the exception where the performance decreases less, this is because brown noise has most of its energy at low frequencies. Moreover, our model decreases less than other models at all types of noise as shown in Figure 3. The second best model CREPE decreases 3.96% under white noise, 15.34% under pink noise, 4.48% under brown noise and 29.67% under pub noise from 10 dB to 0 dB. While RMVPE _Poly_ decreases 1.08% under white noise, 3.48% under pink noise, 0.08% under brown noise and 10.07% under pub noise from 10 dB to 0 dB. These results show that RMVPE has the ability to process different types of noise at all levels of noise.

## 3.6. Results on Clean Vocals

We also evaluate the performance of RMVPE on monophonic music (MDB-stem-synth [17]) and clean vocals (MIR-1K [18]). The compared algorithms are PYIN [6], CREPE [7], HARMOF0 [9] and JDC [14]. Different from experiments in Section 3.4, CREPE and HARMOF0 is retrained on MDB-stemsynth and clean vocals of MIR-1K.

Table 3: _Average RPA(%), RCA(%) and OA(%) on clean vocals or monophonic music datasets._

|Datasets|Methods||Metrics||
|---|---|---|---|---|
|||RPA|RCA|OA|
||PYIN [6]|65.83±21.29|67.01±21.36|77.57±16.33|
||CREPE [7]|97.50±4.25|97.97±3.22|98.41±2.93|
|MDB-stem-synth|HARMOF0 [9]|**97.94±2.19**|**98.02±2.12**|**98.47±1.76**|
||JDC [14]|62.61±26.43|62.81±26.50|68.03±30.69|
||RMVPE_Vocal_|97.11±2.70|97.12±2.69|97.68±2.09|
||PYIN [6]|74.71±10.37|74.99±9.48|79.07±7.31|
||CREPE [7]|95.66±4.07|96.52±2.87|95.56±2.90|
|MIR1K (l)|HARMOF0 [9]|96.07±3.54|96.58±2.84|96.31±2.43|
|- vocas|JDC [14]|68.96±10.78|69.54±10.56|66.85±9.86|
||RMVPE_Poly_|96.71±3.62|96.80±3.23|95.68±2.40|
||RMVPE_Vocal_|**97.27±2.35**|**97.28±2.35**|**96.70±1.74**|


All results are shown in Table 3. It can be observed that HARMOF0 achieves the highest performance on MDBstem-synth and RMVPE _Vocal_ achieves the highest performance on MIR-1K. On MDB-stem-synth dataset, RMVPE _Vocal_ decreases only 0.83% at RPA, 0.90% at RCA, 0.79% at OA compared with the best model HARMOF0. On MIR-1K dataset, RMVPE _Vocal_ outperforms the second best model HARMOF0 by 1.2% at RPA, 0.7% at RCA, 0.39% at OA. Besides, RMVPE _Poly_ only decreases 0.56% at RPA, 0.48% at RCA compared with RMVPE _Vocal_ , since RMVPE _Poly_ is trained on the mixture of MIR-1K, and the data distribution of mixture and clean vocals in MIR-1K is different. These results show that our model can also achieve a comparable performance with other models on monophonic music and clean vocals.

# 4. Conclusion

In this paper, we propose a robust vocal pitch estimation model RMVPE to extract vocal pitches from polyphonic music. Our model adopts deep U-Net to directly extract effective hidden features so as to have the ability of handling both accompaniment and noise. Extensive experiments on MIR-1K, MIR ST500 and Cmedia datasets show that our model gets the best performance for vocal pitch estimation in polyphonic music. Moreover, rich experiments on MIR-1K with different types of noise over all levels show that RMVPE is robust for noise. In the future, we will explore the lighter model to achieve the same performance for vocal pitch estimation in polyphonic music.

# 5. Acknowledgements

This work is supported by the National Science Foundation of China under the grant 62272466, and Public Computing Cloud, Renmin University of China.


# 6. References

- [1] J. Carroll, S. Tiaden, and F.-G. Zeng, “Fundamental frequency is critical to speech perception in noise in combined acoustic and electric hearing,” _The Journal of the Acoustical Society of America_ , pp. 2054–2062, 2011.

- [2] Y. V. S. Murthy and S. G. Koolagudi, “Content-based music information retrieval (cb-mir) and its applications toward the music industry: A review,” _ACM Computing Surveys (CSUR)_ , 2018.

- [3] M. K. Reddy and K. S. Rao, “Excitation modelling using epoch features for statistical parametric speech synthesis,” _Computer Speech & Language_ , 2020.

- [4] A. De Cheveign´e and H. Kawahara, “Yin, a fundamental frequency estimator for speech and music,” in _The Journal of the Acoustical Society of America_ , 2002, pp. 1917–1930.

- [5] A. Camacho and J. G. Harris, “A sawtooth waveform inspired pitch estimator for speech and music,” in _The Journal of the Acoustical Society of America_ , 2008, pp. 1638–1652.

- [6] M. Mauch and S. Dixon, “pyin: A fundamental frequency estimator using probabilistic threshold distributions,” in _Proceeding of IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , 2014, pp. 659–663.

   - [19] J.-Y. Wang and J.-S. R. Jang, “On the preparation and validation of a large-scale dataset of singing transcription,” in _Proceeding of IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , 2021, pp. 276–280.

   - [20] C. Raffel, B. McFee, E. J. Humphrey, J. Salamon, O. Nieto, D. Liang, D. P. Ellis, and C. C. Raffel, “mir ~~e~~ val: A transparent implementation of common mir metrics,” in _Proceeding of International Society for Music Information Retrieval (ISMIR)_ , 2014.

   - [21] B. McFee, C. Raffel, D. Liang, D. P. Ellis, M. McVicar, E. Battenberg, and O. Nieto, “librosa: Audio and music signal analysis in python,” in _Proceedings of the 14th python in science conference_ , 2015, pp. 18–25.

   - [22] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” _arXiv preprint arXiv:1412.6980_ , 2014.

   - [23] M. Mauch and S. Ewert, “The audio degradation toolbox and its application to robustness evaluation,” in _Proceeding of International Society for Music Information Retrieval (ISMIR)_ , 2013.

   - [24] X. Sun, X. Liang, Q. He, B. Zhu, and Z. Ma, “Gio: A timbreinformed approach for pitch tracking in highly noisy environments,” in _International Conference on Multimedia Retrieval (ICMR)_ , 2022, pp. 480–488.

- [7] J. W. Kim, J. Salamon, P. Li, and J. P. Bello, “Crepe: A convolutional representation for pitch estimation,” in _Proceeding of IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , 2018.

- [8] S. Singh, R. Wang, and Y. Qiu, “Deepf0: End-to-end fundamental frequency estimation for music and speech signals,” in _Proceeding of IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , 2021, pp. 61–65.

- [9] W. Wei, P. Li, Y. Yu, and W. Li, “Harmof0: Logarithmic scale dilated convolution for pitch estimation,” in _IEEE International Conference on Multimedia and Expo (ICME)_ , 2022, pp. 1–6.

- [10] F.-R. St¨oter, S. Uhlich, A. Liutkus, and Y. Mitsufuji, “Openunmix-a reference implementation for music source separation,” _Journal of Open Source Software_ , p. 1667, 2019.

- [11] A. D´efossez, N. Usunier, L. Bottou, and F. Bach, “Music source separation in the waveform domain,” _arXiv preprint arXiv:1911.13254_ , 2019.

- [12] R. Hennequin, A. Khlif, F. Voituret, and M. Moussallam, “Spleeter: a fast and efficient music source separation tool with pre-trained models,” _Journal of Open Source Software_ , p. 2154, 2020.

- [13] M. Dong, J. Wu, and J. Luan, “Vocal pitch extraction in polyphonic music using convolutional residual network.” in _Proceeding of INTERSPEECH_ , 2019, pp. 2010–2014.

- [14] S. Kum, J. Lee, K. L. Kim, T. Kim, and J. Nam, “Pseudo-label transfer from frame-level to note-level in a teacher-student framework for singing transcription from polyphonic music,” in _Proceeding of IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , 2022, pp. 796–800.

- [15] S. Ioffe and C. Szegedy, “Batch normalization: Accelerating deep network training by reducing internal covariate shift,” in _International Conference on International Conference on Machine Learning (ICML)_ , 2015, p. 448–456.

- [16] J. Chung, C¸ . G¨ulc¸ehre, K. Cho, and Y. Bengio, “Empirical evaluation of gated recurrent neural networks on sequence modeling,” _arXiv preprint arXiv:1412.3555_ , 2014.

- [17] J. Salamon, R. M. Bittner, J. Bonada, J. J. Bosch, E. G´omez Guti´errez, and J. P. Bello, “An analysis/synthesis framework for automatic f0 annotation of multitrack datasets,” in _Proceeding of International Society for Music Information Retrieval (ISMIR)_ , 2017.

- [18] C.-L. Hsu and J.-S. R. Jang, “On the improvement of singing voice separation for monaural recordings using the mir-1k dataset,” _IEEE Transactions on Audio, Speech, and Language Processing (TASLP)_ , pp. 310–319, 2010.
