---
title: "Montreal Forced Aligner: Trainable Text-Speech Alignment Using Kaldi"
authors: "Michael McAuliffe; Michaela Socolof; Sarah Mihuc; Michael Wagner; Morgan Sonderegger"
venue: "Proc. Interspeech"
year: 2017
bib_key: "mcauliffe2017mfa"
source_url: "https://www.isca-archive.org/interspeech_2017/mcauliffe17_interspeech.pdf"
version: "published"
cited_in_paper: true
---

_INTERSPEECH 2017_ August 20–24, 2017, Stockholm, Sweden


# Montreal Forced Aligner: trainable text-speech alignment using Kaldi

_Michael McAuliffe_<sup>1</sup> _, Michaela Socolof_<sup>2</sup> _, Sarah Mihuc_<sup>1</sup> _, Michael Wagner_<sup>1</sup><sup>_,_3</sup> _, Morgan Sonderegger_<sup>1</sup><sup>_,_3</sup>

1Department of Linguistics, McGill University, Canada

2Department of Linguistics, University of Maryland, USA

3Centre for Research on Brain, Language, and Music, McGill University, Canada

michael.mcauliffe@mail.mcgill.ca, msocolof@umd.edu, sarah.mihuc@mail.mcgill.ca, chael@mcgill.ca, morgan.sonderegger@mcgill.ca

## Abstract

We present the Montreal Forced Aligner (MFA), a new opensource system for speech-text alignment. MFA is an update to the Prosodylab-Aligner, and maintains its key functionality of trainability on new data, as well as incorporating improved architecture (triphone acoustic models and speaker adaptation), and other features. MFA uses Kaldi instead of HTK, allowing MFA to be distributed as a stand-alone package, and to exploit parallel processing for computationally-intensive training and scaling to larger datasets. We evaluate MFA’s performance on aligning word and phone boundaries in English conversational and laboratory speech, relative to human-annotated boundaries, focusing on the effects of aligner architecture and training on the data to be aligned. MFA performs well relative to two existing open-source aligners with simpler architecture (Prosodylab-Aligner and FAVE), and both its improved architecture and training on data to be aligned generally result in more accurate boundaries.

**Index Terms** : forced alignment, automatic segmentation, acoustic analysis

# 1. Introduction

In _forced alignment_ , speech and its corresponding orthographic transcription are automatically aligned at the word and phone level, given a way to map graphemes to phonemes (typically a pronunciation lexicon) and a statistical model of how phones are realized. Forced alignment has become widely used in scientific research on language over the past _∼_ 10 years, including in sociolinguistics, phonetics, language documentation, and psycholinguistics (e.g. [1, 2, 3, 4, 5]). This use has been driven by the availability of accurate, pre-built, and easily usable aligners, such as FAVE/P2FA, (Web)MAUS, and Prosodylab-Aligner [6, 7, 8]. We focus on this broad use case: forced alignment for language sciences using publicly-available software, when at least an orthographic transcript is available.<sup>1</sup>

Many such forced aligners now exist (e.g. [6, 7, 8, 12, 13, 14, 15, 16, 17]), which differ in two key ways. First, in _architecture_ , including the acoustic model used to model the realization of phones, and whether the acoustic features are transformed to account for speaker variability. Second, in _trainability_ : most aligners ship with pre-trained acoustic models only, while others can be retrained on new data [8, 17].

We describe the Montreal Forced Aligner (MFA), new open-source forced alignment software which is a successor to the Prosodylab-Aligner. MFA maintains Prosodylab-Aligner’s

> 1We do not address related work, such as on linguistic analysis of untranscribed speech [9], or phoneme boundary detection [10], or textspeech alignment for TTS [11].

trainability and updates its architecture. MFA uses triphone acoustic models to capture contextual variability in phone realization, in contrast to monophone acoustic models used in Prosodylab-Aligner and other current aligners (e.g. FAVE). MFA also includes speaker adaptation of acoustic features to model interspeaker differences. MFA uses the Kaldi speech recognition toolkit [18], which offers advantages over the HTK toolkit underlying most existing aligners.

We evaluate MFA’s performance on detecting word and phone boundaries in laboratory and conversational speech. Our experiments test whether the more complex architecture and trainability of MFA affect performance, by comparing to two existing monophone acoustic model aligners and varying the training data.

# 2. Montreal Forced Aligner

MFA is an open-source command line utility, with prebuilt executables for Windows and Mac OSX, and online documentation.<sup>2</sup> MFA is built on top of Kaldi, an actively maintained, open-source automatic speech recognition toolkit [18], and has three key usability features: it builds on the _trainability_ of Prosodylab-Aligner, and improves _portability_ and _scalability_ . The use of Kaldi as the ASR toolkit rather than HTK allows for easier distribution due to Kaldi’s more permissive license, so no compilation from source is required by the user. MFA’s use of Kaldi is highly parallel, which mitigates run time when using larger corpora and more computationally-intensive training.

The ASR pipeline that MFA implements uses a standard GMM/HMM architecture, adapted from existing Kaldi recipes. To train a model, monophone GMMs are first iteratively trained and used to generate a basic alignment. Triphone GMMs are then trained to take surrounding phonetic context into account, along with clustering of triphones to combat sparsity. The triphone models are used to generate alignments, which are then used for learning acoustic feature transforms on a per-speaker basis, in order to make the models more applicable to speakers in other datasets [19]. MFA has been successfully applied to 29 languages from GlobalPhone [20], the NCHLT corpora of South African languages [21], and other corpora.

MFA uses mel-frequency cepstral coefficients (MFCCs) as acoustic features. Thirteen MFCCs are calculated with a 25 ms window size and 10 ms frame shift. The feature calculation has a frequency ceiling of 8 kHz, allowing for acoustic models to be built and used regardless of sampling rate (i.e., models trained on 16 kHz sampled files can be applied to 44.1 kHz sampled files without manual resampling). Delta and delta-delta features from surrounding MFCC frames are also included, giving

2https://montrealcorpustools.github.io/ Montreal-Forced-Aligner/


**http://dx.doi.org/10.21437/Interspeech.2017-1386**

39 features per frame. Following MFCC generation, CMVN is applied to the features on a per-speaker basis to increase robustness to speaker variability. In the final round of training, feature transforms for each speaker are estimated using feature space Maximum Likelihood Linear Regression (fMLLR) [19]. Speaker adaptation is also done when aligning using pre-trained models, but can be disabled for faster alignment.

During training, MFA does 40 iterations of monophone GMM training, with realignment done during 20 of the iterations. Following monophone training, 35 iterations of triphone training are done, with 15 iterations that perform realignment. Speaker-adapted triphone training includes another 35 iterations with 15 realignment iterations, as well as 5 iterations that include fMLLR estimation. Multiprocessing is used extensively during feature calculation and training, allowing MFA to handle training and alignment of large corpora. For instance, the 1000hour LibriSpeech corpus was aligned in 80 hours (on a desktop using 12 3.4-Ghz processors, 32 GB memory), and training from scratch on the 20-hour Buckeye corpus (Sec. 3) took 2 hours (on a laptop using 4 2.5-GHz processors, 8GB memory).

MFA ships with a pre-trained model for English that has been trained on the LibriSpeech corpus [22] ( _∼_ 1000 hours of audiobooks), and pre-trained acoustic models (mostly from GlobalPhone corpora [20]) and grapheme-phoneme models for generating pronunciation dictionaries are publicly available in the online documentation for 20+ languages. A key feature of MFA is trainability of acoustic models on new data, as in the Prosodylab-Aligner [8]. Thus, a user can align their dataset either using pre-trained models, or by training from scratch on the dataset. Alignment can be significantly better when using acoustic models trained from scratch—especially when the dataset to be aligned is sufficiently large and varied. We recommend experimenting with pre-trained models and retraining, as it is an empirical question which method gives better alignments.<sup>3</sup> The experiments in Section 3 address this question.

There are two primary transcription formats used in current forced aligners, exemplified by Prosodylab-Aligner and FAVE. Prosodylab-Aligner aligns short wav files, each with an associated text file specifying the transcription. This format is common to lab speech where individual trials keep speech segments naturally short. FAVE aligns long files containing time-aligned periods of transcribed speech, a format more common to sociolinguistic data and spontaneous speech. MFA supports both formats, building on the Prosodylab-Aligner format and adding support for Praat [23] TextGrids as a way to specify transcriptions in longer sound files. The TextGrid format allows for the user to specify transcriptions for multiple speakers in the same file. The output of alignment is then a TextGrid for each input file, with separate word and phone tiers for each speaker.

MFA contains other upgrades to the Prosodylab-Aligner. Instead of requiring every word in the transcripts to be in the pronunciation dictionary, MFA includes an explicit model for unknown words as having a unique phone, which allows them to be modeled while maintaining alignment of surrounding words. The unknown word’s phone is constructed similarly to the silence phone, and can match any amount of vocal noise or speech (e.g. words of different lengths). Before performing alignment, MFA prompts the user if unknown words are found, including their location, to deal with simple typos for existing words. Anecdotally, MFA’s alignment quality remains very good when up to 5–10% of word types are unknown.

> 3Similarly, disabling speaker adaptation may lead to better alignments if there is little enough data per speaker.

A common source of alignment errors in read speech like audio books or laboratory experiments is deviations from the prompt, such as filled pauses, restarts, or speech errors. Transcriptions of spontaneous speech often contains analogous transcription errors, since listeners are prone to filtering out such deviations. Rather than manual inspection of each audio file for deviations from the transcription, MFA offers a feature from Kaldi to facilitate finding and correcting them. A limited lexicon per utterance is generated, supplemented with frequent words, and a simple speech recognition pass is run on the file to generate a transcript. This generated transcript is compared to the original transcript and deviations are saved to facilitate manual inspection.

# 3. Evaluation

Our evaluation of MFA addresses three questions: (1) how good is the aligner’s performance relative to manual annotation, and what is the effect on performance of the two key aspects of MFA: (2) architecture (acoustic model and speaker adaptation) and (3) trainability? We evaluate MFA’s performance by examining its accuracy on detecting phone and word boundaries in two datasets, representing types of speech commonly used in language research: isolated-word lab speech and conversational interview speech. We compare MFA to two existing widely-used aligners with simpler architectures—FAVE and Prosodylab-Aligner—and vary the training data for aligners where possible.

## 3.1. Datasets

The first dataset used in our evaluations was the Buckeye Corpus [24], which contains 20.7 hours of conversational speech from 40 speakers. Buckeye comes with manual transcription and boundaries at the phone and word level, which were produced by forced alignment followed by manual correction. The Buckeye phone set represents more subphonemic detail (e.g. flapping) than needed for our evaluations; we thus mapped it to the phone set used in our pronunciation dictionary (see below).

HTK-based aligners, such as FAVE and ProsodylabAligner, require relatively short speech chunks. We thus broke up Buckeye into chunks bounded by non-speech (pauses, noise, interviewer speech) of _>_ 150 msec marked in the transcription files, using PolyglotDB.<sup>4</sup> Each of these chunks consists of an orthographic transcription and speech, as well as corresponding word and phone-level manual alignments. In our evaluation, the transcription and speech are force-aligned, and the manual alignments used as the gold standard.

Utterances were excluded if they contained words not in the pronunciation dictionary used in evaluation, for comparability between FAVE/Prosodylab-Aligner (which require all words to be in the dictionary) and MFA (which does not).

The second dataset, Phonsay, consists of 48 minutes of lab speech from 45 participants from two experiments. Participants said words in the frame ”Please say again”. The target words all contained vowels followed by a consonant: a voiced obstruent, unvoiced obstruent, or sonorant (e.g. _buzz_ , _bus_ , _bun_ ). The boundaries of the vowel and the following consonant were hand-annotated, and these manual annotations are the gold standard in our evaluation.

In the evaluation, we examine two kinds of boundaries. First, left and right _word boundaries_ , across all words, for Buckeye only. (Most word boundaries in Phonsay were not anno-

4https://github.com/MontrealCorpusTools/PolyglotDB


Word boundaries (Buckeye) Phone boundaries (Buckeye) Phone boundaries (Phonsay)<br>8000<br>40000<br>600<br>6000<br>30000<br>400<br>4000<br>20000<br>10000 2000 200<br>0 0 0<br>1 10 100 1000 1 10 100 1000 1 10 100 1000<br>Absolute manual/aligned difference (msec) Absolute manual/aligned difference (msec) Absolute manual/aligned difference (msec)<br>count count count<br>

Figure 1: _Histograms of absolute differences (on log scale) between force-aligned word and phone boundaries using MFA-LS aligner and gold-standard annotations. Dashed line is at 1/2 frame rate (5 msec), which is a lower bound on average absolute difference._

tated.) Second, _phone boundaries_ , for each phone boundary of CVC words in either dataset, that corresponds to a manuallyannotated boundary. For Buckeye, this is all four boundaries (denoted .CVC, C.VC, CV.C, CVC.). The CVC words in Buckeye were those from the list of [25], with the additional criterion of having all three segments realized in some way according to the manual transcription. For Phonsay, the boundaries were C.VC, CV.C, and CVC. for the target word in every sentence.

Table 1: _Accuracies at different tolerances (percentage below a cutoff) for absolute differences between force-aligned boundaries using MFA-LS aligner, and gold-standard annotations._

||_<_10|Toleran<br>_<_25|ce (ms)<br>_<_50|<br>_<_100|
|---|---|---|---|---|
|Word boundaries (Buckeye)|0.33|0.68|0.88|0.97|
|Phone boundaries (Buckeye)|0.41|0.77|0.93|0.98|
|Phone boundaries(Phonsay)|0.36|0.72|0.88|0.95|


## 3.2. Aligners and training

Our evaluation uses MFA and two HTK-based aligners which are currently used in language research: FAVE, the most widely-used aligner in recent work, and Prosodylab-Aligner (PLA). PLA and FAVE are used as representative of aligners using GMM-HMM monophone acoustic models<sup>5</sup> without speaker adaptation, which are and are not trainable, respectively. Many existing aligners fall into these two categories (e.g. [6, 7, 8, 15]).

In order to minimize out-of-vocabulary words for PLA and FAVE, the pronunciation dictionaries which ship with each of the three aligners were combined into one Arpabet-based dictionary, which was used across all three aligners for training (MFA, PLA) and alignment (MFA, PLA, FAVE).

Both MFA and PLA were trained in two ways: on the LibriSpeech corpus, and on the corpus to be aligned: Buckeye (the subset without unknown words) or Phonsay. For training on LibriSpeech, MFA was trained on the full corpus ( _∼_ 1000 hours), while PLA was trained on the ‘clean’ subset ( _∼_ 450 hours), due to technical difficulties in HTK training on large datasets. For training on Buckeye, we treated the corpus as if only utterance boundaries and the orthographic transcription were known, to simulate the most common case in aligning speech in linguistic research. We refer to the resulting trained aligners as _MFA-LS_ , _MFA-Retrained_ , _PLA-LS_ , and _PLARetrained_ , where the “retrained” aligners refer to the version trained on Buckeye or the version trained on Phonsay, when discussing each corpus. We also used the existing version of _FAVE_ , which uses acoustic models trained on the SCOTUS corpus (25 hours) [26]. Thus, our experiments compare five types of aligner ( _MFA-{LS, Retrained}_ , _PLA-{LS, Retrained}_ , _FAVE_ ).

Each type of aligner was applied to align the Buckeye and Phonsay datasets, resulting in predicted word and phone boundaries. Note that we did not split the datasets into training and

5While it is possible to use triphone models in HTK, all distributed software packages for alignment use monophone models.

test sets, as the common use case for a trainable aligner is to simultaneously train on and align the entire dataset of interest.

Our evaluation considers two subsets of the predicted boundaries, described above: word boundaries (Buckeye only), and phone boundaries (Buckeye and Phonsay). The metric we use for accuracy of a force-aligned boundary is the absolute difference (in msec) from the manually-annotated boundary.

## 3.3. Results

Our results address questions (1)–(3): how good are MFA’s alignments ‘out of the box’ compared to hand annotation, and do the more complex architecture and trainability of MFA lead to more accurate alignments?

### _3.3.1. Alignment quality_

We first consider the performance of MFA-LS, which is the version distributed with the current version of MFA. Performance on the two datasets approximates the performance a user can expect if MFA-LS is applied to lab (Phonsay) or conversational (Buckeye) English data, without retraining.

Figure 1 and Table 1 show the distribution of manual/forcealigned differences, for each kind of boundary, for the two datasets. The distributions of differences are highly rightskewed, as for other forced aligners [8, 26]: 2–5% of tokens have differences of at least 100 msec, while about 90% have differences of less than 50 msec. Table 2 (row 1) gives the mean and median of manual/aligned boundary differences for each case. These measures can be compared for the Buckeye corpus to differences between human transcribers reported by [27]— bearing in mind that the set of word and phone boundaries used there differs from the set used in our evaluation.

For word boundaries, the mean manual/aligned difference is 24 msec, which is comparable to 26 msec intertranscriber


Table 2: _Comparison of aligners in detecting word boundaries (Buckeye only) and phone boundaries (Buckeye and Phonsay). Means and medians are over differences between aligned and gold-standard boundaries._

||Word b<br>Buck|ound.<br>eye|P<br>Buck|hone bo<br>eye|undaries<br>Phon|say|
|---|---|---|---|---|---|---|
|Aligner|mean<br>(ms)|med<br>(ms)|mean<br>(ms)|med<br>(ms)|mean<br>(ms)|med<br>(ms)|
|MFA-LS|24.1|15.8|**17.0**|**11.2**|25.2|11.3|
|MFA-Retrained|**22.6**|**15.0**|17.3|11.8|**16.6**|**10.8**|
|PLA-LS|30.5|15.6|24.0|13.9|40.1|21.5|
|PLA-Retrained|27.2|15.6|24.7|15.8|25.9|16.5|
|FAVE|24.7|16.6|19.3|12.0|21.8|13.0|


reliability [27]. 68% of manual/aligned differences are under 25 msec, which is significantly lower than the 90% intertranscriber agreement reported at 26 msec tolerance.

For phone boundaries, the mean difference is 17 msec for Buckeye and 25 msec for Phonsay. For Buckeye, an identical figure (17 msec) is reported for intertranscriber agreement [27]. The median difference is comparable (11 msec) for Phonsay and Buckeye, suggesting that the main difference between them is more gross misalignments for Phonsay (visible in Fig. 1 right).

In sum, MFA performs well across both datasets and boundary types. While phone and word-level alignment is comparable to human annotators on average, the force-aligned boundaries do contain more medium-to-large alignment errors ( _>_ 25 msec).

### _3.3.2. Architecture_

To examine the effect of MFA’s more complex architecture— triphone acoustic models and speaker-adapted features, compared to monophone acoustic models without speaker adaptation—we compare MFA-LS to PLA-LS and FAVE. The comparison with PLA-LS is most important, since MFA is essentially the same as PLA except for this modified architecture. Rows 1, 3, 5 of Table 2 show, for these three aligners, the mean and median differences between manual and forcealigned boundaries for each condition. In most cases (columns of Table 2), the ordering is MFA-LS _<_ FAVE _<_ PLA-LS. However, MFA-LS and PLA-LS have roughly the same median for word boundaries for Buckeye (below FAVE), and FAVE has the lowest mean for phone boundaries for Phonsay.<sup>6</sup> Still, MFALS has the best overall performance of the three aligners. The difference between MFA-LS and PLA-LS suggests that MFA’s different architecture led to better alignments.

To what extent is MFA’s performance in this comparison due to the updated acoustic model versus speaker adaptation? Experiments with a version of MFA with speaker adaptation disabled suggest that it is the triphone acoustic model that primarily accounts for MFA’s performance relative to PLA, with 88%/95% of the performance difference for word/phone boundaries (as measured by mean absolute manual/aligned difference) between PLA-LS and MFA-LS on Buckeye coming from just changing the acoustic model.<sup>7</sup>

### _3.3.3. Experiment 3: Training_

To examine the effect of retraining on the dataset to be aligned, we compare MFA-Retrained to MFA-LS and PLA-Retrained to

> 6All comparisons are significant (paired Wilcoxson rank-sum test).

> 7Disabling speaker adaptation gives _better_ performance as measured by the median, suggesting that enabling speaker adaptation may induce more gross errors, while increasing mean alignment accuracy.

PLA-LS. This comparison represents a common use case: a researcher has a medium-to-large dataset (say 5–50 hours) of speech from speakers of a given type (e.g. Buckeye: Columbusdialect adults). She can either re-train the aligner’s acoustic models on this data, or use acoustic models which have been pre-trained on a much larger dataset that contains significant interspeaker variation (e.g. LibriSpeech: 1000 hours). Will training on a smaller amount of more similar data or a larger amount of more variable data give better alignments?

The effect of retraining can be evaluated by comparing rows 1 and 2 of Table 2 for MFA, and rows 3 and 4 for PLA, again examining the mean and median of absolute differences between manual and aligned boundaries. In five cases (Buckeye word boundary mean for MFA/PLA, Phonsay phone boundary mean for MFA and mean/median for PLA), retraining leads to better performance, decreasing the mean or median difference by at least 1 msec. In six of the remaining seven cases, retraining makes little difference ( _<_ 1 msec mean or median). In only one case (Buckeye phone boundary median for PLA) does retraining lead to clearly worse performance ( _>_ 1 msec difference).

On balance, retraining on the dataset to be aligned often improves alignment accuracy relative to using acoustic models pretrained on a larger dataset—and rarely hurts. However, the discrepancy between mean and median values in some conditions suggests that a more thorough evaluation should examine the effect of retraining on gross alignment errors.

# 4. Conclusion

We have presented a new open-source trainable forced aligner for language research, the Montreal Forced Aligner, which updates the Prosodylab-Aligner. MFA uses more complex acoustic models (triphones), and is built using the Kaldi toolkit instead of HTK. MFA showed good performance in aligning word and phone boundaries in one lab speech dataset and one spontaneous speech dataset. Notably, in each test case (columns of Table 2), it is one of the MFA aligners which gives the most accurate alignment relative to the gold standard.

Our evaluation suggests that both MFA’s more complex architecture and the ability to retrain on new data generally improve performance. Using triphone acoustic models in particular seems to improve accuracy, compared to the monophone models commonly used in current aligners. More complex architectures, such as using artificial neural network models implemented in Kaldi (as in [14]), could improve accuracy further and are planned in future development. Retraining on the data to be aligned generally improved alignment accuracy, though it often had little effect—perhaps reflecting the similarity of training data for all aligners tested (North American English).

The mixed results of our evaluations point to the need for more thorough evaluations of forced aligners, to establish best practices for deploying forced alignment in language research [2, 28, 29]. Future work could examine the conditions under which adding speaker adaptation, or adapting an existing forced aligner versus retraining, improves alignment [30].

# 5. Acknowledgements

We acknowledge funding from SSHRC #430-2014-00018, FRQSC #183356 and CFI #32451 to MS, and SSHRC #4352014-1504 and the SSHRC CRC program to MW.


# 6. References

- [1] M. Adda-Decker and N. D. Snoeren, “Quantifying temporal speech reduction in French using forced speech alignment,” _Journal of Phonetics_ , vol. 39, no. 3, pp. 261–270, 2011.

- [2] C. DiCanio, H. Nam, D. H. Whalen, H. Timothy Bunnell, J. D. Amith, and R. C. Garc´ıa, “Using automatic alignment to analyze endangered language data: Testing the viability of untrained alignment,” _The Journal of the Acoustical Society of America_ , vol. 134, no. 3, pp. 2235–2246, 2013.

- [3] W. Labov, I. Rosenfelder, and J. Fruehwald, “One hundred years of sound change in Philadelphia: Linear incrementation, reversal, and reanalysis,” _Language_ , vol. 89, no. 1, pp. 30–65, 2013.

- [4] B. Schuppler, M. Ernestus, O. Scharenborg, and L. Boves, “Acoustic reduction in conversational Dutch: A quantitative analysis based on automatically generated segmental transcriptions,” _Journal of Phonetics_ , vol. 39, no. 1, pp. 96–109, 2011.

- [5] J. Yuan, M. Liberman, and C. Cieri, “Towards an integrated understanding of speaking rate in conversation,” in _Proceedings of Interspeech_ , 2006, pp. 541–544.

- [6] I. Rosenfelder, J. Fruehwald, K. Evanini, and J. Yuan, “FAVE (Forced Alignment and Vowel Extraction) Program Suite [Computer program],” 2011, available at http://fave.ling.upenn.edu.

- [7] T. Kisler, F. Schiel, and H. Sloetjes, “Signal processing via web services: the use case WebMAUS,” in _Digital Humanities Conference 2012_ , 2012.

- [8] K. Gorman, J. Howell, and M. Wagner, “Prosodylab-aligner: A tool for forced alignment of laboratory speech,” _Canadian Acoustics_ , vol. 39, no. 3, pp. 192–193, 2011.

- [9] S. Reddy and J. Stanford, “A web application for automated dialect analysis,” in _Proceedings of HLT-NAACL_ , 2015, pp. 71–75.

- [10] J. Yuan, N. Ryant, M. Liberman, A. Stolcke, V. Mitra, and W. Wang, “Automatic phonetic segmentation using boundary models,” in _Proceedings of Interspeech_ , 2013, pp. 2306–2310.

   - [21] E. Barnard, M. H. Davel, C. J. van Heerden, F. De Wet, and J. Badenhorst, “The NCHLT speech corpus of the South African languages,” in _Proceedings of SLTU_ , 2014, pp. 194–200.

   - [22] V. Panayotov, G. Chen, D. Povey, and S. Khudanpur, “Librispeech: an ASR corpus based on public domain audio books,” in _Proceedings of ICASSP 2015_ , 2015, pp. 5206–5210.

   - [23] P. P. G. Boersma _et al._ , “Praat, a system for doing phonetics by computer,” _Glot international_ , vol. 5, 2002.

   - [24] M. Pitt, L. Dilley, K. Johnson, S. Kiesling, W. Raymond, E. Hume, and E. Fosler-Lussier, _Buckeye Corpus of Conversational Speech (2nd release)_ . Department of Psychology, Ohio State University, 2007.

   - [25] S. Gahl, Y. Yao, and K. Johnson, “Why reduce? phonological neighborhood density and phonetic reduction in spontaneous speech,” _Journal of Memory and Language_ , vol. 66, no. 4, pp. 789–806, 2012.

   - [26] J. Yuan and M. Liberman, “Speaker identification on the SCOTUS corpus,” in _Proceedings of Acoustics ’08_ , 2008, pp. 5687–5790.

   - [27] W. D. Raymond, M. A. Pitt, K. Johnson, E. Hume, M. J. Makashay, R. Dautricourt, and C. Hilts, “An analysis of transcription consistency in spontaneous speech from the Buckeye corpus,” in _Proceedings of Interspeech_ , 2002.

   - [28] P. Milne, “The variable pronunciations of word-final consonant clusters in a force aligned corpus of spoken French,” Ph.D. dissertation, Universit´e d’Ottawa/University of Ottawa, 2014.

   - [29] T. Knowles, M. Clayards, M. Sonderegger, M. Wagner, A. Nadig, and K. Onishi, “Automatic forced alignment on child speech: Directions for improvement,” _Proceedings of Meetings on Acoustics_ , vol. 25, p. 060001, 2015.

   - [30] L. MacKenzie and D. Turton, “Crossing the pond: Extending automatic alignment techniques to British English dialect data,” 2013, talk given at _New Ways of Analyzing Variation 42_ .

- [11] A. Stan, P. Bell, and S. King, “A grapheme-based method for automatic alignment of speech and text data,” in _Spoken Language Technology Workshop (SLT), 2012 IEEE_ . IEEE, 2012, pp. 286– 290.

- [12] A. Pettarin, “Aeneas [computer program],” 2017, available at https://www.readbeyond.it/aeneas/.

- [13] R. Fromont and J. Hay, “LaBB-CAT: An annotation store,” in _Australasian Language Technology Association Workshop 2012_ , vol. 113, 2012, pp. 113–117.

- [14] R. M. Ochshorn and M. Hawkins, “Gentle forced aligner [computer program],” 2017, available at https://github.com/lowerquality/gentle.

- [15] J.-P. Goldman, “EasyAlign: an automatic phonetic alignment tool under Praat,” in _Proceedings of Interspeech_ , 2011, pp. 3233– 3236.

- [16] A. Katsamanis, M. Black, P. G. Georgiou, L. Goldstein, and S. Narayanan, “SailAlign: Robust long speech-text alignment,” in _Proceedings of Workshop on New Tools and Methods for VeryLarge Scale Phonetics Research_ , 2011.

- [17] B. Bigi, “SPPAS: a tool for the phonetic segmentations of speech,” in _Proceedings of LREC_ , 2012, pp. 1748–1755.

- [18] D. Povey, A. Ghoshal, G. Boulianne, L. Burget, O. Glembek, N. Goel, M. Hannemann, P. Motlicek, Y. Qian, P. Schwarz _et al._ , “The Kaldi speech recognition toolkit,” in _IEEE Workshop on Automatic Speech Recognition and Understanding_ , 2011, pp. 1–4.

- [19] D. Povey and G. Saon, “Feature and model space speaker adaptation with full covariance Gaussians,” in _Proceedings of Interspeech_ , 2006, pp. 1145–1148.

- [20] T. Schultz, N. T. Vu, and T. Schlippe, “GlobalPhone: A multilingual text & speech database in 20 languages,” in _Proceedings of ICASSP_ , 2013, pp. 8126–8130.
