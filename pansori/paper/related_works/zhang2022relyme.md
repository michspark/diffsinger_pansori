---
title: "ReLyMe: Improving Lyric-to-Melody Generation by Incorporating Lyric-Melody Relationships"
authors: "Chen Zhang; Luchin Chang; Songruoyao Wu; Xu Tan; Tao Qin; Tie-Yan Liu; Kejun Zhang"
venue: "Proceedings of the 30th ACM International Conference on Multimedia"
year: 2022
bib_key: "zhang2022relyme"
source_url: "https://arxiv.org/pdf/2207.05688"
version: "arXiv v1 (published at ACM Multimedia 2022; open preprint)"
cited_in_paper: true
---

# ReLyMe: Improving Lyric-to-Melody Generation by Incorporating Lyric-Melody Relationships

Chen Zhang<sup>∗</sup> zc99@zju.edu.cn Zhejiang University Hangzhou, China

Luchin Chang<sup>∗</sup> changluchin@gmail.com Zhejiang University Hangzhou, China

Songruoyao Wu<sup>∗</sup> 22021296@zju.edu.cn Zhejiang University Hangzhou, China

Xu Tan xuta@microsoft.com Microsoft Research Asia Beijing, China

Tao Qin taoqin@microsoft.com Microsoft Research Asia Beijing, China

Tie-Yan Liu tyliu@microsoft.com Microsoft Research Asia Beijing, China

Kejun Zhang<sup>†</sup> zhangkejun@zju.edu.cn Zhejiang University Alibaba-Zhejiang University Joint Institute of Frontier Technologies Hangzhou, China

## ABSTRACT

Lyric-to-melody generation, which generates melody according to given lyrics, is one of the most important automatic music composition tasks. With the rapid development of deep learning, previous works address this task with end-to-end neural network models. However, deep learning models cannot well capture the strict but subtle relationships between lyrics and melodies, which compromises the harmony between lyrics and generated melodies. In this paper, we propose ReLyMe, a method that incorporates <u>Relationships between Lyrics and Melodies from music theory to</u> ensure the harmony between lyrics and melodies. Specifically, we first introduce several principles that lyrics and melodies should follow in terms of tone, rhythm, and structure relationships. These principles are then integrated into neural network lyric-to-melody models by adding corresponding constraints during the decoding process to improve the harmony between lyrics and melodies. We use a series of objective and subjective metrics to evaluate the generated melodies. Experiments on both English and Chinese song datasets show the effectiveness of ReLyMe, demonstrating the superiority of incorporating lyric-melody relationships from the music domain into neural lyric-to-melody generation.

∗Both authors contributed equally to this research.

> †Corresponding author.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than ACM must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org. _Lisbon ’22, October 10–14, 2022, Lisbon, Portugal_

© 2022 Association for Computing Machinery. ACM ISBN 978-1-4503-XXXX-X/18/06...$15.00 https://doi.org/10.1145/3503161.3548357

# CCS CONCEPTS

• **Computing methodologies** → **Phonology / morphology** ; _Natural language generation_ ; • **Applied computing** → **Sound and music computing** .

# KEYWORDS

lyric-to-melody, knowledge-guided generation, domain knowledge, deep learning

# ACM Reference Format:

Chen Zhang, Luchin Chang, Songruoyao Wu, Xu Tan, Tao Qin, Tie-Yan Liu, and Kejun Zhang. 2022. ReLyMe: Improving Lyric-to-Melody Generation by Incorporating Lyric-Melody Relationships. In _Lisbon ’22: ACM MultiMedia, October 10–14, 2022, Lisbon, Portugal._ ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3503161.3548357

# 1 INTRODUCTION

Music has extremely versatile applications in daily life, commerce, and culture, yet the music composition process can be quite timeconsuming and expensive for humans. Recently, the progress of generative models [4, 29, 30] and the growing demand for music have further motivated the development of automatic music composition. Lyric-to-melody generation, one of the most important and common automatic music composition tasks, has attracted increasing attention from both academia and industry. Previous works addressed lyric-to-melody generation mainly with neural network models [1, 15, 17, 28, 34]. It is difficult for neural network models to capture the strict but subtle lyric-melody relationships that are crucial for the harmony between lyrics and melodies [9]. Therefore, previous deep learning lyric-melody generation systems face the problem that the generated melodies are dissonant with the lyrics. In this paper, we exploit the relationships between lyrics and melodies to ensure that the generated melodies are in harmony with the corresponding lyrics.

Lisbon ’22, October 10–14, 2022, Lisbon, Portugal

Chen Zhang, Luchin Chang, Songruoyao Wu, Xu Tan, Tao Qin, Tie-Yan Liu, and Kejun Zhang

In the last few decades, musicians and linguists have discovered that satisfying some musical relationships between lyrics and melodies is essential to guarantee the overall harmony in songwriting [23, 26, 33]. Any song, regardless of the genre, requires its composer to examine the interplay between the melodies and the lyrics. When some lyric-melody relationships have not been met, the songs will become dissonant, even leading to semantic misunderstandings. This kind of dissonance can be alleviated by adjusting the melodies to the lyrics or by adapting the lyrics to be consistent with the melodies on the basis of keeping the two in their own laws. Our ReLyMe is applied to lyric-to-melody generation systems and ensures the harmony between lyrics and melodies by tuning the generated melodies. However, our summary of the lyric-melodic relationships also applies to the melody-to-lyric generation.

In this paper, we introduce the principles of lyric-melody relationships based on the research from musicians and linguists. The relationships are from three aspects: 1) tone, the pitch flow of the generated melodies should be in line with the tone of the lyrics; 2) rhythm, the rhythm of the generated melodies need to match with the rhythm of the lyrics, consisting of the strong/weak positions and the pause positions; and 3) structure, the repetition patterns in generated melodies should echo the repetition in lyrics. Based on these principles, we formulate the rewards that influence the probability of the predicted token at each decoding step to integrate the lyric-melody relationships into neural network models.

Most previous works on lyric-to-melody generation have evaluated the melody itself alone, with limited consideration (only as a little part of the subjective evaluation) to evaluate the lyrics and melody as a whole. Considering that the melody is generated conditioned on the given lyrics, we introduce additional evaluation metrics to evaluate the relationships between lyrics and generated melodies. To verify the effectiveness of ReLyMe, we apply it in the state-of-the-art end-to-end and multi-stage lyric-to-melody generation systems respectively (SongMASS [28] and TeleMelody [15]) and conduct the experiments on both English and Chinese song datasets. Both the objective and subjective evaluation results indicate that ReLyMe can significantly reduce the lyrical and melodic dissonance of the original lyric-to-melody generation systems, thus further improving the overall quality of the generated melodies combined with the corresponding lyrics. The improvements on both two systems indicate that ReLyMe is flexible to be applied in different lyric-to-melody generation systems. Moreover, the method analyses verify the effectiveness of the detailed design of principles and the constrained decoding mechanism in ReLyMe. Some demos and our codes can be found in https://ai-muzic.github.io/relyme. To sum up, our main contributions are as follows:

- We formulate the relationships between lyrics and melodies from three aspects (tone, rhythm, and structure) based on the music theory summarized by musicians.

- We propose ReLyMe, a method that exploits lyric-melody relationships during the decoding process of neural lyricto-melody generation systems to ensure harmony between generated melodies and corresponding lyrics.

- We conduct experiments on both English and Chinese song datasets and evaluate the generated melodies with objective and subjective evaluation. The results demonstrate the

effectiveness of ReLyMe in guaranteeing the harmony between lyrics and melodies and show the significance of incorporating lyric-melody relationships into lyric-to-melody generation.

# 2 BACKGROUND

## 2.1 Lyric-to-Melody Generation

Early lyric-to-melody generation works are mainly based on musical rules [22] or statistical methods [6, 7, 20]. [6] regarded lyricto-melody generation task as an optimal-solution search problem under the constraints of lyrics. [22] proposed a rule-based rhythm generation module and then used an n-gram model to predict pitch. [20] built a melody composer based on the lyric-note relationships learned by frequent pattern mining. These methods often require much human effort to design the specific rules and suffer from the problem that generated melodies are monotonous.

Recently, with the rapid development of neural networks and generative models [13, 30], more and more researchers put their eyes on building automatic lyric-to-melody generation systems based on deep learning methods. [17] created melodies to accompany given text with two sequence-to-sequence models to generate the pitch and duration separately without considering other elements in melodies, such as silent intervals. [1] and [34] trained endto-end models to generate melodies based on lyrics. These methods usually require lots of paired data to train a robust lyric-to-melody generation model. [28] leveraged unpaired data by pre-training to cope with the lack of paired data. [15] proposed a two-stage generation pipeline based on musical templates to improve data efficiency and generation controllability. These deep learning methods implicitly model the relationships between lyrics and melodies, however, the subtle lyric-melody relationships are hard to fully capture by neural network models. Once the generated melody violates the criterion of the relationship with corresponding lyrics, it results in the dissonance between lyrics and melodies. In this paper, we propose ReLyMe to leverage lyric-melody relationships from music theory in lyric-to-melody generation systems to improve the performance.

## 2.2 Lyric-Melody Relationships in Songwriting

Earlier, musicians and linguists have studied the relationships between lyrics and melodies in the human songwriting process that influence the overall harmony [3, 8, 26, 33]. They analyzed and summarized these relationships mainly from the perspective of music theory and songwriting laws. [33] studied the lyric-melody relationships in Mandarin songs according to rules summarized during the human songwriting process, but no quantitative formulation was given. [26] summarized and formulated the pitch relationships between lyrics and melodies. In recent years, some researchers start to obtain lyric-melody relationships by statistical methods [23]. [21] built a music structure analyzer to derive the musical structure from melody and generated lyrics under the guidance of musical structures. [9] tackled the song translation problem by involving alignments of pitch, rhythm, and length, while it only considered the tonal languages. These works only considered the lyric-melody relationships from limited aspects or only apply to certain types of languages. In this paper, we discuss the relationships

ReLyMe: Improving Lyric-to-Melody Generation by Incorporating Lyric-Melody Relationships

Lisbon ’22, October 10–14, 2022, Lisbon, Portugal

between lyrics and melodies comprehensively from three aspects and propose ReLyMe which can be applied in different languages based on lyric-melody relationships.

## 2.3 Constrained Generation

Most neural generation tasks for discrete tokens such as machine translation, automatic speech recognition, and melody generation, are free generation without constraints. However, there is a need for some scenarios to involve constraints to guide the generation[9, 10, 16, 19, 25, 37]. Some works added constraints in the training stage, [16] investigated two methods for biasing the output length with a transformer architecture: length-ratio class or length embedding; [19] designed symbol sets and adds format embedding when training. Other works, on the other hand, guide the inference stage with constraints, [12] used a dictionary to control the beam search; [25] applied re-ranking after beam search with the combined score. In this paper, we choose to add constraints during the decoding and make a comparison with adding constraints after inference or in the training stage.

# 3 METHOD

ReLyMe takes effect upon current neural lyric-to-melody generation systems by incorporating relationships between lyrics and melodies. Considering that the generation process of most current neural network lyric-to-melody generation systems is autoregressive, ReLyMe involves lyric-melody relationships into neural network models at each decoding step. Specifically, during inference, ReLyMe adjusts the probabilities of each decoding step to influence the current generated token, and further to affect the subsequent tokens.

Denote the training lyric-melody pair as {( _𝒙, 𝒚_ ) ∈( **X** _,_ **Y** )}. The original score function of beam search in generation system at _𝑖_<sup>_𝑡ℎ_</sup> decoding step is:


At each decoding step, the constrained decoding mechanism adds the reward _𝑅𝑐𝑜𝑛𝑠_ directly to current score. _𝑅𝑐𝑜𝑛𝑠_ ( _𝑦_ 0: _𝑖,𝑥_ ) denotes the total reward at _𝑖_<sup>_𝑡ℎ_</sup> decoding step under the musical constraints, which consists of three parts: 1) the tone reward _𝑅𝑡_ ( _𝑦_ 0: _𝑖,𝑥_ ), 2) the rhythm reward _𝑅𝑟_ ( _𝑦_ 0: _𝑖,𝑥_ ), and 3) the structure reward _𝑅𝑠_ ( _𝑦_ 0: _𝑖,𝑥_ ). Thus, the score function in constrained decoding mechanism can be described as:


Then _𝑠𝑖_<sup>′is used in beam search as the new score function.</sup>

We define the above three rewards based on the relationships between lyrics and melodies. The lyric-melody relationships mean the combining relationships between the rules of lyrics and the rules of the melodies within the scope of unified content, which control the interplay between the lyrics and melodies. According to [33] which described the lyric-melody relationships of traditional Chinese folk music from the view of composer or musicians, these combining relationships are reflected in tone, rhythm, and structure. We follow these three aspects and extend them to different types of languages (tonal languages like Mandarin, “stress accent” languages like English). An example shows the principles we discuss below can be found in Figure 1.


**Figure 1: The music sheet of a Mandarin song “Rainbow” by Jay Chou. The colorful boxes illustrate the lyric-melody relationships we formulate.**


**Figure 2: The pitch shape of four main tones in Mandarin.**


**(a) A bad case that breaks pitch shape principle. The word “di f¯ang” of the lyrics means “place”, but it sounds like “dí f¯ang”, which means “enemy”.**


**(b) A bad case that breaks pitch transition principle. The “d¯eng tˇa” of the lyrics means “lighthouse”, but it sounds like “děng t¯a”, which means “waiting for him”.**

**Figure 3: Bad cases of tone principles.**

For a lyric sequence _𝒙_ , the corresponding syllable sequence is _𝒔_ = { _𝑠_ 1 _, ...,𝑠𝑛_ }, and the tone sequence is _𝒕_ = { _𝑡_ 1 _, ...,𝑡𝑛_ }. The melody _𝒚_ consists of a note sequence _𝒏_ = { _𝑛_ 1 _, ...,𝑛𝑚_ }( _𝑚_ ≥ _𝑛_ ) and one syllable can be matched with one note or multiple notes. In the following subsections, we introduce the principles and define the rewards (used in Equation 2) based on lyric-melody relationships from tone, rhythm, and structure respectively. And then we show the pipelines when applying ReLyMe to different lyric-to-melody generation systems.

Lisbon ’22, October 10–14, 2022, Lisbon, Portugal

Chen Zhang, Luchin Chang, Songruoyao Wu, Xu Tan, Tao Qin, Tie-Yan Liu, and Kejun Zhang


Pitch transition ( unit: semitone )<br>Adjacent tones -14 -13 -12 -11 -10 -9 -8 -7 -6 -5 -4 -3 -2 -1 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14<br>tone 1- tone 1<br>tone 1- tone 2<br>tone 1- tone 3<br>tone 1- tone 4<br>tone 1- tone 5<br>tone 2- tone 1<br>tone 2- tone 2<br>tone 2- tone 3<br>tone 2- tone 4<br>tone 2- tone 5<br>tone 3- tone 1<br>tone 3- tone 2<br>tone 3- tone 3<br>tone 3- tone 4<br>tone 3- tone 5<br>tone 4- tone 1<br>tone 4- tone 2<br>tone 4- tone 3<br>tone 4- tone 4<br>tone 4- tone 5<br>tone 5- tone 1<br>tone 5- tone 2<br>tone 5- tone 3<br>tone 5- tone 4<br>tone 5- tone 5<br>excellent good fair bad<br>

**Figure 4: The degree of harmony (excellent, good, fair, bad) between each pitch pair (melody) and each tone pair (lyric) in Mandarin. For each pitch pair, we calculate the pitch transition (pitch difference) of adjacent notes.**


**(a) A bad case that assigns an auxiliary word to downbeat. The question auxiliary “ma” is heard as “m¯a”, which means mother.**


**(b) A bad case that breaks the principle of pause positions. There is a pause caused by long notes inside a word (red arrow), while an appropriate pause should be at the blue arrow. This mismatch spoils the coherence of the song.**

**Figure 5: Bad cases of rhythm principles.**

## 3.1 Tone

When the lyrics are sung with given melodies, the pitches of the lyrics are determined by the melodies instead of the intrinsic tone or stress [35]. Aligning the pitch of the melody with the original pitch of the lyrics is essential for reducing semantic misunderstandings when performing songs [9, 33]. About 60% of languages have tone<sup>1</sup> (called tonal languages) [32], such as Chinese, Vietnamese, Thai. Tone is the pitch information in tonal languages that helps distinguish words with the same spelling but different meanings. In most tonal languages, a single tone is assigned to a syllable.

In order to align the melodies with the lyrics at the tonal level, we consider the following three granularities and define the corresponding rewards respectively. Denote the pitch sequence corresponding to note sequence as _𝒑_ = { _𝑝_ 1 _, ..., 𝑝𝑛_ }.

### _3.1.1 Pitch Shape of a Single Tone._

For each syllable in lyrics, when it corresponds to more than one notes in the melody, the predefined shape of the tone should coincide with the pitch flow of the corresponding note group. Take Mandarin as an example, which contains four main tones (high level tone, rising tone, falling-rising tone, falling tone, denoted as tone1, tone2, tone3, and tone4) and a light tone (unstressed syllable pronounced without its original pitch in Chinese pronunciation, denoted tone5) [11]. The predefined shapes of four main tones are shown in Figure 2 and an example that breaks this principle can be found in Figure 3a.

Thus, the pitch shape reward is only considered when more than one notes { _𝑛 𝑗 , ...,𝑛 𝑗_ + _𝑙_ } are assigned to a syllable _𝑠𝑖_ . Each tone of a syllable in tonal languages has its intrinsic pitch shape, when the pitch shape of the single tone is matched with that of corresponding notes, we assign the reward _𝑅𝑝𝑠_ ( _𝑦_ 0: _𝑖,𝑥_ ) = 1, otherwise, the reward is 0.

### _3.1.2 Pitch Transition between Adjacent Tones._

For a pair of adjacent syllables within the same sentence, the pitch flow of the corresponding adjacent notes needs to be in line with

1https://en.wikipedia.org/wiki/Tone_(linguistics)

the pitch transition between tones, otherwise may lead to semantic misunderstanding like Figure 3b. According to the practice in natural speech, we measure the degree of harmony (excellent, good, fair, poor) for the pitch difference between neighboring notes given each adjacent tone pair.

Given each adjacent tone pair { _𝑡𝑖_ −1 _,𝑡𝑖_ } in tonal languages, the pitch difference between the corresponding adjacent notes { _𝑛 𝑗_ −1 _,𝑛 𝑗_ } will be classified into four different harmony degrees (excellent, good, fair, bad) according to linguists and musicians (such as Mandarin in Figure 4 [27, 31]). We assign a reward _𝑅𝑝𝑡_ ( _𝑦_ 0: _𝑖,𝑥_ ) to _𝑖_<sup>_𝑡ℎ_</sup> decoding step based on the harmony degree of the pitch difference Δ _𝑝 𝑗_ = _𝑝 𝑗_ − _𝑝 𝑗_ −1 given the corresponding tone pair ( _𝑡𝑖_ −1 _,𝑡𝑖_ ). In practice, we assign the reward of 3, 2, 1, 0 when Δ _𝑝 𝑗_ belongs to the harmony degree of excellent, good, fair, and bad respectively.

### _3.1.3 Pitch Contour of a Whole Sentence._

Lyrics, as a form of text, have their inherent intonation for each sentence to express different emotions and meanings. The pitch contour of a whole sentence depends on the corresponding melody when performing a song, so the pitch direction of the melody needs to be matched to the inherent intonation of the sentence to ensure that the correct meaning is conveyed. For example, an interrogative tends to align with a rising melody (see the orange box of Figure 1).

Assume that { _𝑡𝑖_ − _𝑙 , ...,𝑡𝑖_ } is the tone sequence of a sentence, we calculate the reward of pitch contour _𝑅𝑝𝑐_ ( _𝑦_ 0: _𝑖,𝑥_ ) according to the inherent intonation of the sentence and the pitch contour of corresponding melody. That is to say, when the inherent intonation and the pitch contour are consistent, the reward is set to be 1, otherwise is 0.

The total tone reward _𝑅𝑡_ ( _𝑦_ 0: _𝑖,𝑥_ ) is the sum of the above three rewards. For tonal languages, the pitch of the melody is aligned with the tone of the lyrics, and all of the following three granularities are considered. While for “stress accent” languages such as English, which do not have tones, we only consider the pitch contour relationship.

ReLyMe: Improving Lyric-to-Melody Generation by Incorporating Lyric-Melody Relationships

Lisbon ’22, October 10–14, 2022, Lisbon, Portugal

## 3.2 Rhythm

We consider the rhythmic relationship in two dimensions: 1) intensity at a certain moment, reflected in strong and weak positions; 2) temporal alignment, reflected in pause positions. The total rhythm reward _𝑅𝑟_ ( _𝑦_ 0: _𝑖,𝑥_ ) is the sum of the following two rewards.

### _3.2.1 Strong/Weak Positions._

Both melodies and lyrics have the "strong" and "weak" elements, and the movement marked by the regulated succession of these elements is described as rhythm. A songwriter tends to align downbeats or accents with stressed syllables or important words<sup>2</sup> . Aligning the rhythm of melody and lyrics helps convey meaning, express empathy and render emotion. The strong and weak positions in melody lines could be determined by multiple factors [33]: pitch, duration, syncopation, meter, and so on. During automatic melody generation, for simplicity, the strong and weak beats can be derived from the predefined time signature [2]. In lyrics, on the other hand, the strong and weak positions are mainly influenced by oral speaking habits, specific semantics, intent of speakers, and expressed emotions. For example, in spoken language, the auxiliary words tend to be pronounced lightly, while the keywords in the lyrics are often regarded as accents, which are pronounced louder to attract the attention of listeners and emphasize the elements. The strong and weak positions of melodies are aligned with those of lyrics to ensure harmony of performance and correct meaning (a bad case is Figure 5a). Specifically, the keywords often correspond to the strong positions of the melody while auxiliary words should be placed on the weak positions. When the strong/weak positions of lyrics and melody match, the reward _𝑅𝑠𝑤_ ( _𝑦_ 0: _𝑖,𝑥_ ) is assigned to 1, otherwise 0.

### _3.2.2 Pause Positions._

Pauses in speech divide the entire sentence into several parts with separate meanings (called prosodic phrases) to help the listener understand. Sentences with the same content but different pause positions can have completely different meanings, and the division of prosodic phrases that defies common sense can lead to confusion for listeners. In melody, the pauses are caused by either _REST_ notes<sup>3</sup> or long notes [18] and split the melody into musical phrases, which make complete musical sense when heard on their own [5]. When the lyrics are sung with given melodies, the pause positions are no longer influenced by the spoken idiom but are controlled by the pauses in the melody. A phenomenon called “broken phrases” occurs if syllables of the same prosodic phrase are assigned to different musical phrases. This phenomenon impairs the harmony between the melody and the lyrics and makes the lyrics hard to understand for listeners. Thus, inappropriate _REST_ notes or long notes can be detrimental to the music. We summarize the following cases which lead to the dissonance and semantic misunderstanding (examples can be found in Figure 5b): 1) _REST_ notes are located inside the prosodic phrase; 2) long notes are assigned to non-final syllables of a prosodic phrase; 3) there are neither _REST_ notes nor long notes at the end of the sentence.

By word segmentation, we can obtain the positions that split words and sentences. The pause positions in melodies need to occur


**Figure 6: An example that shows the structure relationships between lyrics and melodies. The** 3<sup>_𝑟𝑑_</sup> **sentence repeats the** 1<sup>_𝑠𝑡_</sup> **sentence (see blue lines), and the** 4<sup>_𝑡ℎ_</sup> **sentence repeats the** 2<sup>_𝑛𝑑_</sup> **sentence (see green lines).**

at those “split” positions, otherwise prosodic phrases will be broken. Besides, there tends to be a corresponding pause in melody at the position which splits two sentences of lyrics. Thus, the reward _𝑅𝑝𝑠_ ( _𝑦_ 0: _𝑖,𝑥_ ) = 0 when the pause in a melody is not at a “split” position in corresponding lyrics or there is no pause in a melody between two sentences in the lyrics. Otherwise, _𝑅𝑝𝑠_ ( _𝑦_ 0: _𝑖,𝑥_ ) = 1.

## 3.3 Structure

At the full song level, the structure of lyrics and melody are constrained and influenced by each other. In music, _form_<sup>4</sup> refers to the structure of a musical composition or performance, including repetition, transition and development [36]. To better convey emotion and show artistry, lyrics and melodies which belong to the same musical form would share the same characteristics like rhyme (in lyrics) or motives (in melodies). Specifically, in lyric-to-melody generation, the melody segments that correspond to the lyrics of the same structure tend to have similar pitch flows (see Figure 6).

After the lyrics structure analysis, we can obtain a matrix _𝑀𝑠𝑡𝑟𝑢𝑐𝑡_ which indicates the structure of the lyrics. _𝑀𝑠𝑡𝑟𝑢𝑐𝑡_<sup>_𝑖𝑗_= 1 if</sup><sup>_𝑖𝑡ℎ_note</sup> and _𝑗_<sup>_𝑡ℎ_</sup> note are at the corresponding positions in the phrases of the same structure and _𝑀𝑠𝑡𝑟𝑢𝑐𝑡_<sup>_𝑖𝑗_= 0 otherwise. Thus, we assign the</sup> structure reward _𝑅𝑠_ ( _𝑦_ 0: _𝑖,𝑥_ ) = 2 if the pitch difference at _𝑖_<sup>_𝑡ℎ_</sup> note is equal to that of _𝑗_<sup>_𝑡ℎ_</sup> note (i.e. Δ _𝑝𝑖_ = Δ _𝑝 𝑗_ where i and j satisfy _𝑀𝑠𝑡𝑟𝑢𝑐𝑡_<sup>_𝑖𝑗_= 1 and</sup><sup>_𝑗< 𝑖_), that is the pitch sequences of two sentences</sup> are equal after overall shifting. And we assign _𝑅𝑠_ = 1 if Δ _𝑝𝑖_ − Δ _𝑝 𝑗_ is a multiple of 12. Otherwise, the structure reward is 0.

## 3.4 Applications to Different Lyric-to-Melody Generation Systems

ReLyMe is flexible to be applied to different lyric-to-melody generation systems, and in this work, we apply ReLyMe in two systems: 1) TeleMelody [15], a two-stage lyric-to-melody generation system with musical templates; and 2) SongMASS [28], an end-to-end lyricto-melody generation system with a pre-training process. These two are state-of-the-art systems in end-to-end and multi-stage lyricto-melody generation systems respectively.

**For TeleMelody.** TeleMelody consists of a lyric-to-template module and a template-to-melody module, using the template as the bridge to narrow the gap between lyric and melody, as well as improving the generation controllability by adjusting the template during the generating process. ReLyMe adjusts both the lyric-totemplate module and template-to-melody module, and guides them

> 2https://en.wikipedia.org/wiki/Prosody_(music)

> 3https://en.wikipedia.org/wiki/Rest_(music)

> 4https://en.wikipedia.org/wiki/Musical_form

Lisbon ’22, October 10–14, 2022, Lisbon, Portugal

Chen Zhang, Luchin Chang, Songruoyao Wu, Xu Tan, Tao Qin, Tie-Yan Liu, and Kejun Zhang

||Metric|Description|
|---|---|---|
|Melody|singability<br>diversity<br>listenability|Is the melody smooth and easy to sing?<br>Is the melody novel and not boring?<br>Is the melody appealing and pleasant?|
||intelligibility|Is it easy to understand the lyrics sung by<br>the given melody?|
|Melody<br>+ Lyrics|coherence<br>structure<br>overall|Are the melody and lyrics harmonious?<br>Are the melodies and lyrics structures<br>matched?<br>The overall perceptive score for the sample.|


# Table 1: Subjective evaluation metrics.

to generate templates or melodies which obey the principles. During the lyric-to-template module, we adjust the score of each step in beam search with the rhythmic constraints:


where _𝑧_ represents the output template of lyric-to-template module, _𝑅𝑟_ ( _𝑦_ 0: _𝑖,𝑥_ ) is the total rhythm reward of the _𝑖_<sup>_𝑡ℎ_</sup> decoding step and _𝜆𝑟_ represents a hyper-parameter that controls the influence of rhythmic constraint. Meanwhile, the relationships of tone and structure are added in template-to-melody module:


_𝑅𝑡_ ( _𝑦_ 0: _𝑖,𝑥_ ) and _𝑅𝑠_ ( _𝑦_ 0: _𝑖,𝑥_ ) are the tone reward and structure reward of the _𝑖_<sup>_𝑡ℎ_</sup> step and _𝜆𝑡_ and _𝜆𝑠_ refer to hyper-parameters of constraints of tone and structure respectively.

**For SongMASS.** SongMASS pre-trains an end-to-end model to generate melody from the given lyrics directly, so the constraints of all three aspects are incorporated to decoding process at once:


# 4 EXPERIMENTS

## 4.1 Experimental Settings

### _4.1.1 Datasets._

We follow the data collection pipeline of TeleMelody [15] and SongMASS [28] respectively for their training on both English and Chinese song datasets. Considering that ReLyMe requires the pitch contour of a whole sentence, we keep the original punctuations of the lyrics to indicate the intonation of sentences. Besides, we choose 100 English song samples and 100 Chinese song samples as evaluation datasets to assess the generation quality of ReLyMe applied in both TeleMelody and SongMASS. Each sample of the evaluation set does not appear in any of the above training sets. We attach the English and Chinese evaluation sets in https://github.com/microsoft/muzic.

### _4.1.2 System Configuration._

The top-K of stochastic sampling inference is set to 2 and 10 in two

modules of TeleMelody and 5 in SongMASS respectively to guide the model to generate melodies with more diversity. The temperature is set as 1.2 in both two modules of TeleMelody when studying the influence of hyper-parameters and set as 0.5 when conducting the evaluation. When apply ReLyMe to TeleMelody, we set the hyper-parameters _𝜆𝑡_ , _𝜆𝑟_ , _𝜆𝑠_ in constrained decoding as 1.2, 1.5, 1 respectively. And the hyper-parameters _𝜆𝑡_ , _𝜆𝑟_ , _𝜆𝑠_ in constrained decoding as 1.5, 1, 1 respectively in SongMASS. The code and configuration can be found in https://github.com/microsoft/muzic.

### _4.1.3 Evaluation Metrics._

We briefly introduce objective and subjective metrics here, more details about the formulation of objective metrics and the instruction of subjective evaluation are listed in Appendix.

**Objective Evaluation** The scores of transition and contour in tone aspect are derived from the rewards in Section 3.1. We report the ratios of matched strong/weak positions and matched pause positions for each sample to show the performance of rhythm relationship. As for structure relationship, following [15, 28], we calculate PD (Pitch Distribution Similarity), DD (Duration Distribution Similarity), and MD (Melody Distance) of the repetition parts [14] in generated melodies.

**Subjective Evaluation** To evaluate the perceptive quality of the generated melodies, we conduct subjective evaluation and compare ReLyMe with the original TeleMelody and SongMASS. We randomly select 15 samples from our test set for each experimental setting and distributed them to 10 participants with music knowledge. All participants are asked to give their scores of each sample from two aspects: 1) the quality of the generated melody itself, and 2) the quality of the overall sample (consider the melody and corresponding lyrics as a whole). The five-point scales (1 for bad and 5 for excellent) [24] are used to assess overall generation quality (see Table 1). We synthesize the singing voice from the lyrics and melodies using an internal singing voice synthesis tool and provide annotators with the music sheet, lyrics, and singing voice.

## 4.2 Main Results

We apply ReLyMe in TeleMelody and SongMASS and compare the original version of these two systems with the systems equipped with ReLyMe. The objective results in Table 2 show that both TeleMelody and SongMASS are improved after applying ReLyMe, verifying the effectiveness of ReLyMe and the necessity of incorporating lyric-melody relationships into automatic songwriting. For original TeleMelody and the one with ReLyMe, the gaps of tone scores and structure scores are much larger than those of rhythm, because the musical templates of TeleMelody have already contained the information of rhythmic patterns. In contrast, for SongMASS which does not explicitly capture rhythmic information, the improvement of rhythm scores is significant as tone and structure scores. Table 3 shows the results of subjective evaluation. We can draw the same conclusion as from objective results that ReLyMe effectively integrates the relationships between lyrics and melodies into neural network models. By adding rewards of lyric-to-melody relationships to the probability of each token in the decoding stage, ReLyMe not only improves the results of objective evaluation but more importantly, enhances the perception of audiences listening to the generated melodies.

ReLyMe: Improving Lyric-to-Melody Generation by Incorporating Lyric-Melody Relationships

Lisbon ’22, October 10–14, 2022, Lisbon, Portugal

|Language|Model|To|ne|Rhyth|m||Structur|e|
|---|---|---|---|---|---|---|---|---|
|||transition↑|contour↑|matched s/w positions↑|matched pauses↑|PD↑|DD↑|MD↓|
||TeleMelody [15]|0.45|0.26|0.61|0.72|0.73|0.79|2.20|
|Chinese|ReLyMe (in TeleMelody)|**0.61**|**0.30**|**0.73**|**0.79**|**0.79**|**0.82**|**1.53**|
||SongMASS [28]|0.61|0.23|0.29|0.86|0.61|0.44|2.20|
||ReLyMe (in SongMASS)|**0.68**|**0.36**|**0.65**|**0.93**|**0.69**|**0.57**|**0.95**|
||TeleMelody [15]|-|0.20|0.52|0.84|0.60|0.71|1.86|
|English|ReLyMe (in TeleMelody)|-|**0.35**|**0.69**|**0.92**|**0.64**|**0.73**|**1.44**|
||SongMASS [28]|-|0.25|0.44|0.82|0.61|0.54|1.54|
||ReLyMe (in SongMASS)|-|**0.40**|**0.54**|**0.93**|**0.66**|**0.58**|**1.24**|


**Table 2: Objective results of ReLyMe and baseline systems. Transition score and contour score of tone are calculated as described in Section 3.1. For rhythm, we count the average ratio of matched strong/weak positions and pauses respectively. PD, DD and MD are calculated over the repetition parts in melodies.**

|Language|Model||Melody|||Melody + L|yrics||Average|
|---|---|---|---|---|---|---|---|---|---|
|||singability|diversity|listenability|intelligibility|coherence|structure|overall||
||TeleMelody [15]|2.79|**3.29**|2.92|2.94|2.51|2.73|2.78|2.85 (±0_._07)|
|Chinese|ReLyMe (in TeleMelody)|**3.21**|3.20|**3.16**|**3.39**|**2.94**|**3.16**|**3.11**|**3.16 (**±0_._06**)**|
||SongMASS [28]|2.74|**3.20**|2.92|2.93|2.43|2.70|2.68|2.80 (±0_._07)|
||ReLyMe (in SongMASS)|**3.08**|3.12|**2.98**|**3.24**|**2.75**|**2.89**|**3.01**|**3.01 (**±0_._06**)**|
||TeleMelody [15]|3.27|3.45|3.19|3.78|3.32|3.21|3.23|3.35 (±0_._08)|
|English|ReLyMe (in TeleMelody)|**3.72**|**3.47**|**3.69**|**4.02**|**3.61**|**3.66**|**3.63**|**3.68 (**±0_._07**)**|
||SongMASS [28]|3.14|3.08|3.20|3.58|3.01|2.91|3.08|3.14 (±0_._07)|
||ReLyMe (in SongMASS)|**3.58**|**3.26**|**3.44**|**3.85**|**3.20**|**3.26**|**3.23**|**3.40 (**±0_._07**)**|


**Table 3: The results of subjective evaluation for ReLyMe and baseline systems. The average scores are calculated with 95% confidence intervals.**

|Model|To|ne||Rhyth|m||Structur|e|
|---|---|---|---|---|---|---|---|---|
||transition↑|contour↑|matched s/w po|sitions↑|matched pauses↑|PD↑|DD↑|MD↓|
|ReLyMe|0.61|0.35|0.73||0.79|0.79|0.82|1.53|
|–tone relationship|0.41|0.25|0.74||0.80|0.93|0.89|0.98|
|–rhythm relationship|0.58|0.33|0.57||0.69|0.76|0.93|1.41|
|–structure relationship|0.67|0.36|0.76||0.85|0.74|0.62|2.14|
|TeleMelody [15]|0.45|0.26|0.61||0.72|0.73|0.79|2.20|


**Table 4: The objective results of removing tone, rhythm, and structure relationship from ReLyMe respectively. The scores of corresponding metrics (gray blocks) decrease.**

## 4.3 Method Analyses

For simplicity, we performed all method analyses on TeleMelody using the Chinese song dataset.

### _4.3.1 Relationships of Different Aspects._

In this part, we remove tone, rhythm and structure relationship from ReLyMe respectively and compare the results. As shown in Table 4, when removing tone, rhythm or structure relationship, the scores of corresponding objective metrics decrease, which demonstrates the importance of each lyric-melody relationship we introduce.

Three relationships have some influence on each other, so ReLyMe with all three relationships does not get the highest score for each metric. For example, incorporating the tone relationship influences the structure scores. However, ReLyMe achieves a balance between the three aspects, producing the melodies with the highest overall quality.

_4.3.2 Soft Constraints v.s. Hard Constraints._ ReLyMe incorporates lyric-melody relationships by adding rewards to the original probabilities of neural network models and considers

Lisbon ’22, October 10–14, 2022, Lisbon, Portugal

Chen Zhang, Luchin Chang, Songruoyao Wu, Xu Tan, Tao Qin, Tie-Yan Liu, and Kejun Zhang


**Figure 7: The generated melodies of different settings given the same input lyrics.**

the rules of the melody itself (learned by neural network models) and lyric-melody relationships at the same time, which is regarded as using “soft constraints”. If we use “hard constraints” in ReLyMe, we force the generated melodies to keep in line with the lyrics exactly. As shown in Setting #2 of Table 5, the perceptive quality of the generated melodies is corrupted, indicating that we should balance the rules of the melody itself and lyric-melody relationships, rather than sacrificing the melodic rules to enforce conformity to the lyric-melody relationships.

### _4.3.3 Conditioned Training & Re-ranking._

In order to explain why we incorporate the lyric-melody relationships during the decoding process instead of in the training stage or after inference, we compare the results of conditioned training and re-ranking after inference. For conditioned training, in the template-to-melody module of TeleMelody, we add the additional pitch information to the input templates. That is, we extract the pitch directions between adjacent notes and add them as a part of the input musical templates to test if they can help with the tone relationships between lyrics and melodies. According to the subjective results in Setting #3 of Table 5, though the intelligibility is higher, other subjective scores decrease, indicating that adding conditions in the training process provides constraints too strong

that affect other aspects of the model performance. For re-ranking, during the inference, we save several candidates as well as their scores from the model and then re-rank them with the combined scores (the score from the model and the score given by principles of lyric-melody relationships). Then we choose the melody with the highest combined score as the final result. As can be seen in Setting #4 of Table 5, this strategy improves the perceptive quality of generated melodies compared with original TeleMelody, however, performs worse than ReLyMe. The results mean that the original lyric-to-melody system can not well capture the lyric-melody relationships, so even the best one of the candidates is not good enough.

### _4.3.4 Case Study._

We present the music sheets of samples generated by different methods with the same input lyrics for comparison in Figure 7. It is obvious that ReLyMe can alleviate most of the tone and rhythm dissonance (red blocks), ensure the harmony between lyrics and melodies (blue blocks), and generate melodies that have the repetition patterns matched with the lyrics (yellow lines). Meanwhile, unlike “Hard Constraints”, ReLyMe retains the flexibility of generated melodies. More cases and a demo video can be found in https://ai-muzic.github.io/relyme/.

# 5 CONCLUSION

In this paper, we propose ReLyMe, a method that integrates lyricmelody relationships into neural network models to ensure the harmony between lyrics and melodies. Inspired by musicians, we formulate the principles of lyric-melody relationships from three aspects (tone, rhythm, and structure). ReLyMe adjusts the probability of the predicted token at each decoding step according to lyric-melody relationships and can be applied to different lyric-tomelody generation systems without training process. Both objective and subjective results show the effectiveness of ReLyMe in improving lyric-to-melody generation by incorporating the relationships between lyrics and melodies. Method analyses further demonstrate the advantage of detailed designs in ReLyMe. In the future, we will conduct experiments on more languages and plan to apply the idea of ReLyMe to other automatic songwriting tasks, such as melody-to-lyric generation.

## ACKNOWLEDGEMENTS

This work was supported by the Key Project of Natural Science Foundation of Zhejiang Province (No.LZ19F020002), the Key R&D Program of Zhejiang Province (No.2022C03126) and Project of Key Laboratory of Intelligent Processing Technology for Digital Music (Zhejiang Conservatory of Music), Ministry of Culture and Tourism (No.2022DMKLB001). Thanks are due to Shen Zhou for bringing insights in music.

# REFERENCES

- [1] Hangbo Bao, Shaohan Huang, Furu Wei, Lei Cui, Yu Wu, Chuanqi Tan, Songhao Piao, and Ming Zhou. 2019. Neural melody composition from lyrics. In _CCF International Conference on Natural Language Processing and Chinese Computing_ . Springer, 499–511.

- [2] B Benward and M Saker. 2003. Introduction. The materials of music: Sound and time. _Music in theory and practice, pp. xii–25). NY, NY: McGraw-Hill_ (2003).

> [3] David Ronald Bishop. 2012. _Perfect shark music: How can the principle of prosody be used in contemporary song-writing?_ Ph. D. Dissertation. Wintec.

ReLyMe: Improving Lyric-to-Melody Generation by Incorporating Lyric-Melody Relationships

Lisbon ’22, October 10–14, 2022, Lisbon, Portugal

|Setting|Model||Melody|||Melody + L|yrics||Average|
|---|---|---|---|---|---|---|---|---|---|
|||singability|diversity|listenability|intelligibility|coherence|structure|overall||
|#1|TeleMelody [15]|2.79|**3.29**|2.92|2.94|2.51|2.73|2.78|2.85 (±0_._07)|
|#2|+ hard constraints|2.97|3.03|2.72|3.37|2.53|2.79|2.77|2.88 (±0_._06)|
|#3|+ conditioned training|2.96|2.92|2.79|3.08|2.67|2.72|2.75|2.84 (±0_._07)|
|#4|+ re-ranking|2.89|3.08|2.74|3.32|2.59|2.79|2.81|2.90 (±0_._06)|
|#5|ReLyMe|**3.21**|3.20|**3.16**|**3.39**|**2.94**|**3.16**|**3.11**|**3.16 (**±0_._06**)**|


**Table 5: The results of subjective evaluation for different settings of method analyses. The average scores are calculated with 95% confidence intervals.**

- [4] Zihang Dai, Zhilin Yang, Yiming Yang, Jaime G Carbonell, Quoc Le, and Ruslan Salakhutdinov. 2019. Transformer-XL: Attentive Language Models beyond a Fixed-Length Context. In _Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics_ . 2978–2988.

- [5] Lyn Frazier, Katy Carlson, and Charles Clifton Jr. 2006. Prosodic phrasing is central to language comprehension. _Trends in cognitive sciences_ 10, 6 (2006), 244–249.

- [6] Satoru Fukayama, Kei Nakatsuma, Shinji Sako, Takuya Nishimoto, and Shigeki Sagayama. 2010. Automatic song composition from the lyrics exploiting prosody of the Japanese language. In _Proc. 7th Sound and Music Computing Conference (SMC)_ . 299–302.

- [7] Satoru Fukayama, Daisuke Saito, and Shigeki Sagayama. 2012. Assistance for novice users on creating songs from japanese lyrics. In _ICMC_ .

- [8] Erica Greene, Tugba Bodrumlu, and Kevin Knight. 2010. Automatic analysis of rhythmic poetry with applications to generation and translation. In _Proceedings of the 2010 conference on empirical methods in natural language processing_ . 524–533.

- [9] Fenfei Guo, Chen Zhang, Zhirui Zhang, Qixin He, Kejun Zhang, Jun Xie, and Jordan Boyd-Graber. 2022. Automatic Song Translation for Tonal Languages. (2022).

- [10] Chris Hokamp and Qun Liu. 2017. Lexically Constrained Decoding for Sequence Generation Using Grid Beam Search. In _Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ . 1535–1546.

- [11] Fangzhou Hu. 2017. _Lexical tones in Mandarin sung words: A Phonetic and Psycholiguistic investigation_ . Master’s thesis. Shanghai International Studies University.

- [12] J Edward Hu, Huda Khayrallah, Ryan Culkin, Patrick Xia, Tongfei Chen, Matt Post, and Benjamin Van Durme. 2019. Improved lexically constrained decoding for translation and monolingual rewriting. In _Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)_ . 839–850.

- [13] Yu-Siang Huang and Yi-Hsuan Yang. 2020. Pop music transformer: Beat-based modeling and generation of expressive pop piano compositions. In _Proceedings of the 28th ACM International Conference on Multimedia_ . 1180–1188.

- [14] Harsh Jhamtani and Taylor Berg-Kirkpatrick. 2019. Modeling self-repetition in music generation using generative adversarial networks. In _Machine Learning for Music Discovery Workshop, ICML_ .

- [15] Zeqian Ju, Peiling Lu, Xu Tan, Rui Wang, Chen Zhang, Songruoyao Wu, Kejun Zhang, Xiangyang Li, Tao Qin, and Tie-Yan Liu. 2021. TeleMelody: Lyric-toMelody Generation with a Template-Based Two-Stage Method. _arXiv preprint arXiv:2109.09617_ (2021).

- [16] Surafel Melaku Lakew, Mattia Antonino Di Gangi, and Marcello Federico. 2019. Controlling the Output Length of Neural Machine Translation. In _16th International Workshop on Spoken Language Translation_ .

   - [23] Eric Nichols, Dan Morris, Sumit Basu, and Christopher Raphael. 2009. Relationships between lyrics and melody in popular music. In _ISMIR 2009-Proceedings of the 11th International Society for Music Information Retrieval Conference_ . 471–476.

   - [24] ITUT Rec. 1994. P. 85. a method for subjective performance assessment of the quality of speech voice output devices. _International Telecommunication Union, Geneva_ (1994).

   - [25] Ashutosh Saboo and Timo Baumann. 2019. Integration of dubbing constraints into machine translation. In _Proceedings of the Fourth Conference on Machine Translation (Volume 1: Research Papers)_ . 94–101.

   - [26] Murray Henry Schellenberg. 2013. _The realization of tone in singing in Cantonese and Mandarin_ . Ph. D. Dissertation. University of British Columbia.

   - [27] Jingmin Shao. 2007. _General Theory of Modern Chinese_ . Shanghai Education Publishing House.

   - [28] Zhonghao Sheng, Kaitao Song, Xu Tan, Yi Ren, Wei Ye, Shikun Zhang, and Tao Qin. 2021. SongMASS: Automatic Song Writing with Pre-training and Alignment Constraint. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , Vol. 35. 13798–13805.

   - [29] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. 2015. Deep unsupervised learning using nonequilibrium thermodynamics. In _International Conference on Machine Learning_ . PMLR, 2256–2265.

   - [30] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. _Advances in neural information processing systems_ 30 (2017).

   - [31] Susanna Wang, Doyon Ding, and Higashi Taku. 2015. Tone pitch realization under different intonation conditions. _Journal of Acoustics_ 40, 6 (2015), 902–913.

   - [32] Moira Yip. 2002. _Tone_ . Cambridge University Press.

   - [33] Huiyong Yu. 2008. _A study of cadential word relations_ . Central Conservatory of Music Press.

   - [34] Yi Yu, Abhishek Srivastava, and Simon Canales. 2021. Conditional lstm-gan for melody generation from lyrics. _ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM)_ 17, 1 (2021), 1–20.

   - [35] Chen Zhang, Jiaxing Yu, LuChin Chang, Xu Tan, Jiawei Chen, Tao Qin, and Kejun Zhang. 2021. PDAugment: Data Augmentation by Pitch and Duration Adjustments for Automatic Lyrics Transcription. _arXiv preprint arXiv:2109.07940_ (2021).

   - [36] Xueyao Zhang, Jinchao Zhang, Yao Qiu, Li Wang, and Jie Zhou. 2021. StructureEnhanced Pop Music Generation via Harmony-Aware Learning. _arXiv preprint arXiv:2109.06441_ (2021).

   - [37] Xu Zou, Da Yin, Qingyang Zhong, Hongxia Yang, Zhilin Yang, and Jie Tang. 2021. Controllable generation from pre-trained language models via inverse prompting. In _Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining_ . 2450–2460.

- [17] Hsin-Pei Lee, Jhih-Sheng Fang, and Wei-Yun Ma. 2019. iComposer: An automatic songwriting system for Chinese popular music. In _Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics (Demonstrations)_ . 84–88.

- [18] Fred Lerdahl and Ray Jackendoff. 1983. A Generative Theory pf Tonal Music.

- [19] Piji Li, Haisong Zhang, Xiaojiang Liu, and Shuming Shi. 2020. Rigid Formats Controlled Text Generation. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_ . 742–751.

- [20] Cheng Long, Raymond Chi-Wing Wong, and Raymond Ka Wai Sze. 2013. T- Music: A melody composer based on frequent pattern mining. In _2013 IEEE 29th International Conference on Data Engineering (ICDE)_ . IEEE, 1332–1335.

- [21] Xichu Ma, Ye Wang, Min-Yen Kan, and Wee Sun Lee. 2021. AI-Lyricist: Generating Music and Vocabulary Constrained Lyrics. In _Proceedings of the 29th ACM International Conference on Multimedia_ . 1002–1011.

- [22] Kristine Monteith, Tony R Martinez, and Dan Ventura. 2012. Automatic Generation of Melodic Accompaniments for Lyrics.. In _ICCC_ . 87–94.

Lisbon ’22, October 10–14, 2022, Lisbon, Portugal

Chen Zhang, Luchin Chang, Songruoyao Wu, Xu Tan, Tao Qin, Tie-Yan Liu, and Kejun Zhang

# APPENDIX

# Appendix A. Details of Objective Metrics

# Tone

- Transition: For each adjacent tone pair, we give a score based on the harmony degree of the pitch difference of the corresponding notes. We set the score as 1, 0.5, 0.2, 0 for excellent, good, fair, bad respectively. The transition score is the total score of all adjacent tone pairs divided by the number of all adjacent tone pairs.

- What is the degree of harmony between the repetitiveness of the melody and the structure of the lyrics?

# Overall

   - Please give an overall score to this singing clip.

- Contour: We count the total number of cases in which the pitch contours of melodies match the corresponding lyrics. The contour score is the matched ratio, i.e. the number of matched cases divided by the sentence number.

# Rhythm

- Matched s/w positions: We count the total number of matched cases, including the first letter or syllable of the keyword is on a downbeat and the first letter or syllable of the auxiliary word is on a weak beat. Then the number is divided by the total number of auxiliary words and keywords as the matched ratio.

- Matched pauses: We count the total number of the cases that do not match, i.e. the REST note or long note is between two syllables within the same word. Then the unmatched number is divided by the total number of inner-word syllables (which means not the first syllable of a word) as an unmatched ratio. The final matched ratio is one minus unmatched ratio.

# Structure

Inspired by the evaluation method of SongMASS [28], for the structure objective evaluation, we calculate the pitch and duration distribution similarity and melody distance between the melodies with the same structure.

# Appendix B. Instruction of Subjective Evaluation Singability

- How smooth is the melody? (bonus)

- Is the melody easy to sing? (bonus)

- Are there dissonant pitches or intervals? (penalty)

- Are there odd or unexpected rhythms? (penalty)

# Diversity

- Is the melody creative? Is the melody not easy to predict? (bonus)

- Is the melody monotonous or boring? (penalty)

# Listenability

- How good is the overall listenability of the melody?

# Intelligibility

- How easy could you understand the lyrics only depends on the singing clip?

# Coherence

- Are the melody and the lyric matched in terms of the emotion and the meanings?

- Are the prosodies of the melody and the lyric matched?

# Structure

- Are there reoccurring themes or motives in the music clip?
