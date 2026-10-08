---
title: "Towards Computational Analysis of Pansori Singing"
authors: "Sangheon Park; Danbinaerin Han; Dasaem Jeong"
venue: "25th International Society for Music Information Retrieval Conference (ISMIR), Late-Breaking Demo"
year: 2024
bib_key: null
source_url: "https://arxiv.org/pdf/2410.12956"
version: "arXiv v1 (ISMIR 2024 Late-Breaking Demo)"
cited_in_paper: false
relevance: "Direct pansori MIR precedent analyzing pitch, rhythm, modes, and transcription/audio mismatch."
---
## Why it matters

Direct pansori MIR precedent analyzing pitch, rhythm, modes, and transcription/audio mismatch.


# TOWARDS COMPUTATIONAL ANALYSIS OF PANSORI SINGING

# Sangheon Park

# Danbinaerin Han

**Dasaem Jeong** Sogang University

Georgia Institute of Technology KAIST Sogang University sangheon@gatech.edu naerin71@kaist.ac.kr dasaemj@sogang.ac.kr

## ABSTRACT

_Pansori_ is one of the most representative vocal genres of Korean traditional music, which has an elaborated vocal melody line with strong vibrato. Although the music is transmitted orally without any music notation, transcribing pansori music in Western staff notation has been introduced for several purposes, such as documentation of music, education, or research.

In this paper, we introduce computational analysis of pansori based on both audio and corresponding transcription, how modern Music Information Retrieval tasks can be used in analyzing traditional music and how it revealed different audio characteristics of what pansori contains.

# 1. INTRODUCTION

_Pansori_ , a traditional Korean monologue musical storytelling performed by a solo singer with drum accompaniment, is renowned for its emotional depth and diverse musical expressions [1]. The music was transmitted orally since the late 17th century without musical notation, and each singer made their own improvisation and variation, which enriched the repertoire and expression of pansori.

Musicological research on pansori often relies on manual transcription of music in Western staff notation [2–4]. While using Western notation for transcribing pansori is necessary for analyzing scales and rhythmic patterns, allowing for quantification and comparison, it has clear limitations in capturing the flexible melodies and improvisational nature of the performance.

To address the limitations of relying solely on Western notation, we incorporate a computational approach by aligning the audio recordings of pansori with their corresponding transcriptions. This approach allows us to directly examine the nuances of vocal performance, including variations in pitch, timing, and ornamentation that are difficult to capture through notation alone. By analyzing the audio recordings of a renowned pansori musician alongside her own transcriptions, we gain fuller insights into the performance practices and interpretive decisions that shape pansori’s rich and expressive musical landscape.

> © S. Park, D. Han, and D. Jeong. Licensed under a Creative Commons Attribution 4.0 International License (CC BY 4.0). **Attribution:** S. Park, D. Han, and D. Jeong, “Towards Computational Analysis of Pansori Singing”, in _Extended Abstracts for the Late-Breaking Demo Session of the 25th Int. Society for Music Information Retrieval Conf.,_ San Francisco, United States, 2024.


F0<br>WAV<br>On beat<br>MusicXML<br>

**Figure 1** : Overview of data visualization pipeline from the pansori singer to data.

# 2. DATASET

Among five classic pieces of pansori, we focused on _Jeokbyeokga_ ( `적벽가` , The Song of the Red Cliff), which is based on the famous Battle of Red Cliffs from Romance of the Three Kingdoms ( `三國演義` ). We used a recording sung by Chae Soojung ( `채수정` ), which was released in 2024 [5]. She also published a transcription of entire _Jeokbyeokga_ sung by her mentor, Park Songhee ( `박송희` ), in Western staff notation. As Chae strictly followed her mentor’s version in most of the parts [5], this transcription and her own recording make a pair that is valuable to analyze.

Among 30 different _daemok_ of Jeokbyeokga, we selected 8 of them using _Joongmori jangdan_ . _Daemok_ is a section of the song in pansori. _Jangdan_ is a metrical concept in Korean traditional music, which includes both tempo and rhythmic pattern and _Joongmori_ is one type of jangdan in mid-fast tempo and typically notated in 12/4 meter in western notation [6]. The total length of eight daemok was 166 measures in 12/4 meter.

# 3. METHOD

To extract the fundamental frequency (F0) contour, which captures the most essential melodic information from each audio recording, we employed the CREPE algorithm [7]. To reduce the noise of the extracted F0, we filtered out F0 values with confidence scores below 0.6. Additionally, to avoid including pitch values outside the typical range of a singer’s voice, we excluded frequencies below 350 Hz and above 1000 Hz.

For beat detection, we utilized the madmom library [8], specifically its recurrent neural network beat detection tool. We then manually annotated the beats to ensure that each measure contained exactly 12 beats, maintaining consistency throughout the analysis. Although all daemok were sung in the same joongmori jangdan, in instances


**Figure 2** : Pitch histograms of two daemok, No. 3 and No. 10, which have different modes, Ujo and Gyemyeonjo. Histogram shows F0 contour counts, and note durations from MusicXML transcriptions.

where the singer performed a measure with a dynamic tempo and rhythmic complexity, the beat detection did not function as accurately, occasionally resulting in more than 12 beats per measure and inconsistency. For example, in Daemok 11-3 —a soldiers’ sorrowful longing for his wife—, there is a section where the singer performs with an off beat tempo with syncopation. In this case, we had to adjust most of the beat notation manually. However, in daemok with rhythmic consistency and even interval between beats, the madmom beat detection tool generally demonstrated stable performance.

## 3.1 Pitch Histogram

To analyze and verify the characteristics of the vocal performance, we generated two types of histograms: one representing the distribution of F0 values and the other capturing note duration extracted from the MusicXML file.


(a) (b)<br>(c) (d)<br>

**Figure 3** : F0 contours of selected patterns

mode dominates throughout the piece.

A pitch histogram serves as an effective tool for visualizing the note distribution in the daemok. By comparing the pitch occurrences with the basic Ujo (D-F-G-A-C) and key notes in Gyemyeonjo(D-FE-(G)-A-C), we can observe dominant mode in each section. Figure 2 presents the example from Daemok 3, which is about the episode of _The Three Visits to the Thatched Cottage_ , and Daemok 10, which describes soldiers playing and crying. In Gyemyeonjo, F-E is frequently used to depict crying sounds or sad emotions. The histogram shows a greater use of Gyemyeonjo with frequent use of E note in Daemok 10, both in F0 and the transcription.

## 3.2 Pattern Detection

To analyze note patterns that frequently appear throughout different daemok, we converted all the notes from the MusicXML file into a word-like encoding. Each note was converted as a single text word that contains note pitch value and note duration. Then, we grouped these patterns using n-gram algorithm, especially 2, 3, 4, and 6 grams. From all sorted n-gram patterns that we have, we exported the corresponding F0 value of each pattern and plotted them by pattern so that we can compare the F0 value of the pattern in different parts of the daemok or even in different daemok.

# 4. RESULTS

## 4.1 Pitch Histogram

Analyzing the mode of each daemok in Pansori is key in understanding the structure and musical language of genre. _Jeokbyeokga_ , the piece that we used for data employs two modes: _Gyemyeonjo_ , and _Ujo_ . A significant characteristic of Pansori is the modulation between modes within a single piece. By examining the pitch content and ornamentation used in different sections of the daemok, we can identify the dominant mode in each part and can evaluate which

## 4.2 Pattern Analysis Based n-gram

One of the major characteristics of pansori is its dynamic vibrato, which plays a significant role in conveying emotion and enhancing the musicality of the performance. As illustrated in Figure 3 (a), within the common rhythmic pattern of joongmori, vibrato across different daemok exhibit similarities in both depth and rate.

Using the n-gram pattern, we could find an example where idiomatic patterns appear in different daemok, as presented in Figure 3 (b). The F0 contours also show a close overlap, which emphasizes that this pattern was sung almost identically in different daemok. Figure 3 (c) and (d) shows that the singer sometimes used an ascending portamento when singing a long note after a rest.

# 5. FUTURE WORK

Our future work aims to expand the analysis to include more singers and diverse musical samples. By analyzing the musical styles and patterns transmitted through various schools and traditions, we seek to gain a deeper understanding of the uniqueness of pansori that distinguishes it from other musical forms.

# 6. ACKNOWLEDGEMENT

This work was supported by the Ministry of Education of the Republic of Korea and the National Research Foundation of Korea (NRF-2024S1A5C3A03046168).

# 7. REFERENCES

- [1] T. N. C. for Korean Traditional Performing Arts, “Pansori,” https://www.gugak.go.kr/site/program/ board/basicboard/view?currentpage=2&menuid= 001003002005&searchselect=&searchword= &pagesize=10&boardtypeid=24&boardid=13149& lang=en, 2004, registration No. Y000780, Sort Code 679.4, Demand Code 679.4/N278p.

- [2] E. J. Shin, “A study of kang dokeun’s chunhyangga: From the introduction to jeokseongga ( `강도근 춘향가 연구` : `초앞부터` < `적성가` > `대목을 중심으로` ),” _Journal of Pansori_ , vol. 53, pp. 217–263, 2022.

- [3] H. J. Kim, “Concentration and texture of emotions in the musical expression of sim bongsa’s crying ( `보성소 리 심청가 중 심봉사 탄식 대목에 나타난 음악적 표 현의 농도와 질감` ),” _Journal of Pansori_ , vol. 57, pp. 39–74, 2024.

- [4] S. E. Wang, “Vocal characteristics in “sam-go-chory˘o” from “j˘og-by˘og-ga” sung by gim chang-ryong ( `김 창룡` < `적벽가` > ‘ `삼고초려` ’ `대목에 나타나는 창법적 특징 연구` ),” _Journal of Pansori_ , vol. 57, p. 109 146, 2024.

- [5] S. J. Chae, _Examine Jeokbyeokga with Soojeong Chae (_ _`채수정의 적벽가 톺아보기` )_ . Publishing Company Minsokwon ( `민속원` ), 2024.

- [6] S. M. Min, _A Great Korean Music Pioneer Min-Chong Park: A Performance Guide of His Selected Violin Works_ . Louisiana State University and Agricultural & Mechanical College, 2014.

- [7] J. W. Kim, J. Salamon, P. Li, and J. P. Bello, “Crepe: A convolutional representation for pitch estimation,” in _2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ . IEEE, 2018, pp. 161–165.

- [8] S. Böck, F. Korzeniowski, J. Schlüter, F. Krebs, and G. Widmer, “Madmom: A new python audio and music signal processing library,” in _Proceedings of the 24th ACM international conference on Multimedia_ , 2016, pp. 1174–1178.
