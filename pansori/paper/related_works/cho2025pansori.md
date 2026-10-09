---
title: "Exploring Pansori Generation with ACE-Step"
authors: "Seola Cho; Minjun Kim; Dasaem Jeong"
venue: "International Society for Music Information Retrieval Conference (ISMIR), Late-Breaking/Demo"
year: 2025
bib_key: "cho2025pansori"
source_url: "https://jarammm.github.io/pansorigen/assets/manuscript/463_lbd.pdf"
version: "LBD extended abstract (2 pages)"
cited_in_paper: true
relevance: "Released pansori LoRA for ACE-Step (7 h of one performer); used at its best reported setting (23k steps, LoRA weight 2) as the fine-tuned KAD anchor."
---

## Why it matters

Released pansori LoRA for ACE-Step (7 h of one performer); used at its best reported setting (23k steps, LoRA weight 2) as the fine-tuned KAD anchor.

> Text extracted with pdftotext on 2026-10-09; layout and equations are not cleaned.

                 EXPLORING PANSORI GENERATION WITH ACE-STEP

                              Seola Cho1 , Minjun Kim2 , Dasaem Jeong1
                                   1
                           Department of Art & Technology, Sogang University
                 2
                   Department of Electrical and Electronic Engineering, Yonsei University
             {seola.cho, dasaemj}@sogang.ac.kr, oxymoron@yonsei.ac.kr

                           ABSTRACT

Pansori is a traditional Korean narrative musical form that
combines sung passages with spoken narration (aniri) un-
der explicit rhythmic cycles (jangdan). While recent text-
to-music models achieve strong results on mainstream gen-

                                                                                                                                            Late-Breaking / Demo Session Extended Abstract, ISMIR 2025 Conference
res, their behavior on underrepresented traditions remains
largely unexplored. We adapt ACE-Step to pansori via
LoRA-based fine-tuning and compare its adaptation to Ko-
rean pop as a mainstream baseline. We evaluate our results
through quantitative analysis with Fréchet Audio Distance
(FAD) and qualitative assessment by an expert pansori per-
former. Our study highlights challenges and partial suc-
cesses in adapting to pansori. Future work will extend text-
to-music models toward cultural diversity and traditional
music.

                     1. INTRODUCTION
                                                                                         Figure 1. ACE-Step Overview
Recent advances in text-to-music models [1–3] from com-
mercial services demonstrated that these models are capa-
ble of generating plausible music with proper styles in au-                friendly performances. Particularly noteworthy are the
dio, including vocal with lyric condition. Following these                 sorikkun Jaram Lee’s adaptations of Western literature
services, open source models that can generate whole song                  (e.g., Ernest Hemingway’s The Old Man and the Sea, Vic-
audio have been introduced [4–7]. While these models                       tor Hugo’s Les Misérables). In such cases, there is signifi-
show strong performance in generating popular music gen-                   cant potential for AI generation to assist in creating pansori
res, their usability for generating traditional music of vari-             music for new, non-traditional texts, contributing to the de-
ous countries is still under-explored [8, 9].                              velopment and coexistence of traditional music.
    In this work, we investigate the adaptation of ACE-Step                    Given the complex melodic, rhythmic, and vocal het-
[7] to pansori. Pansori is a traditional Korean narrative                  erogeneity intrinsic to pansori, including its oral tradition
musical form performed by a solo singer (sorikkun) [10]. It                and lack of comprehensive symbolic notation, audio-based
combines singing (sori) with spoken narration (aniri) un-                  models are particularly suitable for pansori generation.
der explicit rhythmic cycles (jangdan). The art form has                   Unlike symbolic music representation, audio models can
its origins in shamanistic music of the Namdo region and                   grasp the subtle vocal timbres, ornamentation, and the in-
evolved through the creativity of many master performers                   terplay between sung and spoken parts that define pansori’s
who contributed melodic variations known as deoneum.                       authenticity.
The transmission of pansori remains primarily oral, em-                        This study explores the capabilities and limitations of
phasizing the individuality of singers who interpret and in-               adapting ACE-Step to pansori, and discusses their future
novate within this living tradition.                                       directions for text-to-music models applied to traditional
    Pansori is actively being revitalized today through cre-               genres across cultures.
ative reinterpretations, new compositions, and audience-

                                                                                                  2. METHOD
              © S. Cho, M. Kim, and D. Jeong. Licensed under a Creative
Commons Attribution 4.0 International License (CC BY 4.0). Attribu-        2.1 Dataset
tion: S. Cho, M. Kim, and D. Jeong, “Exploring Pansori Generation with
ACE-Step”, in Extended Abstracts for the Late-Breaking Demo Session        We used 7 hours of pansori audio performed by Jueun
of the 26th Int. Society for Music Information Retrieval Conf., Daejeon,   Lee, a lead artist of Folk Music Group of National Gu-
South Korea, 2025.                                                         gak Center, with permission from the performer. In this
work, we define segment markers as symbolic labels that          3.2 Expert Feedback
include jangdan, rhythmic patterns performed on a drum
                                                                 To assess the quality of pansori generation, we consulted
when singing, and aniri, spoken narrative passages rather
                                                                 Jueun Lee, the original performer of the dataset and a pro-
than sung parts. These markers were enclosed in square
                                                                 fessional gugak musician. According to her evaluation, the
brackets in the lyrics to provide explicit cues for the model.
                                                                 model captured vocal timbre, expressive ornamentation,
Audio was segmented into clips along lyric lines. We en-
                                                                 and contextually appropriate segmentation when symbolic
sured that each segment marker was preserved and fol-
                                                                 markers like jangdan and aniri were provided.
lowed ACE-Step’s recommended maximum audio length
                                                                    Although jangdan may sound sparse compared to pop
of 4 minutes per clip. For comparison, we included 2 hours
                                                                 drum accompaniment, it follows a well-defined rhythmic
of Korean pop songs performed by Kwang-seok Kim.
                                                                 framework that the model did not successfully learn. How-
                                                                 ever, the model often failed to maintain jangdan cycles, re-
2.2 LoRA Fine-tuning
                                                                 sulting in unstable timing and unclear rhythmic phrasing.
The model was trained for 23,000 steps using LoRA [11]              The model also struggled with aniri—narrative seg-
with the default configuration (r = 256 and α = 32). As          ments—as it attempted to handle them within a singing-
illustrated in Figure 1, LoRA is applied to the lyric encoder    oriented generation framework. While the resulting aniri
and the main transformer.                                        passages were distinct from sung segments, they still felt
                                                                 awkward and unnatural, failing to reflect their theatrical
                                                                 and speech-like nature. We provide audio examples on our
                        3. RESULTS                               supplementary webpage 1 .
3.1 Analysis between Pansori and Korean pop
                                                                                    4. CONCLUSION
                             Training steps                      FAD results suggest that the further a genre is from the
     LoRA
     weights     5k       10k     15k      20k     23k           model’s pretraining distribution, the harder it is to steer
        1       0.929    0.771   0.821    0.812   0.643          generation through fine-tuning.
        2       0.601    0.481   0.430    0.544   0.423             According to expert evaluation, the model captured
        3       0.655    0.626   0.596    0.765   0.648          short-term characteristics of pansori—such as vocal tex-
                                                                 ture, ornamentation, and segment transitions—but failed
Table 1. FAD↓ scores for pansori across LoRA weights             to reproduce long-term characteristics of pansori-such as
and training steps.                                              melodic progression and rhythmic cycles. Aniri lacked
                                                                 the natural theatrical quality expected of spoken narra-
                                                                 tive. These issues reflect the model’s difficulty in handling
                             Training steps                      the mixed characteristics of pansori, which includes both
     LoRA                                                        singing and speech-like narration. These findings imply
     weights     5k       10k     15k      20k     23k
                                                                 that adaptation to traditional genres like pansori demands
        1       0.439    0.398   0.395    0.435   0.396
                                                                 more than just parameter tuning. Rhythm-aware modeling,
        2       0.484    0.598   0.543    0.571   0.595
                                                                 symbolic guidance, or architectural modifications may be
        3       1.170    1.303   1.257    1.378   1.463
                                                                 necessary for more faithful generation.
Table 2. FAD↓ scores for Korean pop across LoRA
weights and training steps.                                                    5. ACKNOWLEDGEMENT
                                                                 This work was supported by the National Research Foun-
    To quantitatively compare the model’s adaptation to
                                                                 dation of Korea(NRF) grant funded by the Korea govern-
pansori and Korean pop, we computed the Fréchet Audio
                                                                 ment(MSIT) (RS-2025-00560548)
Distance (FAD) [12] scores for each genre under different
LoRA weights(rows in Tables) and training steps(columns
in Tables). The results are summarized in Table 1 and Ta-                           6. REFERENCES
ble 2.                                                           [1] “Suno,” https://suno.com, accessed: 2025-08-29.
    For pansori, Table 1 shows that the lowest FAD is 0.423
at 23k steps with LoRA 2. LoRA 3 outperforms LoRA 1 up           [2] “Udio,” https://udio.com, accessed: 2025-08-29.
to 20k. For Korean pop, Table 2 indicates that lower FAD
values generally occur with LoRA 1 and early checkpoints,        [3] S. Forsgren and H. Martiros, “Riffusion - Stable
while LoRA 3 yields higher FAD across steps.                         diffusion for real-time music generation,” 2022.
    Overall, pansori benefits from sufficient LoRA weights           [Online]. Available: https://riffusion.com/about
and training steps, whereas Korean pop tends to perform
                                                                 [4] R. Yuan, H. Lin, S. Guo, G. Zhang, J. Pan, Y. Zang,
best with minimal adaptation. While Korean pop outper-
                                                                     H. Liu, Y. Liang, W. Ma, X. Du, X. Du, Z. Ye,
formed pansori in FAD scores at the lowest LoRA weight,
                                                                     T. Zheng, Y. Ma, M. Liu, Z. Tian, Z. Zhou, L. Xue,
its performance degraded significantly when the LoRA
strength was increased.                                            1 https://jarammm.github.io/pansorigen/
     X. Qu, Y. Li, S. Wu, T. Shen, Z. Ma, J. Zhan,
     C. Wang, Y. Wang, X. Chi, X. Zhang, Z. Yang,
     X. Wang, S. Liu, L. Mei, P. Li, J. Wang, J. Yu,
     G. Pang, X. Li, Z. Wang, X. Zhou, L. Yu, E. Benetos,
     Y. Chen, C. Lin, X. Chen, G. Xia, Z. Zhang, C. Zhang,
     W. Chen, X. Zhou, X. Qiu, R. Dannenberg, J. Liu,
     J. Yang, W. Huang, W. Xue, X. Tan, and Y. Guo,
     “YuE: Scaling Open Foundation Models for Long-
     Form Music Generation,” 2025. [Online]. Available:
     https://arxiv.org/abs/2503.08638

 [5] Z. Liu, S. Ding, Z. Zhang, X. Dong, P. Zhang,
     Y. Zang, Y. Cao, D. Lin, and J. Wang, “SongGen:
     A Single Stage Auto-regressive Transformer for
     Text-to-Song Generation,” 2025. [Online]. Available:
     https://arxiv.org/abs/2502.13128

 [6] Z. Ning, H. Chen, Y. Jiang, C. Hao, G. Ma, S. Wang,
     J. Yao, and L. Xie, “DiffRhythm: Blazingly Fast and
     Embarrassingly Simple End-to-End Full-Length Song
     Generation with Latent Diffusion,” 2025. [Online].
     Available: https://arxiv.org/abs/2503.01183

 [7] J. Gong, S. Zhao, S. Wang, S. Xu, and J. Guo,
     “ACE-Step: A Step Towards Music Generation
     Foundation Model,” 2025. [Online]. Available: https:
     //arxiv.org/abs/2506.00045

 [8] E. Kanhov, A.-K. Kaila, and B. L. Sturm, “Innovation,
     data colonialism and ethics: critical reflections on the
     impacts of AI on Irish traditional music,” Journal of
     New Music Research, vol. 53, no. 1-2, pp. 47–63, 2024.

 [9] A. Mehta, S. Chauhan, A. Djanibekov, A. Kulka-
     rni, G. Xia, and M. Choudhury, “Music for All:
     Representational Bias and Cross-Cultural Adaptabil-
     ity of Music Generation Models,” arXiv preprint
     arXiv:2502.07328, 2025.

[10] Y.-S. Lee and C. ho Kim, Eds., Korean
     Musicology Series, 2: Pansori. Seoul: The
     National Center for Korean Traditional Per-
     forming Arts, 2008. [Online]. Available: https:
     //www.gugak.go.kr/site/program/board/basicboard/
     view?currentpage=2&menuid=001003002005&
     searchselect=&searchword=&pagesize=10&
     boardtypeid=24&boardid=1409&lang=ko

[11] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li,
     S. Wang, L. Wang, and W. Chen, “LoRA: Low-
     Rank Adaptation of Large Language Models,” 2021.
     [Online]. Available: https://arxiv.org/abs/2106.09685

[12] K. Kilgour, M. Zuluaga, D. Roblek, and M. Sharifi,
     “Fréchet Audio Distance: A Metric for Evaluating
     Music Enhancement Algorithms,” 2019. [Online].
     Available: https://arxiv.org/abs/1812.08466
