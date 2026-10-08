---
title: "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow"
authors: "Xingchao Liu; Chengyue Gong; Qiang Liu"
venue: "International Conference on Learning Representations (ICLR)"
year: 2023
bib_key: "liu2023flow"
source_url: "https://arxiv.org/pdf/2209.03003"
version: "arXiv v1 (published at ICLR 2023; OpenReview PDF endpoint returned HTTP 403)"
cited_in_paper: true
---
> Extraction note: 32 replacement glyph(s) remain, mainly in equations or figure text; the source PDF is required when an equation is ambiguous.


# Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow

Xingchao Liu<sup>*</sup>

University of Texas at Austin xcliu@utexas.edu

Chengyue Gong<sup>*</sup>

University of Texas at Austin cygong@cs.utexas.edu

Qiang Liu

University of Texas at Austin lqiang@cs.utexas.edu

## Abstract

We present rectified flow, a surprisingly simple approach to learning (neural) ordinary differential equation (ODE) models to transport between two empirically observed distributions _π_ 0 and _π_ 1, hence providing a unified solution to generative modeling and domain transfer, among various other tasks involving distribution transport. The idea of rectified flow is to learn the ODE to follow the straight paths connecting the points drawn from _π_ 0 and _π_ 1 as much as possible. This is achieved by solving a straightforward nonlinear least squares optimization problem, which can be easily scaled to large models without introducing extra parameters beyond standard supervised learning. The straight paths are special and preferred because they are the shortest paths between two points, and can be simulated exactly without time discretization and hence yield computationally efficient models. We show that the procedure of learning a rectified flow from data, called rectification, turns an arbitrary coupling of _π_ 0 and _π_ 1 to a new deterministic coupling with provably non-increasing convex transport costs. In addition, recursively applying rectification allows us to obtain a sequence of flows with increasingly straight paths, which can be simulated accurately with coarse time discretization in the inference phase. In empirical studies, we show that rectified flow performs superbly on image generation, image-to-image translation, and domain adaptation. In particular, on image generation and translation, our method yields nearly straight flows that give high quality results even with _a single Euler discretization step_ .

# 1 Introduction

Compared with supervised learning, the shared difficulty of various forms of unsupervised learning is the lack of _paired_ input/output data with which standard regression or classification tasks can be invoked. The gist of most unsupervised methods is to find, in one way or another, meaningful correspondences between points from two distributions. For example, generative models such as generative adversarial networks (GAN) and variational autoencoders (VAE) [e.g., 19, 32, 14] seek to map data points to latent codes following a simple elementary (Gaussian) distribution with which the data can be generated and manipulated. Representation learning rests on the idea that if a sufficiently smooth function can map a structured data

*XL and CG contributed equally to this work.


distribution to an elementary distribution, it can (likely) be endowed with certain semantically meaningful interpretation and useful for various downstream learning tasks. On the other hand, domain transfer methods find mappings to transfer points from two different data distributions, both observed empirically, for the purpose of image-to-image translation, style transfer, and domain adaption [e.g., 100, 16, 79, 59]. All these tasks can be framed unifiedly as finding a transport map between two distributions:

**The Transport Mapping Problem** _Given empirical observations of two distributions X_ 0 _∼ π_ 0 _, X_ 1 _∼ π_ 1 _on_ R<sup>_d_</sup> _, find a transport map T_ : R<sup>_d_</sup> _→_ R<sup>_d_</sup> _(hopefully nice or optimal in certain sense), such that Z_ 1 := _T_ ( _Z_ 0) _∼ π_ 1 _when Z_ 0 _∼ π_ 0 _, that is,_ ( _Z_ 0 _, Z_ 1) _is a coupling (a.k.a transport plan) of π_ 0 _and π_ 1 _._

Several lines of techniques have been developed depending on how to represent and train the map _T_ . In traditional generative models, _T_ is parameterized as a neural network, and trained with either GAN-type minimax algorithms or (approximate) maximum likelihood estimation (MLE). However, GANs are known to suffer from numerically instability and mode collapse issues, and require substantial engineering efforts and human tuning, which often do not transfer well across different model architecture and datasets. On the other hand, MLE tends to be intractable for complex models, and hence requires approximate variational or Monte Carlo inference techniques such as those used in variational auto-encoders (VAE), or special model structures such as normalizing flow and auto-regressive models, to yield tractable likelihood, causing difficult trade-offs between expressive power and computational cost.

Recently, advances have been made by representing the transport plan _implicitly as a continuous time process_ , such as flow models with neural ordinary differential equations (ODEs) [e.g., 6, 56] and diffusion models by stochastic differential equations (SDEs) [e.g., 73, 23, 80, 11, 82]; in these models, a neural network is trained to represent the drift force of the processes and a numerical ODE/SDE solver is used to simulate the process during inference. The key idea is that, by leveraging the mathematical structures of ODEs/SDEs, the continuous-time models can be trained efficiently without resorting to minimax or traditional approximate inference techniques. The most notable examples are the recent score-based generative models [71–73] and denoising diffusion probabilistic models (DDPM) [23], which we call denoising diffusion methods collectively. These methods allow us to train large-scale diffusion/SDE-based generative models that surpass GANs on image generation in both image quality and diversity, without the instability and mode collapse issues [e.g., 12, 53, 61, 64]. The learned SDEs can be converted into deterministic ODE models for faster inference with the method of probability flow ODEs [73] and DDIM [70].

However, compared with the traditional one-step models like GAN and VAE, a key drawback of continuoustimes models is the high computational cost in inference time: drawing a single point (e.g., image) requires to solve the ODE/SDE with a numerical solver that needs to repeatedly call the expensive neural drift function. In addition, the existing denoising diffusion techniques require substantial hyper-parameter search in an involved design space and are still poorly understood both empirically and theoretically [29].

In existing approaches, generative modeling and domain transfer are typically treated separately. It often requires to extend or customize a generative learning techniques to solve domain transfer problems; see e.g., Cycle GAN [100] and diffusion-based image-to-image translation [e.g., 75, 97]. One framework that naturally unifies both domains is optimal transport (OT) [e.g., 85, 2, 15, 59], which endows a collection of techniques for finding optimal couplings with minimum transport costs of form E[ _c_ ( _Z_ 1 _− Z_ 0)] w.r.t. a cost function _c_ : R<sup>_d_</sup> _→_ R, yielding natural applications to both generative and transfer learning. However, the existing OT techniques are slow for problems with high dimensional and large volumes of data [59]. Furthermore, as the transport costs do not perfectly align with the actual learning performance, methods that faithfully find the optimal transport maps do not necessarily have better learning performance [34].


Figure 1: The trajectories of rectified flows for image generation ( _π_ 0: standard Gaussian noise, _π_ 1: cat faces, top two rows), and image transfer between human and cat faces ( _π_ 0: human faces, _π_ 1: cat faces, bottom two rows), when simulated using Euler method with step size 1 _/N_ for _N_ steps. The first rectified flow induced from the training data ( _1-rectified flow_ ) yields good results with a very small number (e.g., _≥_ 2) of steps; the straightened reflow induced from _1-rectified flow_ (denoted as _2-rectified flow_ ) has nearly straight line trajectories and yield good results even with one discretization step.

# Contribution

We introduce _rectified flow_ , a surprisingly simple approach to the transport mapping problem, which unifiedly solves both generative modeling and domain transfer. The rectified flow is an ODE model that transport distribution _π_ 0 to _π_ 1 by _following straight line paths as much as possible._ The straight paths are preferred both theoretically because it is the shortest path between two end points, and computationally because it can be exactly simulated without time discretization. Hence, flows with straight paths bridge the gap between one-step and continuous-time models.

Algorithmically, the rectified flow is trained with a simple and scalable unconstrained least squares optimization procedure, which avoids the instability issues of GANs, the intractable likelihood of MLE methods, and the subtle hyper-parameter decisions of denoising diffusion models. The procedure of obtaining the rectified flow from the training data has the attractive theoretical property of 1) yielding a coupling with non-increasing transport cost jointly for all convex cost _c_ , and 2) making the paths of flow increasingly straight and hence incurring lower error with numerical solvers. Therefore, with a _reflow_ procedure that iteratively trains new rectified flows with the data simulated from the previously obtained rectified flow, we obtain nearly straight flows that yield good results even with the coarsest time discretization, i.e., one Euler step. Our method is purely ODE-based, and is both conceptually simpler and practically faster in inference time than the SDE-based approaches of [23, 73, 70].


(a) Linear interpolation (b) Rectified flow  Zt (c) Linear interpolation (d) Rectified flow  Zt ′<br>Xt =  tX 1 + (1  − t ) X 0 induced by ( X 0 , X 1) Zt =  tZ 1 + (1  − t ) Z 0 induced by ( Z 0 , Z 1)<br>

Figure 2: (a) Linear interpolation of data input ( _X_ 0 _, X_ 1) _∼ π_ 0 _× π_ 1. (b) The rectified flow _Zt_ induced by ( _X_ 0 _, X_ 1); the trajectories are “rewired” at the intersection points to avoid the crossing. (c) The linear interpolation of the end points ( _Z_ 0 _, Z_ 1) of flow _Zt_ . (d) The rectified flow induced from ( _Z_ 0 _, Z_ 1), which follows straight paths.

Empirically, rectified flow can yield high-quality results for image generation when simulated with a very few number of Euler steps (see Figure 1, top row). Moreover, with just one step of reflow, the flow becomes nearly straight and hence yield good results with a single Euler discretization step (Figure 1, the second row). This substantially improves over the standard denoising diffusion methods. Quantitatively, we claim a state-of-the-art result of FID (4.85) and recall (0.51) on CIFAR10 for one-step fast diffusion/flow models [5, 48, 91, 99, 47]. The same algorithm also achieves superb result on domain transfer tasks such as image-toimage translation (see the bottom two rows of Figure 1) and transfer learning.

# 2 Method

We provide a quick overview of the method in Section 2.1, followed with some discussion and remarks in Section 2.2. We introduce a nonlinear extension of our method in Section 2.3, with which we clarify the connection and advantages of our method with the method of probability flow ODEs [73] and DDIM [70].

## 2.1 Overview

**Rectified flow** Given empirical observations of _X_ 0 _∼ π_ 0 _, X_ 1 _∼ π_ 1, the rectified flow induced from ( _X_ 0 _, X_ 1) is an ordinary differentiable model (ODE) on time _t ∈_ [0 _,_ 1],


which converts _Z_ 0 from _π_ 0 to a _Z_ 1 following _π_ 1. The drift force _v_ : R<sup>_d_</sup> _→_ R<sup>_d_</sup> is set to drive the flow to follow the direction ( _X_ 1 _− X_ 0) of the linear path pointing from _X_ 0 to _X_ 1 as much as possible, by solving a simple least squares regression problem:


where _Xt_ is the linear interpolation of _X_ 0 and _X_ 1. Naviely, _Xt_ follows the ODE of d _Xt_ = ( _X_ 1 _− X_ 0)d _t,_ which is non-causal (or anticipating) as the update of _Xt_ requires the information of the final point _X_ 1. By fitting the drift _v_ with _X_ 1 _− X_ 0, the rectified flow _causalizes_ the paths of linear interpolation _Xt_ , yielding an ODE flow that can be simulated without seeing the future.

In practice, we parameterize _v_ with a neural network or other nonlinear models and solve (1) with any offthe-shelf stochastic optimizer, such as stochastic gradient descent, with empirical draws of ( _X_ 0 _, X_ 1). See


Algorithm 1. After we get _v_ , we solve the ODE starting from _Z_ 0 _∼ π_ 0 to transfer _π_ 0 to _π_ 1, backwardly starting from _Z_ 1 _∼ π_ 1 to transfer _π_ 1 to _π_ 0. Specifically, for backward sampling, we simply solve d _X_<sup>˜</sup> _t_ = _−v_ ( _X_<sup>˜</sup> _t, t_ )d _t_ initialized from _X_<sup>˜</sup> 0 _∼ π_ 1 and set _Xt_ = _X_ ˜1 _−t_ . The forward and backward sampling are equally favored by the training algorithm, because the objective in (1) is _time-symmetric_ in that it yields the equivalent problem if we exchange _X_ 0 and _X_ 1 and flip the sign of _v_ .

**Flows avoid crossing** A key to understanding the method is the non-crossing property of flows: the different paths following a well defined ODE d _Zt_ = _v_ ( _Zt, t_ )d _t_ , whose solution exists and is unique, _cannot cross each other_ at any time _t ∈_ [0 _,_ 1). Specifically, there exists no location _z ∈_ R<sup>_d_</sup> and time _t ∈_ [0 _,_ 1), such that two paths go across _z_ at time _t_ along different directions, because otherwise the solution of the ODE would be non-unique. On the other hand, the paths of the interpolation process _Xt_ may intersect with each other (Figure 2a), which makes it non-causal. Hence, as shown in Figure 2b, the rectified flow _rewires_ the individual trajectories passing through the intersection points to avoid crossing, while tracing out the same density map as the linear interpolation paths due to the optimization of (1). We can view the linear interpolation _Xt_ as building roads (or tunnels) to connect _π_ 0 and _π_ 1, and the rectified flow as traffics of particles passing through the roads in a myopic, memoryless, non-crossing way, which allows them to ignore the global path information of how _X_ 0 and _X_ 1 are paired, and rebuild a more deterministic pairing of ( _Z_ 0 _, Z_ 1).

**Rectified flows reduce transport costs** If (1) is solved exactly, the pair ( _Z_ 0 _, Z_ 1) of the rectified flow is guaranteed to be a valid coupling of _π_ 0 _, π_ 1 (Theorem 3.3), that is, _Z_ 1 follows _π_ 1 if _Z_ 0 _∼ π_ 0. Moreover, ( _Z_ 0 _, Z_ 1) guarantees to yield no larger transport cost than the data pair ( _X_ 0 _, X_ 1) simultaneously for _all_ convex cost functions _c_ (Theorem 3.5). The data pair ( _X_ 0 _, X_ 1) can be an arbitrary coupling of _π_ 0 _, π_ 1, typically independent (i.e., ( _X_ 0 _, X_ 1) _∼ π_ 0 _× π_ 1) as dictated by the lack of meaningfully paired observations in practical problems. In comparison, the rectified coupling ( _Z_ 0 _, Z_ 1) has a deterministic dependency as it is constructed from an ODE model. Denote by ( _Z_ 0 _, Z_ 1) = `Rectify` (( _X_ 0 _, X_ 1)) the mapping from ( _X_ 0 _, X_ 1) to ( _Z_ 0 _, Z_ 1). Hence, `Rectify` ( _·_ ) converts an arbitrary coupling into a deterministic coupling with lower convex transport costs.

**Straight line flows yield fast simulation** Following Algorithm 1, denote by **_Z_** = `RectFlow` (( _X_ 0 _, X_ 1)) the rectified flow induced from ( _X_ 0 _, X_ 1). Applying this operator recursively yields a sequence of rectified flows **_Z_**<sup>_k_+1</sup> = `RectFlow` (( _Z_ 0<sup>_k, Z_</sup> 1<sup>_k_))with(</sup><sup>_Z_</sup> 0<sup>0</sup><sup>_, Z_</sup> 1<sup>0)=(</sup><sup>_X_0</sup><sup>_, X_1),where</sup><sup>**_Z_**</sup><sup>_k_isthe</sup><sup>_k_-threctifiedflow,or</sup> simply _k-rectified flow_ , induced from ( _X_ 0 _, X_ 1).

This _reflow_ procedure not only decreases transport cost, but also has the important effect of straightening paths of rectified flows, that is, making the paths of the flow more straight. This is highly attractive computationally as flows with nearly straight paths incur small time-discretization error in numerical simulation. Indeed, perfectly straight paths can be simulated exactly with a single Euler step and is effectively a onestep model. This addresses the very bottleneck of high inference cost in existing continuous-time ODE/SDE models.

## 2.2 Main Results and Properties

We provide more in-depth discussions on the main properties of rectified flow. We keep the discussion informal to highlight the intuitions in this section and defer the full course theoretical analysis to Section 3.


**Algorithm 1** Rectified Flow: Main Algorithm

**Procedure** : **_Z_** = `RectFlow` (( _X_ 0 _, X_ 1)):

_Inputs_ : Draws from a coupling ( _X_ 0 _, X_ 1) of _π_ 0 and _π_ 1; velocity model _vθ_ : R<sup>_d_</sup> _→_ R<sup>_d_</sup> with parameter _θ. Training_ : _θ_<sup>ˆ</sup> = arg min E _,_ with _t ∼_ Uniform([0 _,_ 1]). _θ_ � _∥X_ 1 _− X_ 0 _− v_ ( _tX_ 1 + (1 _− t_ ) _X_ 0 _, t_ ) _∥_<sup>2�</sup> _Sampling_ : Draw ( _Z_ 0 _, Z_ 1) following d _Zt_ = _vθ_ ˆ( _Zt, t_ )d _t_ starting from _Z_ 0 _∼ π_ 0 (or backwardly _Z_ 1 _∼ π_ 1). _Return_ : **_Z_** = _{Zt_ : _t ∈_ [0 _,_ 1] _}_ .

**Reflow** (optional): **_Z_**<sup>_k_+1</sup> = `RectFlow` (( _Z_ 0<sup>_k, Z_</sup> 1<sup>_k_)), starting from (</sup><sup>_Z_</sup> 0<sup>0</sup><sup>_, Z_</sup> 1<sup>0) = (</sup><sup>_X_0</sup><sup>_, X_1).</sup> **Distill** (optional): Learn a neural network _T_<sup>ˆ</sup> to distill the _k_ -rectified flow, such that _Z_ 1<sup>_k≈T_ˆ(</sup><sup>_Z_</sup> 0<sup>_k_).</sup>

First, for a given input coupling ( _X_ 0 _, X_ 1), it is easy to see that the exact minimum of (1) is achieved if


which is the expectation of the line directions _X_ 1 _− X_ 0 that pass through _x_ at time _t_ . We discuss below the property of rectified flow d _Zt_ = _v_<sup>_X_</sup> ( _Zt, t_ )d _t_ with _Z_ 0 _∼ π_ 0, assuming that the ODE has an unique solution.

**Marginal preserving property** [Theorem 3.3] _The pair_ ( _Z_ 0 _, Z_ 1) _is a coupling of π_ 0 _and π_ 1 _. In fact, the marginal law of Zt equals that of Xt at every time t, that is,_ Law( _Zt_ ) = Law( _Xt_ ) _, ∀t ∈_ [0 _,_ 1] _._ Intuitively, this is because, by the definition of _v_<sup>_X_</sup> in (2), the expected amount of mass that passes through every infinitesmal volume at all location and time are equal under the dynamics of _Xt_ and _Zt_ , which ensures that they trace out the same marginal distributions:


On the other hand, the joint distributions of the whole trajectory of _Zt_ and that of _Xt_ are different in general. In particular, _Xt_ is in general a non-causal, non-Markov process, with ( _X_ 0 _, X_ 1) a stochastic coupling, and _Zt causalizes_ , _Markovianizes_ and _derandomizes Xt_ , while preserving the marginal distributions at all time.

**Reducing transport costs** [Theorem 3.5] _The coupling_ ( _Z_ 0 _, Z_ 1) _yields lower or equal convex transport costs than the input_ ( _X_ 0 _, X_ 1) _in that_ E[ _c_ ( _Z_ 1 _− Z_ 0)] _≤_ E[ _c_ ( _X_ 1 _− X_ 0)] _for any convex cost c_ : R<sup>_d_</sup> _→_ R _._ The transport costs measure the expense of transporting the mass of one distribution to another following the assignment relation specified by the coupling and is a central topic in optimal transport [e.g., 84, 85, 65, 59, 15]. Typical examples are _c_ ( _·_ ) = _∥·∥_<sup>_α_</sup> with _α ≥_ 1. Hence, `Rectify` ( _·_ ) yields a Pareto descent on the collection of all convex transport costs, without targeting any specific _c_ . This distinguishes it from the typical optimal transport optimization methods, which are explicitly framed to optimize a given _c_ . As a result, recursive application of `Rectify` ( _·_ ) does not guarantee to attain the _c_ -optimal coupling for any given _c_ , with the exception in the one-dimensional case when the fixed point of `Rectify` ( _·_ ) coincides with the unique monotonic coupling that simultaneously minimizes all non-negative convex costs _c_ ; see Section 3.4. Intuitively, the convex transport costs are guaranteed to decrease because the paths of the rectified flow _Zt_ is a rewiring of the straight paths connecting ( _X_ 0 _, X_ 1). To give an illustration, consider the simple case of


_c_ ( _·_ ) = _∥·∥_ when transport costs E[ _∥X_ 0 _− X_ 1 _∥_ ] and E[ _∥Z_ 0 _− Z_ 1 _∥_ ] are the expected length of the straight lines connecting the end points. The inequality can be proved graphically as follows:


( _∗_ ) ( _∗∗_ ) where _≤_ uses the triangle inequality, and = holds because the paths of _Zt_ is a rewiring of the straight paths of _Xt_ , following the construction of _v_<sup>**_X_**</sup> in (2). For general convex _c_ , a similar proof using Jensen’s inequality is shown in Section 3.2.

**Reflow, straightening, fast simulation** As shown in Figure 3, when we recursively apply the procedure **_Z_**<sup>_k_+1</sup> = `RectFlow` (( _Z_ 0<sup>_k, Z_</sup> 1<sup>_k_)),thepathsofthe</sup><sup>_k_-rectifiedflow</sup><sup>**_Z_**</sup><sup>_k_areincreasinglystraight,andhence</sup> easier to simulate numerically, as _k_ increases. This straightening tendency can be guaranteed theoretically.


(a)The 1st rectified flow  Z 1 (b) Reflow  Z 2 (c) Reflow  Z 3 (d) Transport cost,<br>Z 1 =  RectFlow (( X 0 , X 1)) Z 2 =  RectFlow (( Z 0 1 , Z 1 1)) Z 3 =  RectFlow (( Z 0 2 , Z 1 2)) Straightness<br>

Figure 3: (a)-(c) Samples of trajectories drawn from the reflows on a toy example ( _π_ 0: purple dots, _π_ 1: red dots; the green and blue lines are trajectories connecting different modes of _π_ 0 _, π_ 1). (d) The straightness and the relative L2 transport cost v.s. the reflow steps; the values are scaled into [0 _,_ 1], so 0 corresponds to straight lines and _L_ 2 optimal transport; see Section 5.1 for more information. We use the non-parametric model in (5) with bandwidth _h_ = 0 _._ 1.

Specifically, we say that a flow d _Zt_ = _v_ ( _Zt, t_ )d _t_ is straight if we have almost surely that _Zt_ = _tZ_ 1 + (1 _− t_ ) _Z_ 0 for _∀t ∈_ [0 _,_ 1], or equivalently _v_ ( _Zt, t_ ) = _Z_ 1 _− Z_ 0 = _const_ following each path. (More precisely, “straight” here refers to straight with a constant speed.) Such straight flows are highly attractive computationally as it is effective a one-step model: a single Euler step update _Z_ 1 = _Z_ 0 + _v_ ( _Z_ 0 _,_ 0) calculates the exact _Z_ 1 from _Z_ 0. Note that the linear interpolation **_X_** = _{Xt}_ is straight by this definition but it is not a (causal) flow and hence can not be simulated without an oracle assess to draws of both _π_ 0 and _π_ 1. In comparison, it is non-trivial to make a flow d _Zt_ = _v_ ( _Zt, t_ )d _t_ straight, because if so _v_ must satisfy the inviscid Burgers’ equation _∂tv_ + ( _∂zv_ ) _v_ = 0:


More generally, we can measure the straightness of any continuously differentiable process **_Z_** = _{Zt}_ by


_S_ ( **_Z_** ) = 0 means exact straightness. A flow whose _S_ ( **_Z_** ) is small has nearly straight paths and hence can be simulated accurately using numerical solvers with a small number of discretization steps. Section 3.3 shows that applying rectification recursively provably decreases _S_ ( **_Z_** ) towards zero.


_[Theorem 3.7] Let_ **_Z_**<sup>_k_</sup> _be the k-th rectified flow induced from_ ( _X_ 0 _, X_ 1) _. Then_


As shown Figure 1, applying one step of reflow can already provide nearly straight flows that yield good performance when simulated with a single Euler step. It is not recommended to apply too many reflow steps as it may accumulate estimation error on _v_<sup>_X_</sup> .

**Distillation** After obtaining the _k_ -th rectified flow **_Z_**<sup>_k_</sup> , we can further improve the inference speed by distilling the relation of ( _Z_ 0<sup>_k, Z_</sup> 1<sup>_k_) into a neural network</sup><sup>_T_ˆto directly predict</sup><sup>_Z_</sup> 1<sup>_k_from</sup><sup>_Z_</sup> 0<sup>_k_without simulating</sup> the flow. Given that the flow is already nearly straight (and hence well approximated by the one-step update), and the distillation can be done efficiently. In particular, if we take _T_<sup>ˆ</sup> ( _z_ 0) = _z_ 0 + _v_ ( _z_ 0 _,_ 0), then the loss function for distilling **_Z_**<sup>_k_</sup> is E 1<sup>_−Z_</sup> 0<sup>_k_)</sup><sup>_−v_(</sup><sup>_Z_</sup> 0<sup>_k,_0)</sup> , which is the term in (1) when _t_ = 0. ���( _Zk_ ��2<sup>�</sup>

We should highlight the difference between distillation and rectification: distillation attempts to faithfully approximate the coupling ( _Z_ 0<sup>_k, Z_</sup> 1<sup>_k_) while rectification yields a different coupling (</sup><sup>_Z_</sup> 0<sup>_k_+1</sup> _, Z_ 1<sup>_k_+1</sup> ) with lower transport cost and more straight flow. Hence, distillation should be applied only in the final stage when we want to fine-tune the model for fast one-step inference.

**On the velocity field** _v_<sup>_X_</sup> If _X_ 0 yields a conditional density function _ρ_ ( _x_ 0 _| x_ 1) when conditioned on _X_ 1 = _x_ 1 _,_ then the optimal velocity field _v_<sup>_X_</sup> ( _z, t_ ) = E[ _X_ 1 _− X_ 0 _|Xt_ = _z_ ] can be represented by


where the expectation E [ _·_ ] is taken w.r.t. _X_ 1 _∼ π_ 1. This can be seen by noting that _X_ 0 =<sup>_<u>z−</u>_</sup> 1 _−_<sup>_<u>tX</u>_</sup> _t_<sup><u>1</u></sup> and _X_ 1 _− X_ 0 =<sup>_<u>X</u>_</sup> 1<sup><u>1</u></sup> _−_<sup>_−_</sup> _t_<sup>_<u>z</u>_, when conditioned on</sup><sup>_Xt_=</sup><sup>_z_.Hence, if</sup><sup>_ρ_is positive and continuous everywhere, then</sup> _v_<sup>_X_</sup> is well defined and continuous on R<sup>_d_</sup> _×_ [0 _,_ 1). Further, if log _ηt_ is continuously differentiable w.r.t. _z_ , we can show that


Note that d _Zt_ = _v_<sup>_X_</sup> ( _Zt, t_ )d _t_ is guaranteed to have a unique solution if _v_<sup>_X_</sup> is uniformly Lipschitz continuous on [0 _, a_ ] for any _a <_ 1.

If _X_ 0 _|X_ 1 = _x_ 1 does not yield a conditional density function, _v_<sup>_X_</sup> ( _z, t_ ) may be undefined or discontinuous, making the ODE d _Zt_ = _v_<sup>_X_</sup> ( _Zt, t_ )d _t_ ill-behaved. A simple fix is to add _X_ 0 with a Gaussian noise _ξ ∼ N_ (0 _, σ_<sup>2</sup> _I_ ) independent of ( _X_ 0 _, X_ 1) to yield a smoothed variable _X_<sup>˜</sup> 0 = _X_ 0 + _ξ_ , and transfer _X_<sup>˜</sup> 0 to _X_ 1 using rectified flow. This would effectively give a randomized mapping of form _T_ ( _X_ 0 + _ξ_ ) transporting _π_ 0 to _π_ 1.

**Smooth function approximation** Following (4), we can _exactly_ calculate _v_<sup>_X_</sup> if the conditional density function _ρ_ ( _·|x_ 1) exists and is known, and _π_ 1 is the empirical measure of a finite number of points (whose expectation can be evaluated exactly). In this case, running the rectified flow forwardly would precisely recover the points in _π_ 1. This, however, is not practically useful in most cases as it completely overfits the


data. Hence, it is both necessary and beneficial to fit _v_<sup>_X_</sup> with a smooth function approximator such as neural network or non-parametric models, to obtain smoothed distributions with novel samples that are practically useful.

Deep neural networks are no doubt the best function approximators for large scale problems. For low dimensional problems, the following simple Nadaraya–Watson style non-parametric estimator of _v_<sup>_X_</sup> can yield a good approximation to the exact rectified flow without knowing the conditional density _ρ_ :


_κh_ <u>(</u> _Xt,z_ <u>)</u> where _ωh_ ( _Xt, z_ ) = E[ _κh_ ( _Xt,z_ )]<sup>,and</sup><sup>_κh_(</sup><sup>_x, z_) is a smoothing kernel with a bandwith parameter</sup><sup>_h>_0 that</sup> measures the similarity between _z_ and _x_ . Taking the Gaussian RBF kernel _κh_ ( _x, z_ ) = exp( _−∥x − z∥_<sup>2</sup> _/_ 2 _h_<sup>2</sup> ), then when _h →_ 0<sup>+</sup> , it can be shown that _v_<sup>_X,h_</sup> ( _z, t_ ) converges to _v_<sup>_X_</sup> ( _z, t_ ) = E _<u>X</u>_ 11 _−−tz_<sup>_| Xt_=</sup><sup>_z_</sup> on points � � _z_ that can be attained by _Xt_ (i.e., the conditional expectation E [ _· | Xt_ = _z_ ] exists. ). On points _z_ that _Xt_ can not attain, _v_<sup>_X,h_</sup> ( _z, t_ ) extrapolates the value by finding the _Xt_ that is close to _z_ . In practice, we replace the expectations in (5) with empirical averaging. We find that _v_<sup>_X,h_</sup> performs well in practice because it is a mixture of linear functions that always point to a point in the support of _π_ 1.

## 2.3 A Nonlinear Extension

We present a nonlinear extension of rectified flow in which the linear interpolation _Xt_ is replaced by any time-differentiable curve connecting _X_ 0 and _X_ 1. Such generalized rectified flows can still transport _π_ 0 to _π_ 1 (Theorem 3.3), but no longer guarantee to decrease convex transport costs, or have the straightening effect. Importantly, the method of probability flows [73] and DDIM [70] can be viewed (approximately) as special cases of this framework, allows us to clarify the connection with and the advantages over these methods.

Let **_X_** = _{Xt_ : _t ∈_ [0 _,_ 1] _}_ be any time-differentiable random process that connects _X_ 0 and _X_ 1. Let _X_<sup>˙</sup> _t_ be the time derivative of _Xt_ . The (nonlinear) rectified flow induced from **_X_** is defined as


We can estimate _v_<sup>**_X_**</sup> by solving


where _wt_ : (0 _,_ 1) _→_ (0 _,_ + _∞_ ) is a positive weighting sequence ( _wt_ = 1 by default). When using the linear interpolation _Xt_ = _tX_ 1 + (1 _− t_ ) _X_ 0, we have _X_<sup>˙</sup> _t_ = _X_ 1 _− X_ 0 and (6) with _wt_ = 1 reduces to (1). As we show in Theorem 3.3, the flow **_Z_** given by this method still preserves the marginal laws of **_X_** , that is, Law( _Zt_ ) = Law( _Xt_ ), _∀t ∈_ [0 _,_ 1], and hence ( _Z_ 0 _, Z_ 1) remains to be a coupling of _π_ 0 _, π_ 1. However, if **_X_** is not straight, ( _Z_ 0 _, Z_ 1) no longer guarantees to decrease the convex transport costs over ( _X_ 0 _, X_ 1). More importantly, the reflow procedure no longer straightens the paths of _Zt_ .

A simple class of interpolation processes is _Xt_ = _αtX_ 1 + _βtX_ 0 where _αt_ and _βt_ are two differentiable sequences that satisfy _α_ 1 = _β_ 0 = 1 and _α_ 0 = _β_ 1 = 0 to ensure that the process equals _X_ 0 _, X_ 1 at the starting and end points. In this case, we have _X_<sup>˙</sup> _t_ = _α_ ˙ _tX_ 1 + _β_<sup>˙</sup> _tX_ 0 in (6) where _α_ ˙ _t_ and _β_<sup>˙</sup> _t_ are the time derivatives of _αt_ and _βt_ . The shape of the curve is controlled by the relation of _αt_ and _βt_ . If we take


_βt_ = 1 _− αt_ for all _t_ , then _Xt_ have straight paths but does not travel at a constant speed; it can be viewed as a time-changed variant of the canonical case _Xt_ = _tX_ 1 + (1 _− t_ ) _X_ 0 when _t_ is reparameterized to _αt_ . When _βt̸_ = 1 _− αt_ , the paths of _Xt_ are not straight lines except some special cases (e.g., _α_ ˙ _tX_ 1 = 0 or _β_<sup>˙</sup> _tX_ 0 = 0, or _X_ 1 = _aX_ 1 for some _a ∈_ R).

## 2.3.1 Probability Flow ODEs and DDIM

The probability flow ODEs (PF-ODEs) [73] and denoising diffusion implicit models (DDIM) [70] are methods for learning ODE-based generative models of _π_ 1 from a spherical Gaussian initial distribution _π_ 0, derived by converting a SDE learned by denoising diffusion methods to an ODE with equivalent marginal laws. In [73], three types of PF-ODEs are derived from three types of SDEs learned as score-based generative models, including variance-exploding (VE) SDE, variance-preserving (VP) SDE, and sub-VP SDE, which we denote by VE ODE, VP ODE, and sub-VP ODE, respectively. VP ODE is equivalent to the continuous time limit of DDIM, which is derived from the denoising diffusion probability model (DDPM) [23]. As the derivations of PF-ODEs and DDIM require advanced tools in stochastic calculus, we limit our discussion on the final algorithmic procedures suggested in [73, 23], which we summarize in Section 3.5. The readers are referred to [73, 70] for the details.

_[Proposition 3.11] All variants of PF-ODEs can be viewed as instances of_ (6) _when using Xt_ = _αtX_ 1 + _βtξ for some αt, βt with α_ 1 = 1 _, β_ 1 = 0 _, where ξ ∼N_ (0 _, I_ ) _is a standard Gaussian random variable._

Here we need to use introduce _ξ_ to replace _X_ 0 because the choices of _αt_ and _βt_ suggested in [73, 70, 23] do not satisfy the boundary condition of _α_ 0 = 0 and _β_ 0 = 1 at _t_ = 0, and hence _X_ 0 _̸_ = _ξ_ . Instead, in these methods, the initial distribution _X_ 0 _∼ π_ 0 is implicitly defined as _X_ 0 = _α_ 0 _X_ 1 + _β_ 0 _ξ_ , which is approximated by _X_ 0 _≈ β_ 0 _ξ_ by making _α_ 0 _X_ 1 _≪ β_ 0 _ξ_ . Hence, _π_ 0 is set to be _N_ (0 _, β_ 0<sup>2</sup><sup>_I_) in these methods.Viewed through</sup> our framework, there is no reason to restrict _ξ_ to be _N_ (0 _, β_ 0<sup>2</sup><sup>_I_),ornotset</sup><sup>_α_0=0</sup><sup>_, β_0=1toavoidthe</sup> approximation.


Rectified flow VP ODE sub-VP ODE<br>( αt =  t ,  βt = 1  − t ) ( αt  in (7),  βt = �1  − αt 2) ( αt  in (7),  βt = 1  − αt 2)<br>1-Rectified Flow 2-Rectified Flow 1-Rectified Flow 2-Rectified Flow 1-Rectified Flow 2-Rectified Flow<br>

Figure 4: Comparing rectified flow with VP ODE and sub-VP ODE when _π_ 0 = _N_ (0 _, I_ ) (purple dots) and _π_ 1 is a low variance Gaussian mixture shown as the red dots. The linear rectified flow yields nearly straight trajectories with one step of reflow. But the trajectories of VP ODE and sub-VP ODE are curved and can not be straightened by reflowing.

**VP ODE and sub-VP ODE** The VP ODE and sub-VP ODE of [73] use the following shared _αt_ :


|Time-Discretization|Rectified Flow|VP ODE<br>|sub-VP ODE<br>|VP ODE (const speed)<br>|
|---|---|---|---|---|
|Steps_N_|_αt_ =_t, βt_ = 1_−t_|_αt_in (7), _βt_ =<br>�<br>1_−α_<sup>2</sup><br>_t_|_αt_in (7), _βt_ = 1_−α_<sup>2</sup><br>_t_|_αt_ =_t_, _βt_ =<br>�<br>1_−α_<sup>2</sup><br>_t_|
|_N_ = 1|||||
|_N_ = 2|||||
|_N_ = 5|||||
|_N_ = 100|||||


Figure 5: Trajectories of different methods when varying the number of discretization steps _N_ (purple dots: _π_ 0; red dots: _π_ 1; orangle dots: intermediate steps; blue curves: flow trajectories). The rectified flow travels in straight lines and progresses uniformly in time; it generates the mean of _π_ 1 when simulated with a single Euler step, and quickly covers the whole distribution _π_ 1 with more steps (in this case _N_ = 2 is sufficient). In comparison, VP ODE and sub-VP ODE travel in curves with non-uniform speed: they tend to be slow in the beginning and speed up in the later phase (much of the update happens when _t_ ⪆0 _._ 5). The non-uniform speed can be avoided by setting _αt_ = _t_ (see the last column).

where the default values of _a, b_ are chosen to match the continuous time limit of the shared training procedure of DDIM and DDPM. The difference of VP ODE and sub-VP ODE is on the choice of _βt_ , given as follows:


As _β_ 0 _≈_ 1 in both VP and sub-VP ODE, the _π_ 0 in both cases are taken as _N_ (0 _, I_ ).

The choices of _αt, βt_ above are the consequence of the SDE-based derivation in [73]. However, they are not well-motivated when we exam the path properties of the induced ODEs:

_• Non-straight paths:_ Due to choices of _βt_ in (8), the trajectories of VP ODE and sub-VP ODE are curved in general, and can not be straightened by the reflow procedure. We should choose _βt_ = 1 _− αt_ to induce straight paths.

_• Non-uniform speed:_ The exponential form of _αt_ in (7) is a consequence of using Ornstein–Uhlenbeck processes in the derivation of SDE models [73, 23]. However, there is no clear advantage of using (7) for ODEs. As shown in Figure 5, the _αt_ and _βt_ of VP and sub-VP ODE change slowly in the early phase ( _t_ ⪅0 _._ 5). As a result, the flow also moves slowly in beginning and hence most of the updates are concentrated in the later phase. Such non-uniform update speed, in addition to the non-straight paths, make VP ODE and sub-VP ODE perform


Figure 6: _t_ vs. _αt, βt_ of different methods.

sub-optimally when using large step sizes, even for transport between simple spherical Gaussian distributions (see Figure 5). As we show in the last column of Figure 5, changing the exponential _αt_ to the linear function _αt_ = _t_ in VP ODE allows us to get a uniform update speed while preserving the same continuoustime trajectories.


**VE ODE** The VE ODE of [73] uses _αt_ = 1 and _βt_ = _σ_ min ~~�~~ _r_<sup>2(1</sup><sup>_−t_)</sup> _−_ 1 where _σ_ min = 0 _._ 01 by default _r_ is set such that _σ_ max := _rσ_ min is as large as the maximum Euclidean distance between all pairs of training data points from _π_ 1 (Technique 1 of [72]). Assume that _σ_ max<sup>2is much larger than both</sup><sup>_σ_</sup> min<sup>2and the variance</sup> of _X_ 1, then _X_ 0 = _X_ 1 + _β_ 0 _ξ ≈ σ_ max _ξ_ , and we can set the initial distribution to be _π_ 0 _∼N_ (0 _, σ_ max<sup>2</sup><sup>_I_),</sup> which has much larger variance than _π_ 1. Hence, VE ODE can not be applied to (and not shown in) the toys in Figure 4 and Figure 5. As the case of (sub-)VP ODE, the restriction on _ξ_ is in fact unnecessary and requirement that _σ_ max is unnatural viewed from our framework. On the other hand, the trajectories of _Xt_ in VE ODE are indeed straight lines, because the direction of _X_<sup>˙</sup> _t_ = _β_<sup>˙</sup> _tξ_ is always the same as _ξ_ . However, the choice of _βt_ causes a non-uniform speed issue similar to that of (sub-)VP ODE.

Following [73, 23], a line of works have been proposed to improve the choices of _αt, βt_ , but remain to be constrained by the basic design space from the SDE-to-ODE derivation; see for example [54, 29, 95].

To summarize, the simple nonlinear rectified flow framework in (6) both simplifies and extends the existing framework, and sheds a number of importance insights:

- Learning ODEs can be considered directly and independently without resorting to diffusion/SDE methods;

- The paths of the learned ODEs can be specified by any smooth interpolation curve _Xt_ of _X_ 0 and _X_ 1;

- The initial distribution _π_ 0 can be chosen arbitrarily, independent with the choice of the interpolation _Xt_ .

- The canonical linear interpolation _Xt_ = _tX_ 1 + (1 _− t_ ) _X_ 0 should be recommended as a default choice.

On the other hand, non-linear choices of _Xt_ can be useful when we want to incorporate certain non-Euclidan geometry structure of the variable, or want to place certain constraints on the trajectories of the ODEs. We leave this for future works.

# 3 Theoretical Analysis

We present the theoretical analysis for rectified flow. The results are summarized as follows.

- [Section 3.1] All nonlinear rectified flows with any interpolation _Xt_ preserve the marginal laws.

- [Section 3.2] The rectified flow (with the canonical linear interpolation) reduces convex transport costs.

- [Section 3.3] Reflow guarantees to straighten the (linear) rectified flows.

- [Section 3.4] We clarify the relation between straight couplings and _c_ -optimal couplings.

- [Section 3.5] We establish PF-ODEs as instances of nonlinear rectified flows.

## 3.1 The Marginal Preserving Property

The marginal preserving property that Law( _Zt_ ) = Law( _Xt_ ) for _∀t_ is a general property of the nonlinear rectified flows in (6), regardless whether the interpolation _Xt_ is straight or not.

**Definition 3.1.** _For a path-wise continuously differentiable random process_ **_X_** = _{Xt_ : _t ∈_ [0 _,_ 1] _}, its expected velocity v_<sup>**_X_**</sup> _is defined as_


_For x̸ ∈_ supp( _Xt_ ) _, the conditional expectation is not defined and we set v_<sup>**_X_**</sup> _arbitrarily, say v_<sup>**_X_**</sup> ( _x, t_ ) = 0 _._


**Definition 3.2.** _We call that_ **_X_** _is rectifiable if v_<sup>**_X_**</sup> _is locally bounded and the solution of the integral equation below exists and is unique:_


_In this case,_ **_Z_** = _{Zt_ : _t ∈_ [0 _,_ 1] _} is called the rectified flow induced from_ **_X_** _._

**Theorem 3.3.** _Assume_ **_X_** _is rectifiable and_ **_Z_** _is its rectified flow. Then_ Law( _Zt_ ) = Law( _Xt_ ) _for ∀t ∈_ [0 _,_ 1] _._

_Proof._ For any compactly supported continuously differentiable test function _h_ : R<sup>_d_</sup> _→_ R, we have


where we used _v_<sup>**_X_**</sup> ( _Xt, t_ ) = E[ _X_<sup>˙</sup> _t|Xt_ ]. By definition, this is equivalent to that _πt_ := Law( _Xt_ ) solves in the sense of distributions the continuity equation with drift _vt_<sup>**_X_**</sup> := _v_<sup>**_X_**</sup> ( _·, t_ ):


To see the equivalence of (10) and (11), we can multiply (11) with _h_ and integrate both sides:


Because _Zt_ is driven by the same velocity field _v_<sup>**_X_**</sup> , its marginal law Law( _Zt_ ) solves the very same equation with the same initial condition ( _Z_ 0 = _X_ 0). Hence, the equivalence of Law( _Zt_ ) and Law( _Xt_ ) follows if the solution of (11) is unique, which is equivalent to the uniqueness of the solution of d _Zt_ = _v_<sup>**_X_**</sup> ( _Zt, t_ ) following Corollary 1.3 of Kurtz [37] (see also Theorem 4.1 of Ambrosio and Crippa [1]).

## 3.2 Reducing Convex Transport Costs

The fact that ( _Z_ 0 _, Z_ 1) yields no larger convex transport costs than ( _X_ 0 _, X_ 1) is a consequence of using the special linear interpolation _Xt_ = _tX_ 1 + (1 _− t_ ) _X_ 0 as the geodesic of Euclidean space.

**Definition 3.4.** _A coupling_ ( _X_ 0 _, X_ 1) _is called rectifiable if its linear interpolation process_ **_X_** = _{tX_ 1 + (1 _− t_ ) _X_ 0 : _t ∈_ [0 _,_ 1] _} is rectifiable. In this case, the_ **_Z_** = _{Zt_ : _t ∈_ [0 _,_ 1] _} in_ (9) _is called the rectified flow of coupling_ ( _X_ 0 _, X_ 1) _, denoted as_ **_Z_** = `RectFlow` (( _X_ 0 _, X_ 1)) _, and_ ( _Z_ 0 _, Z_ 1) _is called the rectified coupling of_ ( _X_ 0 _, X_ 1) _, denoted as_ ( _Z_ 0 _, Z_ 1) = `Rectify` (( _X_ 0 _, X_ 1)) _._

**Theorem 3.5.** _Assume_ ( _X_ 0 _, X_ 1) _is rectifiable and_ ( _Z_ 0 _, Z_ 1) = `Rectify` (( _X_ 0 _, X_ 1)) _. Then for any convex function c_ : R<sup>_d_</sup> _→_ R _, we have_


_Proof._ The proof is based on elementary applications of Jensen’s inequality.


If _Xt_ is straight but with positive non-constant speed, that is, _Xt_ = _αtX_ 1 + _βtX_ 0 with _βt_ = 1 _− αt_ and _α_ ˙ _t ≥_ 0, then we still have E[ _c_ ( _Z_ 1 _− Z_ 0)] _≤_ E[ _c_ ( _X_ 1 _− X_ 0)] if _c_ is convex and _m_ -homogeneous in that _c_ ( _ax_ ) = _|a|_<sup>_m_</sup> _c_ ( _x_ ) for _∀a ∈_ R _, x ∈_ R<sup>_d_</sup> , with some constant _m ∈_ (0 _,_ 1].

## 3.3 The Straightening Effect

A coupling ( _X_ 0 _, X_ 1) is said to be straight (or fully rectified) if it is a fixed point of the `Rectify` ( _·_ ) mapping. It is desirable to obtain a straight coupling because its rectified flow is straight and hence can be simulated exactly with one step using numerical solvers. In this section, we first characterize the basic properties of straight couplings, showing that a coupling is straight iff its linear interpolation paths do not intersect with each other. Then, we prove that recursive rectification straightens the coupling and its related flow with a _O_ (1 _/k_ ) rate, where _k_ is the number of rectification steps.

**Theorem 3.6.** _Assume_ ( _X_ 0 _, X_ 1) _is rectifiable. Let Xt_ = _tX_ 1 + (1 _− t_ ) _X_ 0 _and_ **_Z_** = `RectFlow` (( _X_ 0 _, X_ 1)) _. Then_ ( _X_ 0 _, X_ 1) _is a straight coupling iff the following equivalent statements hold._

_1. There exists a strictly convex function c_ : R<sup>_d_</sup> _→_ R _, such that_ E[ _c_ ( _Z_ 1 _− Z_ 0)] = E[ _c_ ( _X_ 1 _− X_ 0)] _._

_2._ ( _X_ 0 _, X_ 1) _is a fixed point of_ `Rectify` ( _·_ ) _, that is,_ ( _X_ 0 _, X_ 1) = ( _Z_ 0 _, Z_ 1) _._

_3. The rectified flow coincides with the linear interpolation process:_ **_X_** = **_Z_** _._

_4. The paths of the linear interpolation_ **_X_** _do not intersect:_


_where V_ (( _X_ 0 _, X_ 1)) = 0 _indicates that X_ 1 _− X_ 0 = E[ _X_ 1 _− X_ 0 _|Xt_ ] _almost surely when t ∼ Uniform_ ([0 _,_ 1]) _, meaning that the lines passing through each Xt is unique, and hence no linear interpolation paths intersect._


_Proof._ 3 _→_ 2 _→_ 1: Obvious.

1 _→_ 4: If E[ _c_ ( _Z_ 1 _− Z_ 0)] = E[ _c_ ( _X_ 1 _− X_ 0)], the two applications of Jensen’s inequality in the proof of Theorem 3.5 are tight. Because _c_ is strictly convex, the second Jensen’s inequality in the proof implies that _X_ 1 _−X_ 0 = E[ _X_ 1 _−X_ 0 _| Xt_ ] almost surely w.r.t. _X_ and _t ∼_ Uniform([0 _,_ 1]) _,_ which implies that _V_ ( **_X_** ) = 0 _._ 4 _→_ 3: If _V_ ( **_X_** ) = 0, we have �0 _s_<sup>(</sup><sup>_X_1</sup><sup>_−X_0)d</sup><sup>_t_=</sup> �0 _s_<sup>E[</sup><sup>_X_1</sup><sup>_−X_0</sup><sup>_|Xt_]d</sup><sup>_t_=</sup> �0 _s_<sup>_vX_(</sup><sup>_Xt, t_)d</sup><sup>_t_for</sup><sup>_s∈_(0</sup><sup>_,_1].</sup> Hence


Because **_Z_** satisfies the same equation (9), we have **_X_** = **_Z_** by the uniqueness of the solution.

_O_ (1 _/K_ ) **convergence rate** We now show that as we apply rectification recursively, the rectified flows become increasingly straight and the linear interpolation of the couplings becomes increasingly non-intersecting. **Theorem 3.7.** _Let_ **_Z_**<sup>_k_</sup> _the k-th rectified flow of_ ( _X_ 0 _, X_ 1) _, that is,_ **_Z_**<sup>_k_+1</sup> = `RectFlow` (( _Z_ 0<sup>_k, Z_</sup> 1<sup>_k_))</sup><sup>_and_</sup> ( _Z_ 0<sup>0</sup><sup>_, Z_</sup> 1<sup>0) = (</sup><sup>_X_0</sup><sup>_, X_1)</sup><sup>_.Assume each_(</sup><sup>_Z_</sup> 0<sup>_k, Z_</sup> 1<sup>_k_)</sup><sup>_is rectifiable for k_= 0</sup><sup>_, . . . , K._</sup> _Then_


Hence, E[ _∥X_ 1 _− X_ 0 _∥_<sup>2</sup> ] _<_ + _∞_ , we have min _k≤K_ ( _S_ ( **_Z_**<sup>_k_</sup> ) + _V_ (( _Z_ 0<sup>_k, Z_</sup> 1<sup>_k_)) =</sup><sup>_O_(1</sup><sup>_/K_).</sup> _Proof._ Taking _c_ ( _x_ ) = _∥x∥_<sup>2</sup> in the proof of Theorem 3.5, we can obtain that


Applying it to each rectification step yields


A telescoping sum on _k_ = 0 _, . . . , K_ gives the result.

## 3.4 Straight vs. Optimal Couplings

A coupling ( _X_ 0 _, X_ 1) is called _c_ -optimal if it achieves the minimum of E[ _c_ ( _X_ 1 _− X_ 0)] among all couplings that share the same marginals. Understanding and computing the optimal couplings have been the main focus of optimal transport [e.g., 84, 2, 15, 59]. Straight couplings is a different desirable property. In the following, we show that straightness is a necessary but not sufficient condition of being _c_ -optimal for a strictly convex function _c_ , except in the one dimensional case when the two concepts coincides. Hence, it is “easier” to find a straight coupling than a _c_ -optimal couplings.


**Theorem 3.8.** _If a rectifiable coupling_ ( _X_ 0 _, X_ 1) _is c-optimal for some strictly convex cost function c, then_ ( _X_ 0 _, X_ 1) _is a straight coupling._

_Proof._ Let ( _Z_ 0 _, Z_ 1) = `Rectify` (( _X_ 0 _, X_ 1)). If ( _X_ 0 _, X_ 1) is _c_ -optimal, we must have E[ _c_ ( _Z_ 1 _− Z_ 0)] = E[ _c_ ( _X_ 1 _− X_ 0)]. This implies Statement 1 in Theorem 3.6 and hence that ( _X_ 0 _, X_ 1) is straight.

**1D Case** For any _π_ 0 _, π_ 1 on R, there exists an unique coupling ( _X_ 0<sup>_∗, X_</sup> 1<sup>_∗_) that is simultaneously optimal for</sup> all non-negative convex cost functions _c_ . This coupling is uniquely characterized by a monotonic property: for every ( _x_ 0 _, x_ 1) and ( _x_<sup>_′_</sup> 0<sup>_, x_</sup> 1<sup>_′_) in the support of (</sup><sup>_X_</sup> 0<sup>_∗, X_</sup> 1<sup>_∗_),if</sup><sup>_x_0</sup><sup>_<x′_</sup> 0<sup>,then</sup><sup>_x_1</sup><sup>_≤x_</sup> 1<sup>_′_.Furthermore,if</sup><sup>_π_0</sup> is absolutely continuously w.r.t. the Lebesgue measure, then ( _X_ 0<sup>_∗, X_</sup> 1<sup>_∗_) must be deterministic in that there</sup> exists a mapping _T_ : R _→_ R such that _X_ 1<sup>_∗_=</sup><sup>_T_(</sup><sup>_X_</sup> 0<sup>_∗_).See [65].</sup>

In the following, we show that straight couplings on R coincides with the deterministic monotonic coupling ( _X_ 0<sup>_∗, X_</sup> 1<sup>_∗_)andhenceisuniqueandsimultaneouslyoptimalforallconvex</sup><sup>_c_when</sup><sup>_π_0isabsolutelycontin-</sup> uous. The idea is that, in R, a coupling is monotonic iff its linear interpolation paths do not intersect, a characteristic feature of straight couplings.

**Lemma 3.9.** _A coupling on_ R _is straight iff it is deterministic and monotonic._

**Theorem 3.10.** _For any π_ 0 _, π_ 1 _on_ R _, there exists either no straight coupling, or a unique straight coupling. Further, if exists, the unique straight coupling is deterministic and monotonic, and jointly optimal w.r.t. all convex cost functions c_ : R<sup>_d_</sup> _→_ [0 _,_ + _∞_ ) _for which the minimum value of_ E [ _c_ ( _X_ 1 _− X_ 0)] _exists and is finite._

_Proof of Lemma 3.9._ If ( _X_ 0 _, X_ 1) on R is straight, then it coincides with its rectified coupling ( _Z_ 0 _, Z_ 1) = `Rectify` (( _X_ 0 _, X_ 1)) _._ But because ( _Z_ 0 _, Z_ 1) is induced from the rectified flow d _Zt_ = _v_<sup>_X_</sup> ( _Zt, t_ )d _t_ , it is obviously deterministic. It is also monotonic due to the non-crossing property of flows. Specifically, if ( _Z_ 0 _, Z_ 1) is not monotonic, there exists ( _z_ 0 _, z_ 1) and ( _z_ 0<sup>_′, z_</sup> 1<sup>_′_) in the support of (</sup><sup>_Z_0</sup><sup>_, Z_1) such that</sup><sup>_z_0</sup><sup>_< z_</sup> 0<sup>_′_and</sup> _z_ 1 _> z_ 1<sup>_′_.Ifthishappens,theremustexists</sup><sup>_t_0</sup><sup>_∈_(0</sup><sup>_,_1),suchthat</sup><sup>_zt_</sup> 0<sup>=</sup><sup>_z_</sup> _t_<sup>_′_</sup> 0<sup>.Butbytheuniquenessofthe</sup> solution, we have _zt_ = _zt_ for _t ≥ t_ 0, which is conflicting with _z_ 1 _> z_ 1<sup>_′_.</sup>

Assume ( _X_ 0 _, X_ 1) is deterministic and monotonic. Due to the monotonicity, there exists no _x_ 0 and _x_<sup>_′_</sup> 0<sup>in</sup> the support of _π_ 0, such that _x_ 0 _̸_ = _x_ 0<sup>_′_and</sup><sup>_xt_</sup> 0<sup>=</sup><sup>_x′_</sup> _t_ 0<sup>forsome</sup><sup>_t_0</sup><sup>_<_1.Thissuggeststhat</sup><sup>_X_1</sup><sup>_−X_0=</sup> E[ _X_ 1 _− X_ 0 _| Xt_ ] = _v_<sup>_X_</sup> ( _Xt, t_ ) for _t ∈_ (0 _,_ 1), and hence d _Xt_ = ( _X_ 1 _− X_ 0)d _t_ = _v_<sup>_X_</sup> ( _Xt_ )d _t_ , which is the ODE of the rectified flow. In addition, _Xt_ is obviously the unique solution of this ODE. Hence ( _X_ 0 _, X_ 1) is rectifiable and straight following Statement 3 of Theorem 3.6.

_Proof of Theorem 3.10._ This is the result of Lemma 3.9 combined with the fact that the monotonic coupling is unique and jointly optimal for all convex _c_ for which the optimal coupling exists, following Lemma 2.8 and Theorem 2.9 of [65].

**Multi-dimensional cases** On the other hand, on R<sup>_d_</sup> with _d ≥_ 2, the different cost functions _c_ do not share a common optimal coupling in general, and a straight coupling is not guaranteed to optimize a specific _c_ ; this is expected because the `Rectify` ( _·_ ) procedure does not depend on a particular choice of _c_ . Hence, one must modify the `Rectify` ( _·_ ) procedure to tailor it to a specific _c_ of interest.

In a recent work [30], it was conjectured that the couplings ( _Z_ 0 _, Z_ 1) induced from VP ODE (equivalently DDIM) yields an optimal coupling w.r.t. the quadratic loss, which was proved to be false in [39, 78]. Here


we show that even straight couplings are not guaranteed to be optimal, not to mention that VP ODE does not follow straight paths by design.

We explore this in a separate work [42] that is devoted to modifying rectified flow to find _c_ -optimal couplings; a result from [42] that can be easily stated is that the optimal coupling w.r.t. the quadratic cost _c_ ( _·_ ) = _∥·∥_<sup>2</sup> can be achieved as the fixed point of `Rectify` ( _·_ ) if _v_ is restricted to be a gradient field of form _v_ ( _x, t_ ) = _∇f_ ( _x, t_ ) when solving (1). Restricting _v_ to be a gradient field removes the rotational component of the velocity field _v_<sup>**_X_**</sup> that causes sub-optimal transport cost.

## 3.5 Denoising Diffusion Models and Probability Flow ODEs

We prove that the probability flow ODEs (PF-ODEs) of [73] can be viewed as nonlinear rectified flows in (6) with _Xt_ = _αtX_ 1 + _βtξ._ We start with introducing the algorithmic procedures of the denoising diffusion models and PF-ODEs, and refer the readers to the original works [73, 23, 70] for the theoretical derivations.

The denoising diffusion methods learn to generative models by constructing an SDE model driven by a standard Brownian motion _Wt_ :


where _σt_ : [0 _,_ 1] _→_ [0 _,_ + _∞_ ) is a (typically) fixed diffusion coefficient, _b_ is a trainable neural network, and the initial distribution _π_ 0 is restricted to a spherical Gaussian distribution determined by hyper-parameter setting of the algorithm. The idea is to first collapse the data into an (approximate) Gaussian distribution using a diffusion process, mostly an Ornstein-Uhlenbeck (OU) process, and then estimate the generative diffusion process (14) as the time reversal [e.g., 3] of the collapsing process.

Without diving into the derivations, the training loss of the VE, VP, sub-VP SDEs for _b_ in [73] can be summarized as follows:


where _ξt_ is a diffusion process satisfying _ξt ∼N_ (0 _, I_ ), and _ηt, σt_ are the hyper-parameter sequences of the algorithm, and _αt, βt_ are determined by _ηt, σt_ via


The relation in (16) is derived to make _V_<sup>˜</sup> _t_ = _V_ 1 _−t_ = _α_ 1 _−tX_ 1 + _β_ 1 _−tξt_ follow the Ornstein-Uhlenbeck (OU) processes d _V_<sup>˜</sup> _t_ = _η_ 1 _−tV_<sup>˜</sup> _t_ d _t_ + _σ_ 1 _−t_ d _Wt_ .

VE SDE, which is equivalent to SMLD in [71, 72], takes _ηt_ = 0 and hence has _αt_ = 1. (sub-)VP SDE takes _ηs_ to be a linear function of _s_ , yielding the exponential _αt_ in (7). VP SDE (which is equivalent to DDPM [23]) takes _ηt_ = _−_ 2<sup><u>1</u></sup><sup>_σ_</sup> _t_<sup>2which yields that</sup><sup>_α_</sup> _t_<sup>2+</sup><sup>_β_</sup> _t_<sup>2= 1 as shown in (8).In DDPM, it was suggested to write</sup> _<u>t</u> b_ ( _x, t_ ) = _−ηtx −_<sup>_σ_</sup> _β_<sup>2</sup> _t_<sup>_ϵ_(</sup><sup>_x, t_) , and estimate</sup><sup>_ϵ_as a neural network that predicts</sup><sup>_ξt_from (</sup><sup>_Vt, t_).</sup>

Theoretically, the SDE in (14) with _b_ solving (15) is ensured to yield Law( _U_ 1) = Law( _X_ 1) = _π_ 1 when initialized from _U_ 0 = _α_ 0 _X_ 1 + _β_ 0 _ξ_ 0, which can be approximated by _U_ 0 _≈ βξ_ 0 when _α_ 0 _X_ 1 _≪ β_ 0 _ξ_ 0.


By using the properties of Fokker-Planck equations, it was observed in [73, 70] that the SDE in (14) with _b_ trained in (15) can be converted into an ODE that share the same marginal laws:


Equivalently, we can regard<sup>˜</sup> _b_ as the solution of


which defers from (14) only by a factor of 1 _/_ 2 in the second term of _Yt_ . This simple equivalence holds only when (14) and (17) use the special initialization of _Z_ 0 = _U_ 0 = _α_ 0 _X_ 1 + _β_ 0 _ξ_ 0.

In the following, we are ready to prove that (18) is can be viewed as the nonlinear rectified flow objective in (6) using _Xt_ = _αtX_ 1 + _βtξ_ with _ξ ∼N_ (0 _, I_ ). We mainly need to show that _Y_<sup>˜</sup> _t_ is equivalent to _X_<sup>˙</sup> _t_ by eliminating _ηt_ and _σt_ using the relation in (16).

**Proposition 3.11.** _Assume_ (16) _hold. Then_ (18) _is equivalent to_ (6) _with Xt_ = _αtX_ 1 + _βtξ._

_Proof._ First, note that we can take _ξt_ = _ξ_ for all time _t_ , as the correlation structure of _ξt_ does not impact the result. Hence, we have _Vt_ = _Xt_ = _αtX_ 1 + _βtξ_ . To show the equivalence of (18) and (6), we just need to verify that _X_<sup>˙</sup> _t_ = _Y_<sup>˜</sup> _t_ .


where in (= _∗_ ) we used that _ηt_ = _−_<sup>_<u>α</u>_</sup> _α_<sup>˙</sup><sup>_<u>t</u>_</sup> _t_<sup>and</sup><sup>_σ_</sup> _t_<sup>2= 2</sup><sup>_β_</sup> _t_<sup>2</sup> � _<u>αα</u>_ ˙ _<u>tt</u>_<sup>_−_</sup> _<u>ββ</u>_ ˙ _<u>tt</u>_ � which can be derived from (16).

# 4 Related Works and Discussion

**Learning one-step models** GANs [19, 4, 43], VAEs [32], and (discrete-time) normalizing flows [62, 13, 14] have been three classical approaches for learning deep generative models. GANs have been most successful in terms of generation qualities (for images in particular), but suffer from the notorious training instability and mode collapse issues due to use of minimax updates. VAEs and normalizing flows are both trained based on the principle of maximum likelihood estimation (MLE) and need to introduce constraints on the model architecture and/or special approximation techniques to ensure tractable likelihood computation: VAEs typically use a conditional Gaussian distribution in addition to the variational approximation of the likelihood; normalizing flows require to use specially designed invertible architectures and need to copy with calculating expensive Jacobian matrices.

The reflow+distillation approach in this work provides another promising approach to training one-step models, avoiding the minimax issues of GANs and the intractability issues of the likelihood-based methods.


**Learning ODEs: MLE and PF-ODEs** There are two major approaches for learning neural ODEs: the PF-ODEs/DDIM approach discussed in Section 2.3, and the more classical MLE based approach of [6].

_• The MLE approach._ In [6], neural ODEs are trained for learning generative models by maximizing the likelihood of the distribution of the ODE outcome _Z_ 1 at time _t_ = 1 under the data distribution _π_ 1. Specifically, with observations from _π_ 1, it estimates a neural drift _v_ of an ODE d _Zt_ = _v_ ( _Zt, t_ )d _t_ by


where D( _·_ ; _·_ ) denotes KL divergence (or other discrepancy measures), and _ρ_<sup>_v,π_0</sup> is the density of _Z_ 1 following d _Zt_ = _v_ ( _Zt, t_ )d _t_ from _Z_ 0 _∼ π_ 0; the density of _π_ 0 should be known and tractable to calculate.

By using an instantaneous change of variables formula, it was observed in [6] that the likelihood of neural ODEs are easier to compute than the discrete-time normalizing flow without constraints on the model structures. However, this MLE approach is still computationally expensive for large scale models as it requires repeated simulation of the ODE during each training step. In addition, as the optimization procedure of MLE requires to backpropagate through time, it can easily suffer the gradient vanishing/exploding problem unless proper regularization is added.

Another fundamental problem is that the MLE (19) of neural ODEs is theoretically under-specified, because MLE only concerns matching the law of the final outcome _Z_ 1 with the data distribution _π_ 1, and there are infinitely many ODEs to achieve the same output law of _Z_ 1 while traveling through different paths. A number of works have been proposed to remedy this by adding regularization terms, such as these based on transport costs, to favor shorter paths; see [54, 55]. With a regularization term, the ODE learned by MLE would be implicitly determined by the initialization and other hyper-parameters of the optimizer used to solve (19).

_• Probability Flow ODEs._ The method of PF-ODEs [73] and DDIM [70] provides a different approach to learning ODEs that avoids the main disadvantages of the MLE approach, including the expensive likelihood calculation, training-time simulation of the ODE models, and the need of backpropagation through time. However, because PF-ODEs and DDIM were derived as the side product of learning the mathematically more involved diffusion/SDE models, their theories and algorithm forms were made unnecessarily restrictive and complicated. The nonlinear rectified flow framework shows that the learning of ODEs can be approached directly in a very simple way, allowing us to identify the canonical case of linear rectified flow and open the door of further improvements with flexible and decoupled choices of the interpolation curves _Xt_ and initial distributions _π_ 0 _._

Viewed through the general non-linear rectified flow framework, the computational and theoretical drawbacks of MLE can be avoided because we can simply pre-determines the “roads” that the ODEs should travel through by specifying the interpolation curve _Xt_ , rather than leaving it for the algorithm to figure out implicitly. It is theoretically valid to pre-specify any interpolation _Xt_ because the neural ODE is highly over-parameterized as a generative model: when _v_ is a universal approximator and _π_ 0 is absolutely continuous, the distribution of _Z_ 1 can approximate any distribution given any fixed interpolation curve _Xt_ . The idea of rectified flow is to the simplest geodesic paths for _Xt_ .

**Learning SDEs with denoising diffusion** Although the scope of this work is limited to learning ODEs, the score-based generative models [71–74] and denoising diffusion probability models (DDPM) [23] are of high relevance as the basis of PF-ODEs and DDIM. The diffusion/SDE models trained with these methods


have been found outperforming GANs in image synthesis in both quality and diversity [12]. Notably, thanks to the stable and scalable optimization-based training procedure, the diffusion models have successfully used in huge text-to-image generation models with astonishing results [e.g., 53, 61, 64]. It has been quickly popularized in other domains, such as video [e.g., 24, 92, 21], music [51], audio [e.g., 33, 40, 60], and text [41, 88], and more tasks such as image editing [97, 50]. A growing literature has been developed for improving the inference speed of denoising diffusion models, an example of which is the PF-ODEs/DDIM approach which gains speedup by turning SDEs into ODEs. We provide below some examples of recent works, which is by no mean exhaustive.

_• Improved training and inference._ A line of works focus on improving the inference and sampling procedure of denoising diffusion models. For example, [54] presents a few simple modifications of DDPM to improve the likelihood, sampling speed, and generation quality. [29] systematic exams the design space of diffusion generative models with empirical studies and identifies a number of training and inference recipes for better generative quality with fewer sampling steps. [94] proposes a diffusion exponential integrator sampler for fast sampling of diffusion models. [46] provides a customized high order solver for PF-ODEs. [5] provides an analytic estimate of the optimal diffusion coefficient.

_• Combination with other methods._ Another direction is to speed up diffusion models by combining them with GANs and other generative models. DDPM Distillation [47] accelerates the inference speed by distilling the trajectories of a diffusion model into a series of conditional GANs. The truncated diffusion probabilistic model (TDPM) of [99] trains a GAN model as _π_ 0 so that the diffusion process can be truncated to improve the speed; the similar idea was explored in [48, 18], and [18] provides an analysis on the optimal truncation time. [68, 89, 81] learns a denoising diffusion model in the latent spaces and combines it with variational auto-encoders. These methods can be potentially applied to rectified flow to gain similar speedups for learning neural ODEs.

_• Unpaired Image-to-Image translation._ The standard denoising diffusion and PF-ODEs methods focus on the generative task of transferring a Gaussian noise ( _π_ 0) to the data ( _π_ 1). A number of works have been proposed to adapt it to transferring data between arbitrary pairs of source-target domains. For example, SDEdit [50] synthesizes realistic images guided by an input image by first adding noising to the input and then denoising the resulting image through a pre-trained SDE model. [8] proposes a method to guide the generative process of DDPM to generate realistic images based on a given reference image. [75] leverages two two PF-ODEs for image translation, one translating source images to a latent variable, and the other constructing the target images from the latent variable. [97] proposes an energy-guided approach that employs an energy function pre-trained on the source and target domains to guide the inference process of a pretrained SDE for better image translation. In comparison, our framework shows that domain transfer can be achieved by essentially the same algorithm as generative modeling, by simply setting _π_ 0 to be the source domain.

_• Diffusion bridges._ Some recent works [57, 44] show that the design space of denoising diffusion models can be made highly flexible with the assistant of diffusion bridge processes that are pinned to a fixed data point at the end time. This reduces the design of denoising diffusion methods to constructing a proper bridge processes. The bridges in Song et al. [73] are constructed by a time-reversal technique, which can be equivalently achieved by Doob’s _h_ -transform as shown in [57, 44], and more general construction techniques are discussed in [44, 90]. Despite the significantly extended design spaces, an unanswered question is what type of diffusion bridge processes should be preferred. This question is made challenging because the presence of diffusion noise and the need of advanced stochastic calculus tools make it hard to intuit how


the methods work. By removing the diffusion noise, our work makes it clear that straight paths should be preferred. We expect that the idea can be extended to provide guidance on designing optimal bridge processes for learning SDEs.

_• Schrodinger bridges._ Another body of works [87, 11, 7, 82] leverages Schrodinger bridges (SB) as an alternative approach to learning diffusion generative models. These approaches are attractive theoretically, but casts significant computational challenges for solving the Schrodinger bridge problem.

**Re-thinking the role of diffusion noise** The introduction of diffusion noise was consider essential due to the key role it plays in the derivations of the successful methods [73, 23]. However, as rectified flow can achieve better or comparable results with a ODE-only framework, the role of diffusion mechanisms should be re-examed and clearly decoupled from the other merits of denoising diffusion models. The success of the denoising diffusion models may be mainly attributed to the simple and stable optimization-based training procedure that allows us to avoid the instability issues and the need of case-by-case tuning of GANs, rather than the presence of diffusion noises.

Because our work shows that there is no need to invoke SDE tools if the goal is to learn ODEs, the remaining question is whether we should learn an ODE or an SDE for a given problem. As already argued by a number of works [73, 70, 29], ODEs should be preferred over SDEs in general. Below is a detailed comparison between ODEs and SDEs.

_• Conceptual simplicity and numerical speed._ SDEs are more mathematically involved and are more difficult to understand. Numerical simulation of ODEs are simpler and faster than SDEs.

_• Time reversibility._ It is equally easy to solve the ODEs forwardly and backwardly. In comparison, the time reversal of SDEs [e.g., 3, 22, 17] is more involved theoretically and may not be computationally tractable.

_• Latent spaces._ The couplings ( _Z_ 0 _, Z_ 1) of ODEs are deterministic and yield low transport cost in the case of rectified flows, hence providing a good latent space for representing and manipulating outputs. Introducing diffusion noises make ( _Z_ 0 _, Z_ 1) more stochastic and hence less useful. In fact, the ( _Z_ 0 _, Z_ 1) given by DDPM [23] and the SDEs of [73] and hence useless for latent presentation.

_• Training difficulty._ There is no reason to believe that training an ODE is harder, if not easier, than training an SDE sharing the same marginal laws: the training loss of both cases would share the distributions of covariant and differ only on the targets. In the setting of [73], the two loss functions (15) and (18) are equivalent upto a linear reparameterization.

_• Expressive power._ As every SDE can be converted into an ODE that has the same marginal distribution using the techniques in [70, 73] (see also [84]), ODEs are as powerful as SDEs for representing marginal distributions, which is what needed for the transport mapping problems considered in this work. On the other hand, SDEs may be preferred if we need to capture richer time-correlation structures.

_• Manifold data._ When equipped with neural network drifts, the outputs of ODEs tend to fall into a smooth low dimensional manifold, a key inductive for structured data in AI such as images and text. In comparison, when using SDEs to model manifold data, one has to carefully anneal the diffusion noise to obtain smooth outcomes, which causes slow computation and a burden of hyperparameter tuning. SDEs might be more useful in for modeling highly noisy data in areas like finance and economics, and in areas that involve diffusion processes physically, such as molecule simulation.


**Optimal vs. straight transport** Optimal transport has been extensively explored in machine learning as a powerful way to compare and transfer between probability measures. For the transport mapping problem considered in this work, a natural approach is to finding the optimal coupling ( _Z_ 0 _, Z_ 1) that minimizes a transport cost E[ _c_ ( _Z_ 1 _− Z_ 0)] for a given _c_ . The most common choice of _c_ is the quadratic cost _c_ ( _·_ ) = _∥·∥_<sup>2</sup> .

However, finding the optimal couplings, especially for high dimensional continuous measures, is highly challenging computationally and is the subject of active research; see for example [67, 34, 35, 49, 63, 10]. In addition, although the optimal couplings are known to have nice smoothness and other regularity properties, it is not necessary to accurately find the optimal coupling because the transport cost do not exactly align with the learning performance of individual problems; see e.g., [34].

In comparison, our reflow procedure finds a straight coupling, which is not optimal w.r.t. a given _c_ (see Section 3.4). From the perspective of fast inference, all straight couplings are equally good because they all yield straight rectified flows and hence can be simulated with one Euler step.

# 5 Experiments

We start by studying the impact of reflow on toy examples. After that, we demonstrate that with multiple times of reflow, rectified flow achieves state-of-the-art performance on CIFAR-10. Moreover, it can also generate high-quality images on high-resolution image datasets. Going beyond unconditioned image generation, we apply our method to unpaired image-to-image translation tasks to generate visually high-quality image pairs.

**Algorithm** We follow the procedure in Algorithm 1. We start with drawing ( _X_ 0 _, X_ 1) _∼ π_ 0 _× π_ 1 and use it to get the first rectified flow **_Z_**<sup>1</sup> by minimizing (1). The second rectified flow **_Z_**<sup>2</sup> is obtained by the same procedure except with the data replaced by the draws from ( _Z_ 0<sup>1</sup><sup>_, Z_</sup> 1<sup>1), obtained by simulating the first</sup> rectified flow **_Z_**<sup>1</sup> . This process is repeated for _k_ times to get the _k-rectified flow_ **_Z_**<sup>_k_</sup> . Finally, we can further distill the _k-rectified flow_ **_Z_**<sup>_k_</sup> into a one step model _z_ 1 = _z_ 0 + _v_ ( _z_ 0 _,_ 0) by fitting it on draws from ( _Z_ 0<sup>_k, Z_</sup> 1<sup>_k_).</sup>

By default, the ODEs are simulated using the vanilla Euler method with constant step size 1 _/N_ for _N_ steps, that is, _Z_<sup>ˆ</sup> _t_ +1 _/N_ = _Z_<sup>ˆ</sup> _t_ + _v_ ( _Z_<sup>ˆ</sup> _t, t_ ) _/N_ for _t ∈{_ 0 _, . . . , N }/N_ . We use the Runge-Kutta method of order 5(4) from Scipy [86], denoted as RK45, which adaptively decide the step size and number of steps _N_ based on user-specified relative and absolute tolerances. In our experiments, we stick to the same parameters as [73].

## 5.1 Toy Examples

To accurately illustrate the theoretical properties, we use the non-parametric estimator _v_<sup>_X,h_</sup> ( _z, t_ ) in (5) in the toy examples in Figure 2, 3, 4, 5. In practice, we approximate the expectation in (5) an nearest neighbor estimator: given a sample _{x_<sup>(</sup> 0<sup>_i_)</sup><sup>_, x_</sup> 1<sup>(</sup><sup>_i_)</sup><sup>_}i_drawn from (</sup><sup>_X_0</sup><sup>_, X_1), we estimate</sup><sup>_vX_by</sup>


where knn( _z, m_ ) denotes the top _m_ nearest neighbors of _z_ in _{x_<sup>(</sup> _t_<sup>_i_)</sup><sup>_}i_.Wefindthattheresultsarenot</sup> sensitive to the choice of _m_ and the bandwidth _h_ (see Figure 7). We use _h_ = 1 and _m_ = 100 by default. The flows are simulated using Euler method with a constant step size of 1 _/N_ for _N_ steps. We use _N_ = 100 steps unless otherwise specified.


Alternatively, _v_<sup>_X_</sup> can be parameterized as a neural network and trained with stochastic gradient descent or Adam. Figure 7 shows an example of when _v_<sup>_X_</sup> is parameterized as an 2-hidden-layer fully connected neural network with 64 neurons in both hidden layers. We see that the neural networks fit less perfectly with the linear interpolation trajectories (which should be piece-wise linear in this toy example). As shown in Figure 7, we find that enhancing the smoothness of the neural networks (by increasing the L2 regularization coefficient during training) can help straighten the flow, in addition to the rectification effect.


1-Rectified Flow 2-Rectified Flow 3-Rectified Flow 1-Rectified Flow 2-Rectified Flow 3-Rectified Flow<br>h  = 001 .<br>L2 Penalty=0<br>h  = 1<br>L2 Penalty= 001 .<br>

Figure 7: Rectified flows fitted with neural networks trained with different L2 penalty (left), and kernel estimator with different bandwidth _h_ (right). _π_ 0: red dots; _π_ 1: purple dots.

In Figure 3 of Section 2.2, the straightness is calculated as the empirical estimation of (3) based on the simulated trajectories. The relative transport cost is calculated based on _{z_ 0<sup>(</sup><sup>_i_)</sup><sup>_, z_</sup> 1<sup>(</sup><sup>_i_)</sup><sup>_}n_</sup> _i_ =1<sup>drawn from (</sup><sup>_Z_0</sup><sup>_, Z_1)</sup> 2 2 by simulating the flow, as _n_<sup><u>1</u></sup> � _ni_ =1 1<sup>_−z_</sup> 0<sup>(</sup><sup>_i_)</sup> _−_ 1 _− z_ 0<sup>(</sup><sup>_i_)</sup> , where _z_ 1<sup>(</sup><sup>_i∗_)</sup> is the optimal L2 assignment ��� _z_ ( _i_ ) ��� ��� _z_ ( _i∗_ ) ��� of _z_ 0<sup>(</sup><sup>_i_)</sup> obtained by solving the discrete L2 optimal transport problem between _{z_ 0<sup>(</sup><sup>_i_)</sup><sup>_}_and</sup><sup>_{z_</sup> 1<sup>(</sup><sup>_i_)</sup><sup>_}_.We should</sup> note that this metric is only useful in low dimensions, as it tends to be identically zero in high dimensional cases even _v_<sup>_X_</sup> is set to be a random neural network. This misleading phenomenon is what causes [30] to make the false hypothesis that DDIM yields L2 optimal transport.

## 5.2 Unconditioned Image Generation

We test rectified flow for unconditioned image generation on CIAFR-10 and a number of high resolution datasets. The methods are evaluated by the quality of generated images by Fr´echet inception distance (FID) and inception score (IS), and the diversity of the generated images by the recall score following [38].

**Experiment settings** For the purpose of generative modeling, we set _π_ 0 to be the standard Gaussian distribution and _π_ 1 the data distribution. Our implementation of rectified flow is modified upon the opensource code of [73]. We adopt the U-Net architecture of DDPM++ [73] for representing the drift _v_<sup>_X_</sup> , and report in Table 1 (a) and Figure 8 the results of our method and the (sub)-VP ODE from [73] using the same architecture. Other recent results using different network architectures are shown in Table 1 (b) for reference. More detailed settings can be found in the Appendix.

**Results** _• Results of fully solved ODEs._ As shown in Table 1 (a), the 1-rectified flow trained on the DDPM++ architecture, solved with RK45, yields the lowest FID (2 _._ 58) and highest recall (0 _._ 57) among all the ODE-based methods. In particular, the recall of 0.57 yields a substantial improvement over existing ODE


|Method|NFE(_↓_)|IS(_↑_)|FID(_↓_)|Recall(_↑_)|
|---|---|---|---|---|
|_ODE_|_One-Ste_|_p Generation(  _|_Euler solver, N_|_=1)_|
|**1-Rectified Flow (****_+Distill_)**|1|1.13 (**_9.08_**)|378 (_6.18_)|0.0 (_0.45_)|
|**2-Rectified Flow (****_+Distill_)**|1|8.08 (_9.01_)|12.21 (**_4.85_**)|0.34 (_0.50_)|
|**3-Rectified Flow (****_+Distill_)**|1|8.47 (_8.79_)|8.15 (_5.21_)|0.41 (**_0.51_**)|
|VP ODE [73] (_+Distill_)|1|1.20 (_8.73_)|451 (_16.23_)|0.0 (_0.29_)|
|sub-VP ODE[73] (_+Distill_)|1|1.21(_8.80_)|451(_14.32_)|0.0(_0.35_)|
|_ODE_|_Full Sim_|_ulation(Runge _|_–Kutta(RK45), _|_Adaptive N)_|
|**1-Rectified Flow**|127|**9.60**|**2.58**|**0.57**|
|**2-Rectified Flow**|110|9.24|3.36|0.54|
|**3-Rectified Flow**|104|9.01|3.96|0.53|
|VP ODE [73]|140|9.37|3.93|0.51|
|sub-VP ODE[73]|146|9.46|3.16|0.55|
|_SDE_|_Full Sim_|_ulation(Euler_|_solver, N=200_|_0)_|
|VP SDE [73]|2000|9.58|2.55|0.58|
|sub-VP SDE[73]|2000|9.56|2.61|0.58|


(a) Results using the DDPM++ architecture.

|Method|NFE(_↓_)|IS(_↑_)|FID(_↓_)|Recall(_↑_)|
|---|---|---|---|---|
|_GAN_|_One-Step_|_Generation_|||
|SNGAN [52]|1|8.22|21.7|0.44|
|StyleGAN2 [28]|1|9.18|8.32|0.41|
|StyleGAN-XL [66]|1|-|1.85|0.47|
|StyleGAN2 + ADA [28]|1|9.40|2.92|0.49|
|StyleGAN2 + DiffAug [98]|1|9.40|5.79|0.42|
|TransGAN + DiffAug [26]|1|9.02|9.26|0.41|
|_GAN with U-Net_|_One-step _|_Generation_|||
|TDPM (T=1) [99]|1|8.65|8.91|0.46|
|DenoisingDiffusion GAN(T=1) [91]|1|8.93|14.6|0.19|
|_ODE_|_One Step_|_Generation(  _|_Euler solver,_|_N=1)_|
|DDIM Distillation [47]|1|8.36|9.36|0.51|
|NCSN++(VE ODE) [73] (_+Distill_)|1|1.18(_2.57_)|461(_254_)|0.0(_0.0_)|
|_ODE_|_Full Sim_|_ulation(Runge _|_–Kutta(RK4_|_5), Adaptive N)_|
|NCSN++(VE ODE) [73]|176|9.35|5.38|0.56|
|_SDE_|_Full Sim_|_ulation(Euler_|_solver)_||
|DDPM [23]|1000|9.46|3.21|0.57|
|NCSN++(VE SDE) [73]|2000|9.83|2.38|0.59|


(b) Recent results with different architectures reported in literature.

Table 1: Results on CIFAR10 unconditioned image generation. Fr´echet Inception Distance (FID) and Inception Score (IS) measure the quality of the generated images, and recall score [38] measures diversity. The number of function evaluation (NFE) denotes the number of times we need to call the main neural network during inference. It coincides with the number of discretization steps _N_ for ODE and SDE models.


sub-VP ODE VP ODE VE ODE sub-VP SDE VP SDE 2-Rectified Flow 3-Rectified Flow N = 1 N = 2 N = 3<br>1-Rectified Flow 2-Rectified Flow 3-Rectified Flow 1-Distilled 2-Distilled 3-Distilled Reflow Reflow Reflow Reflow<br>(a) FID and Recall vs. Number of Euler discretization steps  N (b) FID and Recall vs. Training Iterations<br>

Figure 8: (a) Results of rectified flows and (sub-)VP ODE on CIFAR10 with different number _N_ of Euler discretization steps. (b) The FID and recall during different reflow and training steps. In (a), _k-Distilled_ refers to the one-step model distilled from _k-Rectified Flow_ for _k_ = 1 _,_ 2 _,_ 3.

and GAN methods. Using the same RK45 ODE solver, rectified flows require fewer steps to generate the images compared with VE, VP, sub-VP ODEs. The results are comparable to the fully simulated (sub-)VP SDE, which yields simulation cost.

_• Results on few and single step generation._ As shown in Figure 8, the reflow procedure substantially improves both FID and recall in the small step regime (e.g., _N_ ⪅80), even though it worsens the results in the large step regime due to the accumulation of error on estimating _v_<sup>_x_</sup> . Figure 8 (b) show that each reflow leads to a noticeable improvement in FID and recall. For one-step generation ( _N_ = 1), the results are further boosted by distillation (see the stars in Figure 8 (a)). Overall, the distilled _k_ -Rectified Flow with _k_ = 1 _,_ 2 _,_ 3 yield one-step generative models beating all previous ODEs with distillation; they also beat the reported results of one-step


Reflow Reflow 1-Rectified Flow 2-Rectified Flow<br>1-Rectified Flow<br>2-Rectified Flow<br>3-Rectified Flow<br>Pixel  Value<br>

Figure 9: The straightening effect on CIFAR10. Left: the straightness measure on different reflow steps and training iterations. Right: trajectories of randomly sampled pixels following 1- and 2-rectified flow.


models with similar U-net type architectures trained using GANs (see the _GAN with U-Net_ in Table 1 (b)).

In particular, the distilled 2-rectified flow achieves an FID of 4 _._ 85, beating the best known one-step generative model with U-net architecture, 8 _._ 91 (TDPM, Table 1 (b)). The recalls of both 2-rectified flow (0 _._ 50) and 3-rectified flow (0 _._ 51) outperform the best known results of GANs (0 _._ 49 from StyleGAN2+ADA) showing an advantage in diversity. We should note that the reported results of GANs have been carefully optimized with special techniques such as adaptive discriminator augmentation (ADA) [28], while our results are based on the vanilla implementation of rectified flow. It is likely to further improve rectified flow with proper data augmentation techniques, or the combination of GANs such as those proposed by TDPM [99] and denoising diffusion GAN [91].

_• Reflow straightens the flow._ Figure 9 shows the reflow procedure decreases improves the straightness of the flow on CIFAR10. In Figure 10 visualizes the trajectories of 1-rectified flow and 2-rectified flow on the AFHQ cat dataset: at each point _zt_ , we extrapolate the terminal value at _t_ = 1 by _z_ ˆ1<sup>_t_=</sup><sup>_zt_+ (1</sup><sup>_−t_)</sup><sup>_v_(</sup><sup>_zt, t_);</sup> if the trajectory of ODE follows a straight line, _z_ ˆ1<sup>_t_shouldnotchangeaswevary</sup><sup>_t_whenfollowingthe</sup> same path. We observe that _z_ ˆ1<sup>_t_is almost independent with</sup><sup>_t_for 2-rectified flow, showing the path is almost</sup> straight. Moreover, even though 1-rectified flow is not straight with _z_ ˆ1<sup>_t_over time, it still yields recognizable</sup> and clear images very early ( _t ≈_ 0 _._ 1). In comparison, it is need _t ≈_ 0 _._ 6 to get a clear image from the extrapolation of sub-VP ODE.

**High-resolution image generation** Figure 11 shows the result of 1-rectified flow on image generation on high-resolution (256 _×_ 256) datasets, including LSUN Bedroom [93], LSUN Church [93], CelebA HQ [27] to AFHQ Cat [9]. We can see that it can generate high quality results across the different datasets. Figure 1 & 10 show that 1-(2-)rectified flow yields good results within one or few Euler steps.

Figure 12 shows a simple example of image editing using 1-rectified flow: We first obtain an unnatural image _z_ 1 by stitching the upper and lower parts of two natural images, and then run 1-rectified flow backwards to get a latent code _z_ 0. We then modify _z_ 0 to increase its likelihood under _π_ 0 (which is _N_ (0 _, I_ )) to get more naturally looking variants of the stitched image.

## 5.3 Image-to-Image Translation

Assume we are given two sets of images of different styles (a.k.a. domains), whose distributions are denoted by _π_ 0 _, π_ 1, respectively. We are interested in transferring the style (or other key characteristics) of the images in one domain to the other domain, in the absence of paired examples. A classical approach to achieving this is cycle-consistent adversarial networks (a.k.a. CycleGAN) [100, 25], which jointly learns a forward and backward mapping _F, G_ by minimizing the sum of adversarial losses on the two domains, regularized by a cycle consistency loss to enforce _F_ ( _G_ ( _x_ )) _≈ x_ for all image _x_ .

By constructing the rectified flow of _π_ 0 and _π_ 1, we obtain a simple approach to image translation that requires no adversarial optimization and cycle-consistency regularization: training the rectified flow requires a simple optimization procedure and the cycle consistency is automatically in flow models satisfied due to reversibility of ODEs.

As the main goal here is to obtain good visual results, we are not interested in faithfully transferring _X_ 0 _∼ π_ 0 to an _X_ 1 that exactly follows _π_ 1. Rather, we are interested in transferring the image styles while preserving the identity of the main object in the image. For example, when transferring a human face image to a cat


Figure 10: Sample trajectories _zt_ of different flows on the AFHQ Cat dataset, and the extrapolation _z_ ˆ1<sup>_t_=</sup><sup>_zt_+ (1</sup><sup>_−_</sup> _t_ ) _v_ ( _zt, t_ ) from different _zt_ . The same random seed is adopted for all three methods. The _z_ ˆ1<sup>_t_of2-rectifiedflowis</sup> almost independent with _t_ , indicating that its trajectory is almost straight.


Figure 11: Examples of 256 _×_ 256 images generated by 1-rectified flow.


Figure 12: An example of image editing using 1-rectified flow. Here, we stitch the images of a white cat and a black cat into an unnatural image (denoted as _z_ 1). We simulate the ODE reversely from _z_ 1 to get the latent code _z_ 0. Because _z_ 1 is not a natural image, _z_ 0 should have low likelihood under _π_ 0 = _N_ (0 _, I_ ). Hence, we move _z_ 0 towards the high probability region of _π_ 0 to get _z_ 0<sup>_′_andsolvetheODEforwardlytogetamorerealisticallylookingimage</sup> _z_ 1<sup>_′_.The modification can be done deterministically by improving the</sup><sup>_π_0-likelihood via</sup><sup>_z_</sup> 0<sup>_′_=</sup><sup>_αz_0with</sup><sup>_α∈_(0</sup><sup>_,_1), or</sup> stochastically by Langevin dynamics, which yields a formula of _z_ 0<sup>_′_=</sup><sup>_αz_0+</sup> _√_ 1 _− α_<sup>2</sup> _ξ_ with _ξ ∼N_ (0 _, I_ ).

face, we are interested in getting a unrealistic face of human-cat hybrid that still “looks like” the original human face.

To achieve this, let _h_ ( _x_ ) be a feature mapping of image _x_ representing the styles that we are interested in transferring. Let _Xt_ = _tX_ 1 + (1 _− t_ ) _X_ 0. Then _Ht_ = _h_ ( _Xt_ ) follows an ODE of d _Ht_ = _∇h_ ( _Xt_ )<sup>_⊤_</sup> ( _X_ 1 _− X_ 0)d _t._ Hence, to ensure that the style is transferred correctly, we propose to learn _v_ such that _Ht_<sup>_′_=</sup><sup>_h_(</sup><sup>_Zt_)</sup> with d _Zt_ = _v_ ( _Zt, t_ )d _t_ approximates _Ht_ as much as possible. Because d _Ht_<sup>_′_=</sup><sup>_∇h_(</sup><sup>_Zt_)</sup><sup>_⊤v_(</sup><sup>_Zt, t_)d</sup><sup>_t_,we</sup> propose to minimize the following loss:


In practice, we set _h_ ( _x_ ) to be latent representation of a classifier trained to distinguish the images from the two domains _π_ 0 _, π_ 1, fine-tuned from a pre-trained ImageNet [77] model. Intuitively, _∇xh_ ( _x_ ) serves as a saliency score and re-weights coordinates so that the loss in (20) focuses on penalizing the error that causes significant changes on _h_ .

**Experiment settings** We set the domains _π_ 0 _, π_ 1 to be pairs of the AFHQ [9], MetFace [28] and CelebAHQ [27] dataset. For each dataset, we randomly select 80% as the training data and regard the rest as the test data; and the results are shown by initializing the trained flows from the test data. We resize the image to 512 _×_ 512. The training and network configurations generally follow the experiment settings in Section 5.2. See the appendix for detailed descriptions.

**Results** Figure 1, 13, 14, 15 show examples of results of 1- and 2-rectified flow simulated with Euler method with different number of steps _N_ . We can see that rectified flows can successfully transfer the styles


(A) Cat  → Wild Animals (B) Wild Animals  → Cat<br>


(C) MetFace  → CelebA Face (D) CelebA Face  → MetFace<br>

Figure 13: Samples of 1-rectified flow simulated with _N_ = 100 Euler steps between different domains.


Initialization 1-Rectified Flow 2-Rectified Flow 1-Rectified Flow 2-Rectified Flow<br>N = 100 N = 100 N = 1 N = 1<br>


Figure 14: Samples of results of 1- and 2-rectified flow simulated with _N_ = 1 and _N_ = 100 Euler steps.

and generate high quality images. For example, when transferring cats to wild animals, we can generate diverse images with different animal faces, e.g., fox, lion, tiger and cheetah. Moreover, with one step of reflow, 2-rectified flow returns good results with a single Euler step ( _N_ = 1). See more examples in Appendix.

## 5.4 Domain Adaptation

A key challenge of applying machine learning to real-world problems is the domain shift between the training and test datasets: the performance of machine learning models may degrade significantly when tested on a novel domain different from the training set. Rectified flow can be applied to transfer the novel domain ( _π_ 0) to the training domain ( _π_ 1) to mitigate the impact of domain shift.

**Experiment settings** We test the rectified flow for domain adaptation on a number of datasets. DomainNet [58] is a dataset of common objects in six different domain taken from DomainBed [20]. All domains from DomainNet include 345 categories (classes) of objects such as Bracelet, plane, bird and cello. Office-Home [83] is a benchmark dataset for domain adaptation which contains 4 domains where each domain consists of 65 categories. To apply our method, first we map both the training and testing data to the latent representation from final hidden layer of the pre-trained model, and construct the rectified flow on the latent representation.


1-Rectified Flow<br>2-Rectified Flow<br>1-Rectified Flow<br>2-Rectified Flow<br>


(a) 1-rectified flow between different domains (b) 1- and 2-rectified flow for MetFace _→_ Cat.

Figure 15: (a) Samples of trajectories _zt_ of 1- and 2-rectified flow for transferring between different domains.

|Method|ERM|IRM|ARM|Mixup|MLDG|CORAL|Ours|
|---|---|---|---|---|---|---|---|
|OfficeHome|66_._5_±_0_._3|64_._3_±_2_._2|64_._8_±_0_._3|68_._1_±_0_._3|66_._8_±_0_._6|68_._7_±_0_._3|**69**_._**2**_±_**0**_._**5**|
|DomainNet|40_._9_±_0_._1|33_._9_±_2_._8|35_._5_±_0_._2|39_._2_±_0_._1|41_._2_±_0_._1|**41**_._**5**_±_**0**_._**2**|**41**_._**4**_±_**0**_._**1**|


Table 2: The accuracy of the transferred testing data using different methods, on the OfficeHome and DomainNet dataset. Higher accuracy means the better performance.

We use the same DDPM++ model architecture for training. For inference, we set the number of steps of our flow model as 100 using uniform discretization. The methods are evaluated by the prediction accuracy of the transferred testing data on the classification model trained on the training data.

**Results** As demonstrated in Table 2, the 1-rectified flow shows state-of-the-art performance on both DomainNet and OfficeHome. It is better or on par with the previous best approach (Deep CORAL [76]), while sustainably improve over all other methods.


# References

- [1] Luigi Ambrosio and Gianluca Crippa. Existence, uniqueness, stability and differentiability properties of the flow associated to weakly differentiable vector fields. In _Transport equations and multi-D hyperbolic conservation laws_ , pages 3–57. Springer, 2008.

- [2] Luigi Ambrosio, Elia Bru´e, and Daniele Semola. _Lectures on optimal transport_ . Springer, 2021.

- [3] Brian DO Anderson. Reverse-time diffusion equation models. _Stochastic Processes and their Applications_ , 12(3):313–326, 1982.

- [4] Martin Arjovsky, Soumith Chintala, and L´eon Bottou. Wasserstein generative adversarial networks. In _International conference on machine learning_ , pages 214–223. PMLR, 2017.

- [5] Fan Bao, Chongxuan Li, Jun Zhu, and Bo Zhang. Analytic-DPM: an analytic estimate of the optimal reverse variance in diffusion probabilistic models. _arXiv preprint arXiv:2201.06503_ , 2022.

- [6] Ricky TQ Chen, Yulia Rubanova, Jesse Bettencourt, and David K Duvenaud. Neural ordinary differential equations. _Advances in neural information processing systems_ , 31, 2018.

- [7] Tianrong Chen, Guan-Horng Liu, and Evangelos A Theodorou. Likelihood training of Schr¨odinger bridge using forward-backward sdes theory. _arXiv preprint arXiv:2110.11291_ , 2021.

- [8] Jooyoung Choi, Sungwon Kim, Yonghyun Jeong, Youngjune Gwon, and Sungroh Yoon. Ilvr: Conditioning method for denoising diffusion probabilistic models. _arXiv preprint arXiv:2108.02938_ , 2021.

- [9] Yunjey Choi, Youngjung Uh, Jaejun Yoo, and Jung-Woo Ha. StarGAN v2: Diverse image synthesis for multiple domains. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 8188–8197, 2020.

- [10] Max Daniels, Tyler Maunu, and Paul Hand. Score-based generative neural networks for large-scale optimal transport. _Advances in neural information processing systems_ , 34:12955–12965, 2021.

- [11] Valentin De Bortoli, James Thornton, Jeremy Heng, and Arnaud Doucet. Diffusion Schr¨odinger bridge with applications to score-based generative modeling. _Advances in Neural Information Processing Systems_ , 34, 2021.

- [12] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat GANs on image synthesis. _Advances in Neural Information Processing Systems_ , 34, 2021.

- [13] Laurent Dinh, David Krueger, and Yoshua Bengio. Nice: Non-linear independent components estimation. _arXiv preprint arXiv:1410.8516_ , 2014.

- [14] Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density estimation using real nvp. _arXiv preprint arXiv:1605.08803_ , 2016.

- [15] Alessio Figalli and Federico Glaudo. _An Invitation to Optimal Transport, Wasserstein Distances, and Gradient Flows_ . 2021.

- [16] R Flamary, N Courty, D Tuia, and A Rakotomamonjy. Optimal transport for domain adaptation. _IEEE Trans. Pattern Anal. Mach. Intell_ , 1, 2016.


- [17] Hans F¨ollmer. An entropy approach to the time reversal of diffusion processes. In _Stochastic Differential Systems Filtering and Control_ , pages 156–163. Springer, 1985.

- [18] Giulio Franzese, Simone Rossi, Lixuan Yang, Alessandro Finamore, Dario Rossi, Maurizio Filippone, and Pietro Michiardi. How much is enough? a study on diffusion times in score-based generative models. _arXiv preprint arXiv:2206.05173_ , 2022.

- [19] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. _Advances in neural information processing systems_ , 27, 2014.

- [20] Ishaan Gulrajani and David Lopez-Paz. In search of lost domain generalization. _arXiv preprint arXiv:2007.01434_ , 2020.

- [21] William Harvey, Saeid Naderiparizi, Vaden Masrani, Christian Weilbach, and Frank Wood. Flexible diffusion modeling of long videos. _arXiv preprint arXiv:2205.11495_ , 2022.

- [22] Ulrich G Haussmann and Etienne Pardoux. Time reversal of diffusions. _The Annals of Probability_ , pages 1188–1205, 1986.

- [23] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. _Advances in Neural Information Processing Systems_ , 33:6840–6851, 2020.

- [24] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. _arXiv preprint arXiv:2204.03458_ , 2022.

- [25] Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A Efros. Image-to-image translation with conditional adversarial networks. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 1125–1134, 2017.

- [26] Yifan Jiang, Shiyu Chang, and Zhangyang Wang. TransGAN: Two pure transformers can make one strong GAN, and that can scale up. _Advances in Neural Information Processing Systems_ , 34, 2021.

- [27] Tero Karras, Timo Aila, Samuli Laine, and Jaakko Lehtinen. Progressive growing of GANs for improved quality, stability, and variation. In _International Conference on Learning Representations_ , 2018.

- [28] Tero Karras, Miika Aittala, Janne Hellsten, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Training generative adversarial networks with limited data. _Advances in Neural Information Processing Systems_ , 33:12104–12114, 2020.

- [29] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusionbased generative models. _arXiv preprint arXiv:2206.00364_ , 2022.

- [30] Valentin Khrulkov and Ivan Oseledets. Understanding DDPM latent codes through optimal transport. _arXiv preprint arXiv:2202.07477_ , 2022.

- [31] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. _arXiv preprint arXiv:1412.6980_ , 2014.


- [32] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. _arXiv preprint arXiv:1312.6114_ , 2013.

- [33] Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao, and Bryan Catanzaro. Diffwave: A versatile diffusion model for audio synthesis. In _International Conference on Learning Representations_ , 2020.

- [34] Alexander Korotin, Lingxiao Li, Aude Genevay, Justin M Solomon, Alexander Filippov, and Evgeny Burnaev. Do neural optimal transport solvers work? a continuous wasserstein-2 benchmark. _Advances in Neural Information Processing Systems_ , 34:14593–14605, 2021.

- [35] Alexander Korotin, Daniil Selikhanovych, and Evgeny Burnaev. Neural optimal transport. _arXiv preprint arXiv:2201.12220_ , 2022.

- [36] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.

- [37] Thomas G Kurtz. Equivalence of stochastic equations and martingale problems. In _Stochastic analysis 2010_ , pages 113–130. Springer, 2011.

- [38] Tuomas Kynk¨a¨anniemi, Tero Karras, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Improved precision and recall metric for assessing generative models. _Advances in Neural Information Processing Systems_ , 32, 2019.

- [39] Hugo Lavenant and Filippo Santambrogio. The flow map of the fokker–planck equation does not provide optimal transport. _Applied Mathematics Letters_ , page 108225, 2022.

- [40] Junhyeok Lee and Seungu Han. Nu-wave: A diffusion probabilistic model for neural audio upsampling. _arXiv preprint arXiv:2104.02321_ , 2021.

- [41] Xiang Lisa Li, John Thickstun, Ishaan Gulrajani, Percy Liang, and Tatsunori B Hashimoto. Diffusionlm improves controllable text generation. _arXiv preprint arXiv:2205.14217_ , 2022.

- [42] Qiang Liu. On rectified flow and optimal coupling. _preprint_ , 2022.

- [43] Xingchao Liu, Chengyue Gong, Lemeng Wu, Shujian Zhang, Hao Su, and Qiang Liu. Fusedream: Training-free text-to-image generation with improved clip+ gan space optimization. _arXiv preprint arXiv:2112.01573_ , 2021.

- [44] Xingchao Liu, Lemeng Wu, Mao Ye, and Qiang Liu. Let us build bridges: Understanding and extending diffusion generative models. _arXiv preprint arXiv:2208.14699_ , 2022.

- [45] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. _arXiv preprint arXiv:1711.05101_ , 2017.

- [46] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. DPM-solver: A fast ODE solver for diffusion probabilistic model sampling in around 10 steps. _arXiv preprint arXiv:2206.00927_ , 2022.

- [47] Eric Luhman and Troy Luhman. Knowledge distillation in iterative generative models for improved sampling speed. _arXiv preprint arXiv:2101.02388_ , 2021.

- [48] Zhaoyang Lyu, Xudong Xu, Ceyuan Yang, Dahua Lin, and Bo Dai. Accelerating diffusion models via early stop of the diffusion process. _arXiv preprint arXiv:2205.12524_ , 2022.


- [49] Ashok Makkuva, Amirhossein Taghvaei, Sewoong Oh, and Jason Lee. Optimal transport mapping via input convex neural networks. In _International Conference on Machine Learning_ , pages 6672–6681. PMLR, 2020.

- [50] Chenlin Meng, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Image synthesis and editing with stochastic differential equations. _arXiv preprint arXiv:2108.01073_ , 2021.

- [51] Gautam Mittal, Jesse Engel, Curtis Hawthorne, and Ian Simon. Symbolic music generation with diffusion models. _arXiv preprint arXiv:2103.16091_ , 2021.

- [52] Takeru Miyato, Toshiki Kataoka, Masanori Koyama, and Yuichi Yoshida. Spectral normalization for generative adversarial networks. In _International Conference on Learning Representations_ , 2018.

- [53] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. _arXiv preprint arXiv:2112.10741_ , 2021.

- [54] Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In _International Conference on Machine Learning_ , pages 8162–8171. PMLR, 2021.

- [55] Derek Onken, Samy Wu Fung, Xingjian Li, and Lars Ruthotto. Ot-flow: Fast and accurate continuous normalizing flows via optimal transport. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 35, pages 9223–9232, 2021.

- [56] George Papamakarios, Eric T Nalisnick, Danilo Jimenez Rezende, Shakir Mohamed, and Balaji Lakshminarayanan. Normalizing flows for probabilistic modeling and inference. _J. Mach. Learn. Res._ , 22(57):1–64, 2021.

- [57] Stefano Peluchetti. Non-denoising forward-time diffusions. 2021.

- [58] Xingchao Peng, Qinxun Bai, Xide Xia, Zijun Huang, Kate Saenko, and Bo Wang. Moment matching for multi-source domain adaptation. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 1406–1415, 2019.

- [59] Gabriel Peyr´e, Marco Cuturi, et al. Computational optimal transport: With applications to data science. _Foundations and Trends® in Machine Learning_ , 11(5-6):355–607, 2019.

- [60] Vadim Popov, Ivan Vovk, Vladimir Gogoryan, Tasnima Sadekova, and Mikhail Kudinov. Grad-tts: A diffusion probabilistic model for text-to-speech. In _International Conference on Machine Learning_ , pages 8599–8608. PMLR, 2021.

- [61] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical textconditional image generation with clip latents. _arXiv preprint arXiv:2204.06125_ , 2022.

- [62] Danilo Rezende and Shakir Mohamed. Variational inference with normalizing flows. In _International conference on machine learning_ , pages 1530–1538. PMLR, 2015.

- [63] Litu Rout, Alexander Korotin, and Evgeny Burnaev. Generative modeling with optimal transport maps. _arXiv preprint arXiv:2110.02999_ , 2021.


- [64] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S Sara Mahdavi, Rapha Gontijo Lopes, et al. Photorealistic text-to-image diffusion models with deep language understanding. _arXiv preprint arXiv:2205.11487_ , 2022.

- [65] Filippo Santambrogio. Optimal transport for applied mathematicians. _Birk¨auser, NY_ , 55(58-63):94, 2015.

- [66] Axel Sauer, Katja Schwarz, and Andreas Geiger. StyleGAN-XL: Scaling StyleGAN to large diverse datasets. In _Special Interest Group on Computer Graphics and Interactive Techniques Conference Proceedings_ , pages 1–10, 2022.

- [67] Vivien Seguy, Bharath Bhushan Damodaran, R´emi Flamary, Nicolas Courty, Antoine Rolet, and Mathieu Blondel. Large-scale optimal transport and mapping estimation. _arXiv preprint arXiv:1711.02283_ , 2017.

- [68] Abhishek Sinha, Jiaming Song, Chenlin Meng, and Stefano Ermon. D2C: Diffusion-decoding models for few-shot conditional generation. _Advances in Neural Information Processing Systems_ , 34:12533– 12548, 2021.

- [69] Leslie N Smith and Nicholay Topin. Super-convergence: Very fast training of neural networks using large learning rates. In _Artificial intelligence and machine learning for multi-domain operations applications_ , volume 11006, pages 369–386. SPIE, 2019.

- [70] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In _International Conference on Learning Representations_ , 2020.

- [71] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. _Advances in Neural Information Processing Systems_ , 32, 2019.

- [72] Yang Song and Stefano Ermon. Improved techniques for training score-based generative models. _Advances in neural information processing systems_ , 33:12438–12448, 2020.

- [73] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In _International Conference on Learning Representations_ , 2020.

- [74] Yang Song, Conor Durkan, Iain Murray, and Stefano Ermon. Maximum likelihood training of scorebased diffusion models. _Advances in Neural Information Processing Systems_ , 34, 2021.

- [75] Xuan Su, Jiaming Song, Chenlin Meng, and Stefano Ermon. Dual diffusion implicit bridges for image-to-image translation. _arXiv preprint arXiv:2203.08382_ , 2022.

- [76] Baochen Sun and Kate Saenko. Deep coral: Correlation alignment for deep domain adaptation. In _European conference on computer vision_ , pages 443–450. Springer, 2016.

- [77] Mingxing Tan and Quoc Le. Efficientnet: Rethinking model scaling for convolutional neural networks. In _International conference on machine learning_ , pages 6105–6114. PMLR, 2019.

- [78] Anastasiya Tanana. Comparison of transport map generated by heat flow interpolation and the optimal transport brenier map. _Communications in Contemporary Mathematics_ , 23(06):2050025, 2021.


- [79] Giulio Trigila and Esteban G Tabak. Data-driven optimal transport. _Communications on Pure and Applied Mathematics_ , 69(4):613–648, 2016.

- [80] Belinda Tzen and Maxim Raginsky. Theoretical guarantees for sampling and inference in generative models with latent diffusions. In _Conference on Learning Theory_ , pages 3084–3114. PMLR, 2019.

- [81] Arash Vahdat, Karsten Kreis, and Jan Kautz. Score-based generative modeling in latent space. _Advances in Neural Information Processing Systems_ , 34:11287–11302, 2021.

- [82] Francisco Vargas, Pierre Thodoroff, Austen Lamacraft, and Neil Lawrence. Solving Schr¨odinger bridges via maximum likelihood. _Entropy_ , 23(9):1134, 2021.

- [83] Hemanth Venkateswara, Jose Eusebio, Shayok Chakraborty, and Sethuraman Panchanathan. Deep hashing network for unsupervised domain adaptation. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 5018–5027, 2017.

- [84] C´edric Villani. _Optimal transport: old and new_ , volume 338. Springer, 2009.

- [85] C´edric Villani. _Topics in optimal transportation_ , volume 58. American Mathematical Soc., 2021.

- [86] Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, St´efan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey,<sup>˙</sup> Ilhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antˆonio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, and SciPy 1.0 Contributors. SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. _Nature Methods_ , 17:261–272, 2020. doi: 10.1038/s41592-019-0686-2.

- [87] Gefei Wang, Yuling Jiao, Qian Xu, Yang Wang, and Can Yang. Deep generative learning via Schr¨odinger bridge. In _International Conference on Machine Learning_ , pages 10794–10804. PMLR, 2021.

- [88] Rose E Wang, Esin Durmus, Noah Goodman, and Tatsunori Hashimoto. Language modeling via stochastic processes. _arXiv preprint arXiv:2203.11370_ , 2022.

- [89] Antoine Wehenkel and Gilles Louppe. Diffusion priors in variational autoencoders. _arXiv preprint arXiv:2106.15671_ , 2021.

- [90] Lemeng Wu, Chengyue Gong, Xingchao Liu, Mao Ye, and Qiang Liu. Diffusion-based molecule generation with informative prior bridges. _arXiv preprint_ , 2022.

- [91] Zhisheng Xiao, Karsten Kreis, and Arash Vahdat. Tackling the generative learning trilemma with denoising diffusion GANs. _arXiv preprint arXiv:2112.07804_ , 2021.

- [92] Ruihan Yang, Prakhar Srivastava, and Stephan Mandt. Diffusion probabilistic modeling for video generation. _arXiv preprint arXiv:2203.09481_ , 2022.

- [93] Fisher Yu, Ari Seff, Yinda Zhang, Shuran Song, Thomas Funkhouser, and Jianxiong Xiao. Lsun: Construction of a large-scale image dataset using deep learning with humans in the loop. _arXiv preprint arXiv:1506.03365_ , 2015.


- [94] Qinsheng Zhang and Yongxin Chen. Fast sampling of diffusion models with exponential integrator. _arXiv preprint arXiv:2204.13902_ , 2022.

- [95] Qinsheng Zhang, Molei Tao, and Yongxin Chen. gDDIM: Generalized denoising diffusion implicit models. _arXiv preprint arXiv:2206.05564_ , 2022.

- [96] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 586–595, 2018.

- [97] Min Zhao, Fan Bao, Chongxuan Li, and Jun Zhu. EGSDE: Unpaired image-to-image translation via energy-guided stochastic differential equations. _arXiv preprint arXiv:2207.06635_ , 2022.

- [98] Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han. Differentiable augmentation for data-efficient GAN training. _Advances in Neural Information Processing Systems_ , 33:7559–7570, 2020.

- [99] Huangjie Zheng, Pengcheng He, Weizhu Chen, and Mingyuan Zhou. Truncated diffusion probabilistic models. _arXiv preprint arXiv:2202.09671_ , 2022.

- [100] Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A Efros. Unpaired image-to-image translation using cycle-consistent adversarial networks. In _Proceedings of the IEEE international conference on computer vision_ , pages 2223–2232, 2017.


**Algorithm 2** Train(Data)

# Input: Data= _{_ x0, x1 _}_ # Output: Model _v_ ( _x, t_ ) for the rectified flow initialize Model for x0, x1 in Data: # x0, x1: samples from _π_ 0, _π_ 1 Optimizer.zero ~~g~~ rad() t = torch.rand(batchsize) # Randomly sample _t ∈_ [0,1] Loss = ( Model(t*x1+(1-t)*x0, t) - (x1-x0) ).pow(2).mean() Loss.backward() Optimizer.step() <u>return Model</u>

**Algorithm 3** Sample(Model, Data) # Input: Model _v_ ( _x, t_ ) of the rectified flow # Output: draws of the rectified coupling ( _Z_ 0 _, Z_ 1) coupling = [] for x0, in Data: # x0: samples from _π_ 0 (batchsize _×_ dim) x1 = model.ODE ~~s~~ olver(x0) coupling.append((x0, x1)) return coupling

**Algorithm 4** Reflow(Data) # Input: Data= _{_ x0, x1 _}_ # Output: draws of the _K_ -th rectified coupling Coupling = Data for _k_ = 1 _, . . . , K_ : Model = Train(Coupling) Coupling = sample(Model, Data) return Coupling

# A Additional Experiment Details

**Experiment Configuration on CIFAR10** We conduct unconditional image generation with the CIFAR10 dataset [36]. The resolution of the images are set to 32 _×_ 32. For rectified flow, we adopt the same network structure as DDPM++ in [73]. The training of the network is smoothed by exponential moving average as in [73], with a ratio of 0 _._ 999999. We adopt Adam [31] optimizer with a learning rate of 2 _e −_ 4 and a dropout rate of 0 _._ 15.

For reflow, we first generate 4 million pairs of ( _z_ 0 _, z_ 1) to get a new dataset _D_ , then fine-tune the _i_ -rectified flow model for 300 _,_ 000 steps to get the ( _i_ + 1)-rectified flow model. We further distill these rectified flow models for few-step generation. To get a _k_ -step image generator from the _i_ -rectified flow, we randomly sample _t ∈{_ 0 _,_ 1 _/k, · · · ,_ ( _k −_ 1) _/k}_ during fine-tuning, instead of randomly sampling _t ∈_ [0 _,_ 1]. Specifically, for _k_ = 1, we replace the L2 loss function with the LPIPS similarity [96] since it empirically brings better performance.


Figure 16: Few-step generation with different ODEs. Compared with VE,VP,sub-VP ODE, 1-rectified flow can generate blurry images using only 1,2,3 steps. After one time of rectification, 2-rectified flow can generate clear images with 1,2,3 steps.

**Expreiment Configuration on Image-to-Image Translation** In this experiment, we also adopt the same U-Net structure of DDPM++ [73] for representing the drift _v_<sup>_X_</sup> . We follow the procedure in Algorithm 1. For the purpose of generative modeling, we set _π_ 0 to be one domain dataset and _π_ 1 the other domain dataset. For optimization, we use AdamW [45] optimizer with _β_ (0 _._ 9 _,_ 0 _._ 999), weight decay 0 _._ 1 and dropout rate 0 _._ 1. We train the model with a batch size of 4 for 1 _,_ 000 epochs. We further apply exponential moving average (EMA) optimizer with coefficient 0 _._ 9999. We perform grid-search on the learning rate from _{_ 5 _×_ 10<sup>_−_4</sup> _,_ 2 _×_ 10<sup>_−_4</sup> _,_ 5 _×_ 10<sup>_−_5</sup> _,_ 2 _×_ 10<sup>_−_5</sup> _,_ 5 _×_ 10<sup>_−_6</sup> _}_ and pick the model with the lowest training loss.

We use the AFHQ [9], MetFace [28] and CelebA-HQ [27] dataset. Animal Faces HQ (AFHQ) is an animalface dataset consisting of 15,000 high-quality images at 512 _×_ 512 resolution. The dataset includes three domains of cat, dog, and wild animals, each providing 5000 images. MetFace consists of 1,336 high-quality PNG human-face images at 1024 _×_ 1024 resolution, extracted from works of art. CelebA-HQ is a humanface dataset which consists of 30,000 images at 1024 _×_ 1024 resolution. We randomly select 80% as the training data and regard the rest as the test data, and resize the image to 512 _×_ 512.

**Experiment Configuration on Domain Adaptation** For training the model, we apply AdamW [45] optimizer with batch size 16, number of iterations 50k, learning rate 10<sup>_−_4</sup> , weight decay 0 _._ 1 and OneCycle [69] learning rate schedule.


Figure 17: To visualize the latent space, we randomly sample _z_ 0 and _z_ 1 from _N_ (0 _, I_ ), and show the generated images of<sup>_√_</sup> _<u>αz</u>_ 0 +<sup>_√_</sup> 1 _− αz_ 1 for _α ∈_ [0 _,_ 1].


Figure 18: (a) We compare the latent space between Rectified Flow (0) and (1) using different sampling strategies with the same random seeds. We observe that (i) both 1-Rectified Flow and 2-Rectified Flow can provide a smooth latent interpolation, and their latent spaces look similar; (ii) when using one-step sampling ( _N_ = 1), 2-Rectified Flow can still provide visually recognizable interpolation, while 1-Rectified Flow cannot; (iii) Distilled one-step models can also continuously interpolate between the images, and their latent spaces have little difference with the original flow. (b) We composite the latent codes of two images by replacing the boundary of a black cat with a white cat, then visualize the variation along the trajectory. The black cat turns into a grey cat at first, then a cat with mixing colors, and finally a white cat. (c) We randomly sample _ξ ∼N_ (0 _, I_ ), then generate images with _αξ_ to examine the influence of _α_ on the generated images. We find _α <_ 1 results in overly smooth images, while _α >_ 1 leads to noisy images.


1-Rectified Flow 2-Rectified Flow<br>0 0.0010.002 0.1 0.3 0.5 0.8 1 t 0 0.0010.002 0.1 0.3 0.5 0.8 1 t<br>subVP-ODE<br>0 1 t 0 0.0010.002 0.1 0.3 0.5 0.8 1 t<br>

Figure 19: Sample trajectories _zt_ of different flows on the CIFAR10 dataset, and the extrapolation _z_ ˆ1<sup>_t_=</sup><sup>_zt_+ (1</sup><sup>_−_</sup> _t_ ) _v_ ( _zt, t_ ) from different _zt_ . The same random seed is adopted for all three methods. The _z_ ˆ1<sup>_t_of2-rectifiedflowis</sup> almost independent with _t_ , indicating that its trajectory is almost straight.


Figure 20: We perform latent space embedding / image reconstruction here. Given an image _z_ 1, we use an _reverse ODE solver_ to get a latent code _z_ ˆ0, then use a _forward ODE solver_ to get a reconstruction _z_ ˆ1 of the image. The columns in the figure are _reverse ODE solver (forward ODE solver)_ . (i) Thanks to the‘straightening’ effect, 2-rectified flow can get meaningful latent code with only one reverse step. It can also generate recognizable images using one forward step. (ii) With the help of distilled models, one-step embedding and reconstruction is significantly improved.


Figure 21: More results for image-to-image translation between different domains. The images in each row are timeuniformly sampled from the trajectory of 1-rectified flow solved _N_ = 100 Euler steps with constant step size.
