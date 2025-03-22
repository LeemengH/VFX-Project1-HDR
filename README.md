# VFX-Project1-HDR
NTU VFX 2025 Spring Project1
 
This method has several advantages: first, since the representation is non-parametric, the response function does not need to be assumed to have a shape described by some previously defined class of continuous functions; second, the formulation takes explicitly into account the sensor noise model; and third, the structure of the algorithm naturally uses all information available from image pixels for the estimation of response function and reconstruction of E.
### Problem Description
Our goal is to reconstruct the $g$ (the inverse response curve) and then compute $E_i$  
* $Z_{ij} = f(E_i {\Delta} t_j)$ , where i represents i-th pixel and j represents j-th picture 
* $X_{ij} = E_i {\Delta} t_j$ 
* $g:= ln f^{-1}(Z_{ij})$  , i.e $g: Z \to X$

Except in very specialized situations, the camera response function is not known a priori and must be estimated. To uniquely determine the response function, the 255 values ($g(m), m=0,…,254$ ) must be found.

Robertson et al. exploit the Maximum Likelihood Method to estimate ($g$). More formally,
	
$P(E_i, g | Z_{ij}, \Delta t_j) \propto exp\{ \frac{-1}{2} \sum\limits_{ij} w(Z_{ij})(g(Z_{ij}) - E_i \Delta t_j))^2\}$
where $w(.)$ refer to hat weighting function(Since the z may be overflow(0~255) therefore we need to add some  adjustment i.e fever the number near the middle)

$w(z) = \begin{cases} z - Z_{min} \ \ for\ z \le \frac{1}{2}(Z_{min}+Z_{max})\\ Z_{max} - z \ \ for\ z > \frac{1}{2}(Z_{min}+Z_{max}) \end{cases}$
	
Therefore, to maximize the likelihood we need to minimize the 
	
$\sum\limits_{ij} w(Z_{ij})(g(Z_{ij}) - E_i \Delta t_j))^2$

Since the response function is not yet known, the weighting function (determined by differentiation of the response) is not known either. Rather than jointly attempting to estimate w, g, and E, the weighting function will be fixed _a priori_. (With so many unknowns, the estimation quickly becomes intractable, especially considering the dependence of w on g().)

### Alternating Optimization(2 steps)

Step1: assuming $g(Z_{ij})$ is known, estimate (optimize)for $\widehat{E_i}$
	By first order condition:
		$\widehat{E_i} = \frac{\sum\limits_{j} w(Z_{ij})g(Z_{ij})\Delta t_j}{\sum\limits_{j} w(Z_{ij})\Delta {t_j}^2}$

Step2:  assuming $E_i$ is known, estimate (optimize) for $\widehat{g}(Z_{ij})$
	By first order condition:
		$\widehat{g}(m) = \frac{1}{|\Phi_m|}\sum\limits_{ij\in E_m} E_i \Delta t_j \ \ , where \ \Phi_m = \{(i, j)| Z_{ij} = m\}$
	Note: We need to normalized the result so that
		$g(128) = 1$

### Iterative Estimation
Estimates for the variables of interest at the i-th iteration are denoted as $g^i$ and $E^i$. The initial estimate $g^0$ is chosen as a linear function, with $g^0(128) = 1$. The initial $E^0$  is determined using Step 1, based on the initial linear $g^0$.

This completes one iteration of the algorithm. The process repeats until a convergence criterion is met, defined as the rate of decrease in the objective function falling below a predefined threshold.

### Reference:
https://pages.cs.wisc.edu/~csverma/CS899_09/s00103ed1v01y200711cgr003.pdf
https://www.spiedigitallibrary.org/journals/journal-of-electronic-imaging/volume-12/issue-02/0000/Estimation-theoretic-approach-to-dynamic-range-enhancement-using-multiple-exposures/10.1117/1.1557695.full
