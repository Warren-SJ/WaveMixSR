# WaveMixSR

## Introduction

This codebase is based on the research paper titled: "WaveMixSR: Resource-efficient Neural Network for Image Super-resolution". 

Traditionally super resolution is based on transformers which are computationally expensive. This paper introduces a new architecture called WaveMixSR which is based on wavelet transforms. This architecture is computationally efficient offers similar performance to traditional transformer based super resolution models.

## Wavelet Transform

Wavelet transform is a mathematical tool that decomposes a signal into different frequency components. The Fourier transform decomposes a signal into its frequency components. However, the Fourier transform does not provide information about the time at which the frequency components occur. Therefore the Fourier transform in suitabe for stationary signals. One possible solution is to use the short-time Fourier transform which provides information about the time at which the frequency components occur. Choosing the window has compromises. A small window results in a high frequency resolution but a low time resolution. A large window results in a high time resolution but a low frequency resolution.

Thw wavelet transform improves on the short-time Fourier transform by using a variable window size. The wavelet transform uses a window that is small at high frequencies and large at low frequencies. This results in a high frequency resolution and a high time resolution. Therefore it is called a multi-resolution analysis.

The continuous wavelet transform is defined as:

$$ F(\tau,s) = \frac{1}{\sqrt{\lvert s \rvert}}\int_{-\infty}^{\infty} f(t)\psi^*(\frac{t-\tau}{s}) dt $$

$s$ is the scale parameter which is $\frac{1}{frequency}$. $\psi$ is known as the wavelet. The wavelet acts as a basis function. Previously in the Fourier transfrom, the basis funtions were complexexponentials. The width and central frequency of the wavelet can be changed. An expanded wavelet (large $s$) is suitable for low frequencies with good frequency resolution but bad time resolution and a compressed wavelet (small $s$) is suitable for high frequencies with good time resolution but bad frequency resolution.

$$ \psi_{s,\tau}(t) = \frac{1}{\sqrt{\lvert s \rvert}}\psi(\frac{t-\tau}{s}) $$
where $\psi$ is the mother wavelet. 

This transform results in a 3D plot of the transform with parameters $s$, $\tau$ and $F(\tau,s)$. This is obtained by plotting for different values of $s$ and $\tau$.

If $s$ and $\tau$ are discrete, the wavelet transform is called the discrete wavelet transform. Analysis is efficient if $s$ and $\tau$ are powers of $2$. The equation of the discrete wavelet transform is given by:

$$ D(a,b) = \frac{1}{\sqrt{b}}\sum_{n} f[t_m]\psi[\frac{t_m-a}{b}] $$

$a$ represents $\tau$ and $b$ represents $s$. Instead of the integral, the sum is used. $a$ and $b$ are dyadic.

For computation, the signal is passed into low pass and high pass signals. 