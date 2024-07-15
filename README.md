<p align="center">
  <h2 align="center"> On the connection between least squares, regularization, and classical shadows </h2>
  <p align="center">
    <a style="text-decoration:none" href="https://zhihuizhu.github.io/">
                       Zhihui Zhu</a><sup>1</sup>
    &nbsp;&nbsp;
    <a style="text-decoration:none" href="https://scholar.google.com/citations?user=jl6yjvQAAAAJ&hl=en">
                        Joseph M. Lukens</a><sup>2,3</sup>
    &nbsp;&nbsp;
    <a style="text-decoration:none" href="https://briankirby.github.io/">
                       Brian T. Kirby</a><sup>4,5</sup>
    <br>
    <sup>1</sup>Ohio State University &nbsp;&nbsp;&nbsp; <sup>2</sup>Arizona State University &nbsp;&nbsp;&nbsp; <sup>3</sup>Oak Ridge National Laboratory &nbsp;&nbsp;&nbsp; <sup>4</sup>DEVCOM Army Research Laboratory &nbsp;&nbsp;&nbsp; <sup>5</sup>Tulane University
    </br>
  </p>

</p>

![](./conceptFig.png)


Classical shadows (CS) offer a resource-efficient means to estimate quantum observables, circumventing the need for exhaustive state tomography. 
Here, we clarify and explore the connection between CS techniques and least squares (LS) and regularized least squares (RLS) methods commonly used in machine learning and data analysis. 
By formal identification of LS and RLS ``shadows'' completely analogous to those in CS---namely, point estimators calculated from the empirical frequencies of single measurements---we show that both RLS and CS can be viewed as regularizers for the underdetermined regime, replacing the pseudoinverse with invertible alternatives. 
Through numerical simulations, we evaluate RLS and CS from three distinct angles: the tradeoff in bias and variance, mismatch between the expected and actual measurement distributions, and the interplay between the number of measurements and number of shots per measurement.

Compared to CS, RLS attains lower variance at the expense of bias, is robust to distribution mismatch, and is more sensitive to the number of shots for a fixed number of state copies---differences that can be understood from the distinct approaches taken to regularization. Conceptually, our integration of LS, RLS, and CS under a unifying ``shadow'' umbrella aids in advancing the overall picture of CS techniques, while practically our results highlight the tradeoffs intrinsic to these measurement approaches, illuminating the circumstances under which either RLS or CS would be preferred, such as unverified randomness for the former or unbiased estimation for the latter.

## Citation
If you find our work helpful, please kindly cite our work:
```BibTeX
@article{zhu2023connection,
  title={On the connection between least squares, regularization, and classical shadows},
  author={Zhu, Zhihui and Lukens, Joseph M and Kirby, Brian T},
  journal={arXiv preprint arXiv:2310.16921},
  year={2023}
}
```
