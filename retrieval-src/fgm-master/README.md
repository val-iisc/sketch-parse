Introduction
============

This page contains software and instructions for [factorized graph
matching (FGM)](http://www.f-zhou.com/gm.html) [1] [2].  In addition, we include the following
state-of-the-arts methods as baselines:

- [spectral matching (SM)](https://sites.google.com/site/graphmatchingmethods/) [3],
- [spectral matching with affine constraints (SMAC)](http://www.timotheecour.com/software/graph_matching/graph_matching.html) [4],
- [graduated assignment (GA)](http://www.timotheecour.com/software/graph_matching/graph_matching.html) [5],
- [probabilistic matching (PM)](http://www.cs.huji.ac.il/~zass/gm) [6],
- [integer projected fixed point method (IPFP)](https://sites.google.com/site/graphmatchingmethods/) [7],
- [re-weighted random walk matching (RRWM)](http://cv.snu.ac.kr/research/~RRWM/) [8].

The implementations of the above methods are taken from the authors'
websites (The code of GA was also implemented in the code of SMAC). We
appreciate all the authors for their generosity in sharing codes.


Installation
============

1. unzip `fgm.zip` to your folder;
2. Run `make` to compile all C++ files;
3. Run `addPath` to add sub-directories into the path of Matlab.
4. Run `demoXXX` or `testXXX`.


Instructions
============

The package of `fgm.zip` contains the following files and folders:

- `./data`: This folder contains the [CMU House image dataset](http://vasc.ri.cmu.edu/idb/html/motion/house/).

- `./save`: This folder contains the experimental results reported in the paper.

- `./src`: This folder contains the main implementation of FGM as well
       as other baselines.

- `./lib`: This folder contains some necessary library functions.

- `./make.m`: Matlab makefile for C++ code.

- `./addPath.m`: Adds the sub-directories into the path of Matlab.

- `./demoToy.m`: A demo comparison of different graph matching methods on the synthetic dataset.

- `./demoHouse.m`: A demo comparison of different graph matching methods on the on the [CMU House image dataset](http://vasc.ri.cmu.edu/idb/html/motion/house/).

- `./testToy.m`: Testing the performance of different graph matching
             methods on the synthetic dataset. This is a similar
             function used for reporting (Fig. 4) the first
             experiment (Sec 5.1) in the CVPR 2012 paper [2].

- `./testHouse.m`: Testing the performance of different graph matching
              methods on the [CMU House image dataset](http://vasc.ri.cmu.edu/idb/html/motion/house/).  This is the
              same function used for reporting (Fig. 4) the first
              experiment (Sec 5.1) in the CVPR 2013 paper [1].


C++ Code
========

We provide several C++ codes under `src/asg/fgm/matrix` to perform
matrix products between binary matrices in a more efficient
way. For instance, the function `multiGXH.cpp` is used to more
efficiently compute the matrix product, `G^T * X * H`, where G and
H are two binary matrices.


References
==========

[1] F. Zhou and F. De la Torre, "Deformable Graph Matching," in IEEE
Conference on Computer Vision and Pattern Recognition (CVPR), 2013.

[2] F. Zhou and F. De la Torre, "Factorized Graph Matching," in IEEE
Conference on Computer Vision and Pattern Recognition (CVPR), 2012.

[3] M. Leordeanu and M. Hebert, "A spectral technique for
correspondence problems using pairwise constraints," in International
Conference on Computer Vision (ICCV), 2005.

[4] T. Cour, P. Srinivasan and J. Shi, "Balanced Graph Matching", in
Advances in Neural Information Processing Systems (NIPS), 2006.

[5] S. Gold and A. Rangarajan, "A Graduated Assignment Algorithm for
Graph Matching", IEEE Transactions on Pattern Analysis and Machine
Intelligence (PAMI), 1996.

[6] R. Zass and A. Shashua, "Probabilistic Graph and Hypergraph
Matching", in IEEE Conference on Computer Vision and Pattern
Recognition (CVPR), 2008.

[7] M. Leordeanu, M. Hebert and R. Sukthankar, "An Integer Projected
Fixed Point Method for Graph Matching and MAP Inference", in Advances
in Neural Information Processing Systems (NIPS), 2009.

[8] M. Cho, J. Lee and K. Lee, "Reweighted Random Walks for Graph
Matching", in European Conference on Computer Vision (ECCV), 2010.


Copyright
=========

This software is free for use in research projects. If you
publish results obtained using this software, please use this
citation.

    @inproceedings{ZhouD13,
       author       = {Feng Zhou and Fernando {De la Torre}},
       title        = {Deformable Graph Matching},
       booktitle    = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
       year         = {2013},
    }

If you have any question, please feel free to contact Feng Zhou (zhfe99@gmail.com).
