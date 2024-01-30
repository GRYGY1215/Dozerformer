# Dozerformer
Code of paper ["Dozerformer: Sequence Adaptive Sparse Transformer for Multivariate Time Series Forecasting"] ([Arxiv](https://arxiv.org/abs/2312.06874))

The Dozerformer achieves SOTA on nine benchmarks.

## Introduction
Transformers have achieved remarkable performance in multivariate time series(MTS) forecasting due to their capability to capture long-term dependencies. However, the canonical attention mechanism has two key limitations: (1) its quadratic time complexity limits the sequence length, and (2) it generates future values from the entire historical sequence.
To address this, we propose a Dozer Attention mechanism consisting of three sparse components: (1) Local, each query exclusively attends to keys within a localized window of neighboring time steps. (2) Stride, enables each query to attend to keys at predefined intervals. (3) Vary, allows queries to selectively attend to keys from a subset of the historical sequence. Notably, the size of this subset dynamically expands as forecasting horizons extend. Those three components are designed to capture essential attributes of MTS data, including locality, seasonality, and global temporal dependencies.
Additionally, we present the Dozerformer Framework, incorporating the Dozer Attention mechanism for the MTS forecasting task.
We evaluated the proposed Dozerformer framework with recent state-of-the-art methods on nine benchmark datasets and confirmed its superior performance.

## Train and Test
1. Install the required packages: `pip install -r requirements.txt`
2. Data are publicly available at [Google Drive](https://drive.google.com/file/d/1CC4ZrUD4EKncndzgy5PSTzOPSqcuyqqj/view?usp=sharing) or [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/b8f4a78a39874ac9893e/?dl=1).
3. To reproduce the experimental results presented in the paper. Simply run the scripts at "/scripts/" as follows:
   ```
   bash ./scripts/ETTh1.sh
   bash ./scripts/ETTh2.sh
   bash ./scripts/ETTm1.sh
   bash ./scripts/ETTm2.sh
   bash ./scripts/electricity.sh
   bash ./scripts/Exchange_Rate.sh
   bash ./scripts/Traffic.sh
   bash ./scripts/WTH.sh
   bash ./scripts/ILI.sh
   ```
   
## Train and Test
If you find this repository beneficial for your research, kindly include a citation:
```
@misc{zhang2023dozerformer,
      title={Dozerformer: Sequence Adaptive Sparse Transformer for Multivariate Time Series Forecasting}, 
      author={Yifan Zhang and Rui Wu and Sergiu M. Dascalu and Frederick C. Harris Jr au2},
      year={2023},
      eprint={2312.06874},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Acknowledgements
We sincerely appreciate the foundational code from the following GitHub repositories: \
https://github.com/wanghq21/MICN \
https://github.com/zhouhaoyi/Informer2020 \
https://github.com/Thinklab-SJTU/Crossformer \
https://github.com/thuml/Time-Series-Library \
https://github.com/cure-lab/SCINet
