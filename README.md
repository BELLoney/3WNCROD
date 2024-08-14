# 3WNCROD
Xianyong Zhang, **Zhong Yuan***, and Duoqian Miao,[Outlier Detection Using Three-Way Neighborhood Characteristic Regions and Corresponding Fusion Measurement](Paper/2024-WNCROD.pdf), IEEE Transactions on Knowledge and Data Engineering, vol. 36, no. 5, pp. 2082-2095, May 2024, DOI: [10.1109/TKDE.2023.3312108](https://doi.org/10.1109/TKDE.2023.3312108). (Code)

## Abstract
Outliers carry significant information to reflect an anomaly mechanism, so outlier detection facilitates relevant data mining. In terms of outlier detection, the classical approaches from distances apply to numerical data rather than nominal data, while the recent methods on basic rough sets deal with nominal data rather than numerical data. Aiming at wide outlier detection on numerical, nominal, and hybrid data, this paper investigates three-way neighborhood characteristic regions and corresponding fusion measurement to advance outlier detection. First, neighborhood rough sets are deepened via three-way decision, so they derive three-way neighborhood structures on model boundaries, inner regions, and characteristic regions. Second, the three-way neighborhood characteristic regions motivate the information fusion and weight measurement regarding all features, and thus, a multiple neighborhood outlier factor emerges to establish a new method of outlier detection; furthermore, a relevant outlier detection algorithm (called 3WNCROD) is designed to comprehensively process numerical, nominal, and mixed data. Finally, the 3WNCROD algorithm is experimentally validated, and it generally outperforms 13 contrast algorithms to perform better for outlier detection.

## Framework
![image](Paper/WNCROD_Framework.pdf)

## Usage
You can run Demo_WNCROD.m or WNCROD.py:
```
clc;
clear
format shortG;

load Example.mat

Dataori=Example;

trandata=Dataori;
trandata(:,2:3)=normalize(trandata(:,2:3),'range');

X_tem=[1,2,5,6];
lammda=1;

out_scores=WNCROD(trandata,X_tem,lammda)

```
You can get outputs as follows:
```
out_scores =
   0.24704
   0
   0
   0.13215
```

## Citation
If you find 3WNCROD useful in your research, please consider citing:
```
@article{zhang2023outlier,
  title={Outlier Detection Using Three-way Neighborhood Characteristic Regions and Corresponding Fusion Measurement},
  author={Zhang, Xian Yong and Yuan, Zhong and Miao, Duo Qian},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  volume={36},
  number={5},
  pages={2082--2095},
  year={2024},
  doi={10.1109/TKDE.2023.3312108},
  publisher={IEEE}
}
```
## Contact
If you have any questions, please contact yuanzhong@scu.edu.cn.
