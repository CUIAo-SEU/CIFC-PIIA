# CIFC-PIIA
China's Internet-Famous Cities Progressive Intelligent Identification Algorithm

## 项目介绍
Based on long-time-series Baidu Index data, this algorithm identifies four types of internet-famous cities (High-level Cities, Multi-peak Cities, Seasonal Cities, and Isolated Impulse Cities) through a four-step process: "Threshold Screening - Peak Identification - Dynamic Time Warping - Seasonal Decomposition".

## 代码文件说明
- 01_Threshold Screening.py: Threshold screening to eliminate cities with low attention
- 02_Peak Identification.py: Peak identification to extract Isolated Impulse Cities
- 03_Dynamic Time Warping.py: Dynamic Time Warping (DTW) to cluster High-level Cities
- 04_Seasonal Decomposition.py: Seasonal decomposition to identify Seasonal Cities and Multi-peak Cities

## 依赖库
The following Python libraries are required to run the code:
- pandas
- numpy
- scipy
- tslearn
- statsmodels
- matplotlib

## 安装依赖
pip install pandas numpy scipy tslearn statsmodels matplotlib
