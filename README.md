# GAN-IndexFuture
## Generation of One-minute Charts for Day Trading by Generative Adversarial Nets
### Introduction
Many machine learning techniques such as deep learning models need tons of data to train but in reality the financial data are limited and then restrain the power of models. Further, many investment funds need more new data to test their trading strategies and overcome overfitting problems. In the past, many time series models like ARIMA and GARCH can generate new data , but these mehtods rely on strong model assumption. In comparison, we try to implement the state-of-art GANs method, which is a game-thoery based unsupervised learning model, to generate new financial data.

### Main contributions
In this project, we tried to implement several GANs techniques to simulate open, high, low, and close price of three share price index futures in China and generate one-minute charts for day trading (240 minutes a day since 2016-01-01). 
Our main network structures are DCGANs, and we restructure the raw dataset using a log-return trick, which is essential for GANs to learning continuous time series features. 

### Dataset
The three datasets below are provided by [likelihood lab](http://www.maxlikelihood.cn/), which is a non-profit AI research lab.
1. IF targets on CSI 300 Index and starts trading since 2010-04-16.
2. IH targets on SSE 50 index and starts trading since 2015-04-16.
3. IC targets on CSI 500 index and starts trading since 2015-04-16. 

### Code implementation process 
1. preprocess.py read raw data (CSV files) and transform it into npy files to speed up training.
2. data_loader.py load the transformed data into model.
3. visualization.py and laplotter.py are responsible for data visualization such as Candlestick  charts and loss plots.
3. gan.py train a DCGANs to learn the data distribution of our dataset and generate new data. 


