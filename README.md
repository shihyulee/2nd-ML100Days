# 2nd-ML100Days
[/ 第 2 屆 / 機器學習 百日馬拉松](https://ai100-2.cupoy.com/)
## Materials
### 1-資料清理數據前處理
* Day 1 - 資料介紹與評估資料
    * Hints
        * [如何取平方](https://googoodesign.gitbooks.io/-ezpython/unit-1.html)

* Day 2 - EDA-1/讀取資料EDA: Data summary
    * Hints
        * [基礎教材](https://bookdata.readthedocs.io/en/latest/base/01_pandas.html#DataFrame-%E5%85%A5%E9%97%A8)
* Day 3 - 3-1如何新建一個 dataframe?3-2 如何讀取其他資料? (非 csv 的資料)
    * Hints:
        * [隨機產生數值](https://blog.csdn.net/christianashannon/article/details/78867204)
        * 使用 [Request](https://blog.gtwang.org/programming/python-requests-module-tutorial/) 抓取資料
        * [字串分割](http://www.runoob.com/python/att-string-split.html)
        * 例外處理: [Try-Except](https://pydoing.blogspot.com/2011/01/python-try.html)
    * Refs:
        * [npy file](https://towardsdatascience.com/why-you-should-start-using-npy-file-more-often-df2a13cc0161)
        * [Pickle](https://docs.python.org/3/library/pickle.html) 
        * [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) 
* Day 4 - EDA: 欄位的資料類型介紹及處理
    * Hints:
    * Refs:
        * [label-encoder-vs-one-hot-encoder](https://medium.com/@contactsunny/label-encoder-vs-one-hot-encoder-in-machine-learning-3fc273365621)
        * [Pandas 數據類型](https://blog.csdn.net/claroja/article/details/72622375)
* Day 5 - EDA資料分佈
    * Hints:
        * [Descriptive Statistics For pandas Dataframe](https://chrisalbon.com/python/data_wrangling/pandas_dataframe_descriptive_stats/)
        * [pandas 中的繪圖函數](https://amaozhao.gitbooks.io/pandas-notebook/content/pandas%E4%B8%AD%E7%9A%84%E7%BB%98%E5%9B%BE%E5%87%BD%E6%95%B0.html)
    * Refs:
        * [敘述統計與機率分佈 by 吳漢銘老師](http://www.hmwu.idv.tw/web/R_AI_M/AI-M1-hmwu_R_Stat&Prob.pdf)
        * [Standard Statistical Distributions](https://www.healthknowledge.org.uk/public-health-textbook/research-methods/1b-statistical-methods/statistical-distributions)
        * [List of probability distributions ](https://en.wikipedia.org/wiki/List_of_probability_distributions)
* Day 6 - EDA: Outlier 及處理
    * hints:
        * [ECDF](https://zh.wikipedia.org/wiki/%E7%BB%8F%E9%AA%8C%E5%88%86%E5%B8%83%E5%87%BD%E6%95%B0)
        * [ECDF with Python](https://stackoverflow.com/questions/14006520/ecdf-in-python-without-step-function)
    * Refs:
        * [Ways to Detect and Remove the Outliers](https://towardsdatascience.com/ways-to-detect-and-remove-the-outliers-404d16608dba)
        * [How to Use Statistics to Identify Outliers in Data](https://machinelearningmastery.com/how-to-use-statistics-to-identify-outliers-in-data/)
* Day 7 - 常用的數值取代：中位數與分位數連續數值標準化
    * hints:
    * Refs:
        * [Is it a good practice to always scale/normalize data for machine learning?](https://stats.stackexchange.com/questions/189652/is-it-a-good-practice-to-always-scale-normalize-data-for-machine-learning)
* Day 8 - DataFrame operationData frame merge/常用的 DataFrame 操作
    * hints:
    * Refs:
        * [Cheat sheet Pandas Python (DataCamp)](https://assets.datacamp.com/blog_assets/PandasPythonForDataScience.pdf) 
        * [Standard score](https://en.wikipedia.org/wiki/Standard_score)
* Day 9 - 程式實作 EDA: correlation/相關係數簡介
    * hints:
        * [隨機變數](https://en.wikipedia.org/wiki/Random_variable)
    * Refs:
        * [Pearson’s correlation](http://www.statstutor.ac.uk/resources/uploaded/pearsons.pdf)
        * [Guess The Correlation](http://guessthecorrelation.com/)
* Day 10 - EDA from Correlation (相關係數實作)
* Day 11 - EDA: 不同數值範圍間的特徵如何檢視/繪圖與樣式Kernel Density Estimation (KDE)
    * hints:
        * [KDE](https://scikit-learn.org/stable/auto_examples/neighbors/plot_kde_1d.html#sphx-glr-auto-examples-neighbors-plot-kde-1d-py)
        * [什麼是 KDE](https://blog.csdn.net/unixtch/article/details/78556499)
    * Refs:
        * [Python Graph Gallery](https://python-graph-gallery.com/)
        * [R Graph Gallery](https://www.r-graph-gallery.com/)
        * [R Graph Gallery (Interactive plot, 互動圖)](https://bl.ocks.org/mbostock)
        * [D3.js](https://d3js.org/)
        * [核密度估計基礎](https://blog.csdn.net/david830_wu/article/details/66974189)
* Day 12 - EDA: 把連續型變數離散化
    * hints:
    * Refs:
        * [連續特徵的離散化 : 在什麼情況下可以獲得更好的效果(知乎)](https://www.zhihu.com/question/31989952)