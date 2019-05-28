# 2nd-ML100Days
[/ 第 2 屆 / 機器學習 百日馬拉松](https://ai100-2.cupoy.com/)
## Materials
### 1 - 資料清理數據前處理
* Day 1 - 資料介紹與評估資料
    * Hints
        * [如何取平方](https://googoodesign.gitbooks.io/-ezpython/unit-1.html)
        * [也談談機器學習中的 Evaluation Metrics](https://blog.csdn.net/aws3217150/article/details/50479457)

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
* Day 13 - 程式實作 把連續型變數離散化
* Day 14 - Subplots
    * hints:
    * Refs:
        * [matplotlib 官方範例](https://matplotlib.org/examples/pylab_examples/subplots_demo.html)
        * [Multiple Subplots](https://jakevdp.github.io/PythonDataScienceHandbook/04.08-multiple-subplots.html)
        * [Seaborn.jointplot](https://seaborn.pydata.org/generated/seaborn.jointplot.html)
* Day 15 - Heatmap & Grid
    * hints:
        * [numpy.random.randn()用法](https://blog.csdn.net/u012149181/article/details/78913167)
    * Refs:
        * [Heatmap matplotlib 官方範例](https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html)
        * [Heatmap Seaborn 範例](https://www.jianshu.com/p/363bbf6ec335)
        * [Visualizing Data with Pairs Plots in Python](https://towardsdatascience.com/visualizing-data-with-pair-plots-in-python-f228cf529166)
* Day 16 - 模型初體驗 Logistic Regression

### 2 - 資料科學特徵工程技術

* D17 - 特徵工程簡介
    * hints:
        * [log1p(x) 和 expm1(x)](https://blog.csdn.net/liyuanbhu/article/details/8544644)
    * Refs:
        * [知乎-特徵工程到底是什麼](https://www.zhihu.com/question/29316149)
        * [痞客幫-iT邦2019鐵人賽 : 為什麼特徵工程很重要](https://ithelp.ithome.com.tw/articles/10200041?sc=iThelpR)

* D18 - 特徵類型
    * hints:
        * [Python 字串格式化以及 f-string 字串格式化](https://blog.louie.lu/2017/08/08/outdate-python-string-format-and-fstring/)
    * Refs:
        * [ k- fold cross validation](https://zhuanlan.zhihu.com/p/24825503)
        * [Python Tutorial 第二堂 - 數值與字串型態](https://openhome.cc/Gossip/CodeData/PythonTutorial/NumericStringPy3.html)
        * [Built-in Types Python 官方說明](https://docs.python.org/3/library/stdtypes.html)
* D19 - 數值型特徵-補缺失值與標準化
    * hints:
    * Ref:
        * [掘金 : Python數據分析基礎 : 數據缺失值處理](https://juejin.im/post/5b5c4e6c6fb9a04f90791e0c)
        * [數據標準化 / 歸一化normalization](https://blog.csdn.net/pipisorry/article/details/52247379)
* D20 - 數值型特徵 - 去除離群值
    * hints:
    * Ref:
        * [離群值](https://zhuanlan.zhihu.com/p/33468998)
        
* D21 - 數值型特徵 - 去除偏態
    * hints:
    * Ref:
        * [機器學習數學|偏度與峰度及其python 實現](https://blog.csdn.net/u013555719/article/details/78530879)
* D22 - 類別型特徵 - 基礎處理
    * hints:
    * Ref:
        * [數據預處理：獨熱編碼（One-Hot Encoding）和 LabelEncoder標籤編碼](https://www.twblogs.net/a/5baab6e32b7177781a0e6859/zh-cn/)
* D23 - 類別型特徵 - 均值編碼
    * hints:
    * Ref:
        * [平均數編碼 ：針對高基數定性特徵(類別特徵)的數據處理/ 特徵工程](https://zhuanlan.zhihu.com/p/26308272)
* D24 - 類別型特徵 - 其他進階處理
    * hints:
    * Ref:
        * [Feature hashing (特徵哈希)](https://blog.csdn.net/laolu1573/article/details/79410187)
        * [基於sklearn的文本特徵抽取](https://www.jianshu.com/p/063840752151)
* D25 - 時間型特徵
    * hints:
    Ref: [PYTHON-基礎-時間日期處理小結](http://www.wklken.me/posts/2015/03/03/python-base-datetime.html)

* D26 - 特徵組合 - 數值與數值組合
    * hints:
    * Ref:
        * [特徵組合&特徵交叉 (Feature Crosses)](https://segmentfault.com/a/1190000014799038)
        * [簡單高效的組合特徵自動挖掘框架](https://zhuanlan.zhihu.com/p/42946318)
* D27 - 特徵組合 - 類別與數值組合
    * hints:
    * Ref:
        * [利用 Python 數據分析之數據聚合與分組](https://zhuanlan.zhihu.com/p/27590154)
* D28 - 特徵選擇
    * hints;
    * Ref:
        * [特徵選擇](https://zhuanlan.zhihu.com/p/32749489)
        * [特徵選擇線上手冊](https://machine-learning-python.kspax.io/intro-1)
* D29 - 特徵評估
    * hints:
    * Ref:
        * [機器學習 - 特徵選擇算法流程、分類、優化與發展綜述](https://juejin.im/post/5a1f7903f265da431c70144c)
        * [Permutation Importance](https://www.kaggle.com/dansbecker/permutation-importance?utm_medium=email&utm_source=mailchimp&utm_campaign=ml4insights)
* D30 - 分類型特徵優化 - 葉編碼
    * hints:
    * Ref:
        * [Feature transformations with ensembles of trees](https://scikit-learn.org/stable/auto_examples/ensemble/plot_feature_transformation.html#example-ensemble-plot-feature-transformation-py)
        * [Algorithm-GBDT Encoder](https://zhuanlan.zhihu.com/p/31734283)
        * [Factorization Machine](https://kknews.cc/zh-tw/other/62k4rml.html)

### 3 - 機器學習基礎模型建立
* D31 - 機器學習概論
    * hints:
    * Ref:
        * [機器是如何學習到東西的?](https://kopu.chat/2017/07/28/%E6%A9%9F%E5%99%A8%E6%98%AF%E6%80%8E%E9%BA%BC%E5%BE%9E%E8%B3%87%E6%96%99%E4%B8%AD%E3%80%8C%E5%AD%B8%E3%80%8D%E5%88%B0%E6%9D%B1%E8%A5%BF%E7%9A%84%E5%91%A2/)
        * [如何教懂電腦看圖像](https://www.ted.com/talks/fei_fei_li_how_we_re_teaching_computers_to_understand_pictures?language=zh-tw)
* D32 - 機器學習-流程與步驟
    * 機器學習 blog:
        * [Google AI blog](https://ai.googleblog.com/)
        * [Facebook Research blog](https://research.fb.com/blog/)
        * [Apple machine learning journal](https://machinelearning.apple.com/)
        * [機器之心](https://www.jiqizhixin.com/)
        * [雷鋒網](https://www.leiphone.com/category/ai)
    * Ref:
        * [The 7 Steps of Machine Learning (AI Adventures)](https://www.youtube.com/watch?v=nKW8Ndu7Mjw)
* D33 - 機器如何學習?
    * hints:
        * [ML Lecture 1: Regression - Case Study](https://www.youtube.com/watch?v=fegAeph9UaA)
    * Ref: 
        * [學習曲線與 bias/variance trade-off](http://bangqu.com/yjB839.html)