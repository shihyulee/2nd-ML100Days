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
* D34 - 訓練/測試集切分的概念
    * hints:
        * [sklearn.model_selection.train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
    * Ref:
        * [台大電機李宏毅教授講解訊練/驗證/測試集的意義](https://www.youtube.com/watch?v=D_S6y0Jm6dQ&feature=youtu.be&t=1948)
* D35 - regression vs. classification
    * hints:
    * Ref:
        * [回歸與分類的比較](http://zylix666.blogspot.com/2016/06/supervised-classificationregression.html)
* D36 - 評估指標選定/evaluation metrics
    * hints:
        * [F1 Score Code](https://github.com/scikit-learn/scikit-learn/blob/bac89c2/sklearn/metrics/classification.py#L620)
    * Ref:
        * [ROC curves and Area Under the Curve explained](https://www.dataschool.io/roc-curves-and-auc-explained/)
        * [機器學習模型評估](https://zhuanlan.zhihu.com/p/30721429)
* D37 - regression model 介紹 - 線性迴歸/羅吉斯回歸
    * hints:
        * [線性迴歸的運作原理](https://brohrer.mcknote.com/zh-Hant/how_machine_learning_works/how_linear_regression_works.html)
        * [線性分類-邏輯斯回歸](https://medium.com/@yehjames/%E8%B3%87%E6%96%99%E5%88%86%E6%9E%90-%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E7%AC%AC3-3%E8%AC%9B-%E7%B7%9A%E6%80%A7%E5%88%86%E9%A1%9E-%E9%82%8F%E8%BC%AF%E6%96%AF%E5%9B%9E%E6%AD%B8-logistic-regression-%E4%BB%8B%E7%B4%B9-a1a5f47017e5)
        * [你可能不知道的邏輯迴歸](https://taweihuang.hpd.io/2017/12/22/logreg101/)
    * Ref:
        * [Andrew Ng 教你 Linear regression](https://zh-tw.coursera.org/lecture/machine-learning/model-representation-db3jS)
        * [邏輯回歸常見面試題總結](https://blog.csdn.net/qq_23269761/article/details/81778585)
* D38 - regression model 程式碼撰寫
    * hints:
        * [sklearn 邏輯回歸中的參數的詳解](https://blog.csdn.net/lc574260570/article/details/82116197)
        * [What is the difference between one-vs-all binary logistic regression and multinomial logistic regression?](https://www.quora.com/What-is-the-difference-between-one-vs-all-binary-logistic-regression-and-multinomial-logistic-regression)
        * [What is the difference between logistic and logit regression?](https://stats.stackexchange.com/questions/120329/what-is-the-difference-between-logistic-and-logit-regression/120364#120364)
    * Ref:
        * [Linear Regression / Logistic Regression 的 examples](https://github.com/trekhleb/homemade-machine-learning)
        * [深入了解 multinomial Logistic Regression 的原理](http://dataaspirant.com/2017/05/15/implement-multinomial-logistic-regression-python/)
* D39 - regression model 介紹 - LASSO 回歸/ Ridge 回歸
    * hints:
        * [脊回歸 (Ridge Regression)](https://blog.csdn.net/daunxx/article/details/51578787)
        * [Linear, Ridge, Lasso Regression 本質區別](https://www.zhihu.com/question/38121173)
    * Ref:
* D40 - regression model 程式碼撰寫
    * hints:
    * Ref:
* D41 - tree based model - 決策樹 (Decision Tree) 模型介紹
    * hints: 
        * [Gini Impurity vs Entropy](https://datascience.stackexchange.com/questions/10228/gini-impurity-vs-entropy)
        * [決策樹(Decision Tree)以及隨機森林(Random Forest)介紹](https://medium.com/jameslearningnote/%E8%B3%87%E6%96%99%E5%88%86%E6%9E%90-%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E7%AC%AC3-5%E8%AC%9B-%E6%B1%BA%E7%AD%96%E6%A8%B9-decision-tree-%E4%BB%A5%E5%8F%8A%E9%9A%A8%E6%A9%9F%E6%A3%AE%E6%9E%97-random-forest-%E4%BB%8B%E7%B4%B9-7079b0ddfbda)
        * [HOW DECISION TREE ALGORITHM WORKS](http://dataaspirant.com/2017/01/30/how-decision-tree-algorithm-works/)
    * Ref:
* D42 - tree based model - 決策樹程式碼撰寫
    * hints:
    * Ref:
        * [Creating and Visualizing Decision Trees with Python](https://medium.com/@rnbrown/creating-and-visualizing-decision-trees-with-python-f8e8fa394176)
* D43 - tree based model - 隨機森林 (Random Forest) 介紹
    * hints:
        * [隨機森林 (random forest)](http://hhtucode.blogspot.com/2013/06/ml-random-forest.html)
        * [How Random Forest Algorithm Works in Machine Learning](https://medium.com/@Synced/how-random-forest-algorithm-works-in-machine-learning-3c0fe15b6674)
* D44 - tree based model - 隨機森林程式碼撰寫
    * hints:
    * Ref:
        * [Random Forests - The Math of Intelligence(YouTube)](https://www.youtube.com/watch?v=QHOazyP-YlM)
* D45 - tree based model - 梯度提升機 (Gradient Boosting Machine) 介紹
    * hints:
        * [A Kaggle Master Explains Gradient Boosting](http://blog.kaggle.com/2017/01/23/a-kaggle-master-explains-gradient-boosting/)
        * [ML Lecture 22: Ensemble(Youtube)](https://www.youtube.com/watch?v=tH9FH1DH5n0)
        * [How to explain gradient boosting](https://explained.ai/gradient-boosting/index.html)

    * Ref:
        * [GBDT︰梯度提升決策樹](https://ifun01.com/84A3FW7.html)
        * [Kaggle Winning Solution Xgboost Algorithm - Learn from Its Author, Tong He (Youtube)](https://www.youtube.com/watch?v=ufHo8vbk6g4)
        * [Introduction to Boosted Trees (Slide)](https://homes.cs.washington.edu/~tqchen/pdf/BoostedTree.pdf)
* D46 - tree based model - 梯度提升機程式碼撰寫
    * hints: 
        * [Is multicollinearity a problem with gradient boosted trees?](https://www.quora.com/Is-multicollinearity-a-problem-with-gradient-boosted-trees)
    * Ref:
        * [Complete Guide to Parameter Tuning in Gradient Boosting (GBM) in Python](https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/)

### 4 - 機器學習調整參數
* D47 - 超參數調整與優化
    * hints:
        * [Smarter Parameter Sweeps](https://medium.com/rants-on-machine-learning/smarter-parameter-sweeps-or-why-grid-search-is-plain-stupid-c17d97a0e881)
    * Ref:
        * [Scanning hyperspace: how to tune machine learning models](https://cambridgecoding.wordpress.com/2016/04/03/scanning-hyperspace-how-to-tune-machine-learning-models/)
        * [Hyperparameter Tuning the Random Forest in Python](https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74)
* D48 - Kaggle 競賽平台介紹
    * [Data Science London + Scikit-learn](https://www.kaggle.com/c/data-science-london-scikit-learn)
* D49 - 集成方法 : 混合泛化(Blending)
    * hints:
    * Ref:
        * [機器學習技法 Lecture 7: Blending and Bagging (Slide)](https://www.csie.ntu.edu.tw/~htlin/mooc/doc/207_handout.pdf)
        * [機器學習技法 Lecture 7: Blending and Bagging (Video)](https://www.youtube.com/watch?v=mjUKsp0MvMI&list=PLXVfgk9fNX2IQOYPmqjqWsNUFl2kpk1U2&index=27&t=0s)
        * [Kaggle - Superblend](https://www.kaggle.com/tunguz/superblend)
* D50 - 集成方法 : 堆疊泛化(Stacking)
    * hints:
        * [STACKED GENERALIZATION (Paper)](http://www.machine-learning.martinsewell.com/ensembles/stacking/Wolpert1992.pdf)
        * [StackingCVClassifier](http://rasbt.github.io/mlxtend/user_guide/classifier/StackingCVClassifier/)
    * Ref:
        * [如何在 Kaggle 首戰中進入前 10%](https://dnc1994.com/2016/04/rank-10-percent-in-first-kaggle-competition/)

### 5 - 非監督式機器學習
* D54 - clustering 1 非監督式機器學習簡介
    * hints:
        * [Lecture 1.3 — Introduction Unsupervised Learning — [ Machine Learning | Andrew Ng]](https://youtu.be/jAA2g9ItoAc)
    * Ref:
        * [Unsupervised learning：PCA](http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2017/Lecture/PCA.mp4)
        * [Scikit-learn unsupervised learning](https://scikit-learn.org/stable/unsupervised_learning.html)
* D55 - clustering 2 聚類算法
    * hints:
    * Ref:
        * [Clustering 影片來源：Statistical Learning YT](https://www.youtube.com/watch?v=aIybuNt9ps4)
        * [Clustering Means Algorithm 影片來源： [ Machine Learning | Andrew Ng ] YT](https://www.youtube.com/watch?v=hDmNF9JG3lo)
        * [Unsupervised Machine Learning:Flat Clustering](https://pythonprogramming.net/flat-clustering-machine-learning-python-scikit-learn/)
* D56 - K-mean 觀察 : 使用輪廓分析
    * hints: [D56：K-mean 觀察 : 使用輪廓分析](https://blog.csdn.net/wangxiaopeng0329/article/details/53542606)
    * Ref:
* D57 - clustering 3 階層分群算法
    * hints:
    * Ref:
        * [Hierarchical Clustering 影片來源：Statistical Learning YT](https://www.youtube.com/watch?v=Tuuc9Y06tAc)
        * [Example：Breast cancer Microarray study 影片來源：Statistical Learning YT](https://www.youtube.com/watch?v=yUJcTpWNY_o)
* D58 - 階層分群法 觀察 : 使用 2D 樣版資料集
* D59 - dimension reduction 1 降維方法-主成份分析
    * hints:
    * Ref:
        * [Unsupervised learning 影片來源：Statistical Learning](https://www.youtube.com/watch?v=ipyxSYXgzjQ)
        * [Further Principal Components 影片來源：Statistical Learning](https://www.youtube.com/watch?v=dbuSGWCgdzw)
        * [Principal Components Regression 影片來源：Statistical Learning](https://www.youtube.com/watch?v=eYxwWGJcOfw)
        * [Dimentional Reduction 影片來源 Andrew Ng](https://www.youtube.com/watch?v=rng04VJxUt4)
* D60 - PCA 觀察 : 使用手寫辨識資料集
* D61 - dimension reduction 2 降維方法-T-SNE
    * hints:
    * Ref: 
        * [Visualizing Data using t-SNE 影片來源：GoogleTechTalks YT](https://www.youtube.com/watch?v=RJVL80Gg3lA)
        * [Unsupervised Learning 影片來源：李弘毅 YT](https://www.youtube.com/watch?v=GBUEjkpoxXc)
* D62 - t-sne 觀察 : 分群與流形還原

### 6 - 深度學習理論與實作
* D63 - 神經網路介紹
    * hints:
    * Ref:
        * [人工智慧大歷史](https://medium.com/@suipichen/%E4%BA%BA%E5%B7%A5%E6%99%BA%E6%85%A7%E5%A4%A7%E6%AD%B7%E5%8F%B2-ffe46a350543)
        * [3 分鐘搞懂深度學習到底在深什麼](https://panx.asia/archives/53209)
        * [Deep Learning Theory 1-1: Can shallow network fit any function?](https://www.youtube.com/watch?v=KKT2VkTdFyc)
* D64 - 深度學習體驗 : 模型調整與學習曲線
    * hints:
        * [TensorFlowPlayGround](https://playground.tensorflow.org/)
    * Ref:
        * [中文版 TF PlayGround 科技部AI普適研究中心](https://pairlabs.ai/tensorflow-playground/index_tw.html#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.81970&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)
* D65 - 深度學習體驗 : 啟動函數與正規化
    * hints:
    * Ref:
        * [Understanding neural networks with TensorFlow Playground](https://cloud.google.com/blog/products/gcp/understanding-neural-networks-with-tensorflow-playground)
        * [深度深度學習網路調參技巧 with TensorFlow Playground](https://zhuanlan.zhihu.com/p/24720954)

### 7 - 初探深度學習使用Keras
* D66 - Keras 安裝與介紹
    * hints:
    * Ref:
        * [Keras Documentation](https://github.com/keras-team/keras/tree/master/docs)
        * [Keras: 中文文檔](https://keras.io/zh/#keras_1)
* D67 - Keras Dataset
    * hints:
    * Ref:
        * [Keras: The Python Deep Learning library](https://github.com/keras-team/keras/)
        * [Keras dataset](https://keras.io/datasets/)
        * [Predicting Boston House Prices](https://www.kaggle.com/sagarnildass/predicting-boston-house-prices)
        * [Imagenet](http://www.image-net.org/about-stats)
        * [COCO](http://mscoco.org/)
* D68 - Keras Sequential API
    * hints:
    * Ref:
        [Getting started with the Keras Sequential model](https://keras.io/getting-started/sequential-model-guide/)
* D69 - Keras Module API
* D70 - Multi-layer Perception多層感知
    * hints:
    * Ref:
        * [機器學習- 神經網路(多層感知機 Multilayer perceptron, MLP)運作方式](https://medium.com/@chih.sheng.huang821/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E7%A5%9E%E7%B6%93%E7%B6%B2%E8%B7%AF-%E5%A4%9A%E5%B1%A4%E6%84%9F%E7%9F%A5%E6%A9%9F-multilayer-perceptron-mlp-%E9%81%8B%E4%BD%9C%E6%96%B9%E5%BC%8F-f0e108e8b9af)
* D71 - 損失函數
    * hints:
    * Ref:
        * [TensorFlow筆記-06-神經網絡優化-​​損失函數，自定義損失函數，交叉熵](https://blog.csdn.net/qq_40147863/article/details/82015360)
        * [Usage of loss functions](https://keras.io/losses/)
* D72 - 啟動函數
    * hints:
    * Ref:
        * [神經網路常用啟動函數總結](https://zhuanlan.zhihu.com/p/39673127)
        * [激活函數的圖示及其一階導數](https://dashee87.github.io/deep%20learning/visualising-activation-functions-in-neural-networks/)
        * [CS231N Lecture](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture6.pdf)
* D73 - 梯度下降Gradient Descent
    * hints:
    * Ref:
        * [Tensorflow中learning rate decay的技巧](https://zhuanlan.zhihu.com/p/32923584)
        * [機器/深度學習-基礎數學(二):梯度下降法(gradient descent)](https://medium.com/@chih.sheng.huang821/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E5%9F%BA%E7%A4%8E%E6%95%B8%E5%AD%B8-%E4%BA%8C-%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E6%B3%95-gradient-descent-406e1fd001f)
* D74 - Gradient Descent 數學原理
    * hints:
    * Ref:
        * [gradient descent using python and numpy](https://stackoverflow.com/questions/17784587/gradient-descent-using-python-and-numpy)
        * [梯度下降算法的參數更新公式](https://blog.csdn.net/hrkxhll/article/details/80395033)
* D75 - BackPropagation
    * hints:
    * Ref:
        * [Backpropagation](https://en.wikipedia.org/wiki/Backpropagation)
        * [深度學習(Deep Learning)-反向傳播](https://ithelp.ithome.com.tw/articles/10198813)
        * [BP神經網路的原理及Python實現](https://blog.csdn.net/conggova/article/details/77799464)
        * [SimpleBPNetwork](https://github.com/conggova/SimpleBPNetwork)
