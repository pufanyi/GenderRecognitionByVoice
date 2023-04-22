[![](./images/Cover/cover.svg)](https://pufanyi.github.io/GenderRecognitionByVoice/)

This is the project for NTU course *SC1015 Introduction to Data Science and Artificial Intelligence*.

Our goal is to study the relationship between sound data and the gender of the speaker, and to attempt to estimate the gender of the speaker through various models.

The main page of our project is [here](https://pufanyi.github.io/GenderRecognitionByVoice).

## Content

All code is located under the src directory.

Please read through the code in the flowing sequence:

- [`DataPreparationAndExploration.ipynb`](./src/DataPreparationAndExploration.ipynb)
- [`GenderRecognitionUsingTreeBasedAlgorithms.ipynb`](./src/GenderRecognitionUsingTreeBasedAlgorithms.ipynb)
- [`GenderRecognitionUsingNumericalAlgorithms.ipynb`](./src/GenderRecognitionUsingNumericalAlgorithms.ipynb)
- [`SVMFurtherExploration.ipynb`](./src/SVMFurtherExploration.ipynb)
- [`PCAFurtherExploration.ipynb`](./src/PCAFurtherExploration.ipynb)

## Problem Definition

- How can we differentiate the gender of a speaker through their voice?
  - What are the key features to achieve this?
  - Which models can better predict the gender of a speaker?

## Highlights of Data preparation

### Remove Duplicate Data

We removed the `meanfreq` (mean frequency) and `centroid` (frequency centroid). They are the same in definition.

### Log Transformation

To prepare the input data, we applied a log transformation, which helped to reduce the impact of extreme values and normalize the distribution of the data. This transformation reduced skewness, brought the data closer to normal distribution, and improved the accuracy of our model by reducing the influence of extreme values.

![](./images/DataPreparation/LogTransform.png)

### Data Normalization

The purpose of normalization is to ensure that all features are treated equally in terms of their scale.

Before normalization, the accuracy of SVM was approximately 60%. Even after changing various parameters, carrying out plenty of trials, the accuracy did not have a prominent improvement. After the normalization, the accuracy can be enhanced to nearly 99%.

### Outlier Removal

When dealing with datasets with a large number of predictors, it can be challenging to perform outlier removal on each specific predictor. Therefore, we utilized the Isolation Forest algorithm to identify and remove outliers from the input data.

## Models Used

| Model | Training Accuracy | Testing Accuracy |
| --- | --- | --- |
| Classification Tree | 1.0000 | 0.9751 |
| Random Forest | 1.0000 | 0.9801 |
| Logistic Regression | 0.9763 | 0.9734 |
| K-Nearest Neighbors | 1.0000 | 0.9817 |
| Support Vector Machine | 0.9896 | 0.9834 |

## Highlights of Machine Learning

### Cross Validation

Previously, we employed a conventional train-test split to evaluate the performance of our gender classification model. In order to further improve the accuracy and efficiency of our algorithm, we will utilize CV to evaluate the model's generalization performance and reduce overfitting.

### Support Vector Machines Exploration

We conducted an in-depth analysis of SVM by exploring and adjusting its parameters to achieve optimal performance. To explicitly refine our understanding of each parameter, we plotted the separating hyperplane for each kernel. This process allowed us to fine-tune the SVM algorithm and gain a better understanding of its behavior.

### Principal Component Analysis

PCA is a type of unsupervised learning algorithm that reduces the number of features while still retaining most of the original variance in the data.

According to our data, the initial accuracy is approximately 0.98. When the features are reduced below 8, the accuracy rapidly declines to 0.88.

### Prediction Modelling

After utilizing various models, we started to consider whether we could improve our results by integrating the outputs of multiple high-performing models and selecting the majority vote. However, we eventually discovered that the accuracy of this algorithm was not as ideal as we had hoped, and did not even outperform the highest individual accuracy achieved. In other words, blindly combining all models does not necessarily lead to better results.

## Conclusion

What are the key features to achieve this?

> According to classification tree analysis, `IQR` and `meanfun` have been identified as the two main predictors for differentiating male and female voices.

Which models can better predict the gender of a speaker?

> Among the various models, we found that the SVM model with an RBF kernel achieved the highest accuracy, with a score of 0.9834.

## What We Learnt

- Importance of data preparation
  - The initial lack of normalization has resulted in poor performance of the SVM model. Despite spending significant time adjusting the SVM parameters, the model still showed poor accuracy. However, after performing normalization, we observed a significant improvement in the accuracy of our SVM model.
- The concepts and implementations of multiple models
  - Supervised learning: SVM, KNN, Random Forest, Logistic Regression
  - Unsupervised learning: PCA
  - Using cross-validation to get the accuracy
- The prediction model created by ourselves

## Group Members

| Name | GitHub Account | Email | Contribution |
| --- | --- | --- | --- |
| Pu Fanyi | [pufanyi](https://github.com/pufanyi) | FPU001@e.ntu.edu.sg | ??? |
| Jiang Jinyi | [Jinyi087](https://github.com/Jinyi087) | D220006@e.ntu.edu.sg | ??? |
| Shan Yi | [shanyi26](https://github.com/shanyi26) | SH005YI@e.ntu.edu.sg | ??? |