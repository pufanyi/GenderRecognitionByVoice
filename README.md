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

How can we classify the gender of a speaker through their voice?
- What are the key features to achieve this?
- Which models can better predict the gender of a speaker?

## Highlights of Data preparation

### Remove Duplicate Data

The features `meanfreq` (mean frequency) and `centroid` (frequency centroid) were found to be identical in definition, so we removed the duplicate data to avoid redundancy and potential confusion in the analysis.

### Data Correction

To prepare the input data, we performed data correction by applying a log transformation. This helped to mitigate the impact of extreme values and normalize the distribution of the data. The log transformation effectively reduced skewness, brought the data closer to a normal distribution, and improved the accuracy of our model by mitigating the influence of outliers. Overall, this data correction technique proved to be an effective way to preprocess the input data and enhance the performance of our model.

![](./images/DataPreparation/LogTransform.png "Log Transformation")

### Data Normalization

The purpose of normalization is to ensure that all features are treated equally in terms of their scale. After applying normalization, we saw a remarkable increase in accuracy of our SVM model from 0.6934 to 0.9834.

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

### Cross Validation (CV)

Previously, we employed a conventional train-test split to evaluate the performance of our gender classification model. In order to further improve the accuracy and efficiency of our algorithm, we utilized CV to evaluate the model's generalization performance and reduce overfitting.

### Support Vector Machines (SVM) Exploration

We conducted an in-depth analysis of SVM by exploring and adjusting its parameters to achieve optimal performance. To explicitly refine our understanding of each parameter, we plotted the separating hyperplane for each kernel. This process allowed us to fine-tune the SVM algorithm and gain a better understanding of its behavior.

### Principal Component Analysis (PCA)

We aimed to improve efficiency by compressing the predictor data using PCA. Through our exploration of compressing the data to varying dimensions and assessing the resulting accuracy, we gained a deeper understanding of the application of PCA. Our findings demonstrate that by compressing the data to a certain degree, we can achieve a good balance between accuracy and efficiency, leading to better performance in our predictive modeling.

### Ensemble Vote Model

We developed an Ensemble Vote model that integrated the outputs of multiple high-performing models, including Multi-Layer Perceptron (MLP), Random Forest (RF), and Support Vector Machine (SVM), and selected the majority vote to improve our prediction results. However, the accuracy of the Ensemble Vote model did not meet our expectations. This experience taught us the importance of carefully selecting and combining models based on their individual strengths and weaknesses, and considering the underlying assumptions and limitations of each model. We also learned the significance of interpreting the results and understanding the reasoning behind the outputs, rather than blindly relying on a model's prediction. Overall, our project highlights the importance of thoughtful model selection and interpretation in achieving accurate and reliable predictions.


```python
def predict(self, X):
    rf_predictions = self.rfClassifier.predict(X)
    svm_predictions = self.svmClassifier.predict(X)
    mlp_predictions = self.mlpClassifier.predict(X)
    predictions = []
    for i in range(len(X)):
        predictions.append(np.argmax(np.bincount([rf_predictions[i], svm_predictions[i], mlp_predictions[i]])))
    return predictions
```

## Conclusion

What are the key features to classify the gender of a speaker through their voice?

> According to classification tree analysis, `IQR` and `meanfun` have been identified as the two main predictors for differentiating male and female voices. A higher IQR and lower meanfun are more indicative of a male speaker.

Which models can predict the gender of a speaker with higher accuracy?

> Among the various models, the SVM model with an RBF kernel achieved the highest accuracy, with a score of 0.9834.

## What We Learnt

- Importance of data preparation
  - The initial lack of normalization has resulted in poor performance of the SVM model. Despite spending significant time adjusting the SVM parameters, the model still showed poor accuracy. However, after performing normalization, we observed a significant improvement in the accuracy of our SVM model.
- Concepts and implementations of multiple models
  - Supervised learning: Random Forest, Logistic Regression, SVM, KNN
  - Unsupervised learning: PCA
  - Using cross-validation to get the accuracy
- Ensemble Vote model

## Group Members

| Name | GitHub Account | Email | Contribution |
| --- | --- | --- | --- |
| Pu Fanyi | [pufanyi](https://github.com/pufanyi) | FPU001@e.ntu.edu.sg | ??? |
| Jiang Jinyi | [Jinyi087](https://github.com/Jinyi087) | D220006@e.ntu.edu.sg | ??? |
| Shan Yi | [shanyi26](https://github.com/shanyi26) | SH005YI@e.ntu.edu.sg | ??? |
