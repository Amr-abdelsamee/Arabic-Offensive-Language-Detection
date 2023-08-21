# Arabic Offensive Language Detection

## 1. Introduction
In the era of social media, Twitter has emerged as a powerful platform for people to express their opinions, share information, and engage in conversations. However, this openness also exposes users to offensive and harmful content. Offensive tweets can spread quickly and have serious consequences, including inciting hate speech, cyberbullying, and spreading misinformation. To address this issue, an effective method for detecting offensive tweets is essential to ensure a safer online environment.

This project focuses on building a robust offensive tweet classification system that utilizes three different approaches: traditional Machine Learning (ML) algorithms, and some NLP Transformers(BERT, ARABERTv2, MARBERTv2). The primary objective is to conduct a comparative analysis of these methods to understand their performance in accurately identifying offensive tweets.

## 2. Motivations
- **Online Safety and Moderation:** Detecting offensive tweets is crucial for maintaining a safe and respectful online environment. By identifying and flagging harmful content, social media platforms can take appropriate moderation actions to protect their users.
- **Understanding Model Performance:** Comparing different approaches, including ML algorithms, LSTM, and Transformers, helps us gain insights into their strengths and weaknesses. This understanding is crucial for selecting the most suitable model for offensive tweet classification tasks.

By comparing traditional ML algorithms, and Transformers, this project contributes to the advancement of offensive tweet detection in the Arabic language. The findings can guide the development of reliable and efficient systems that promote a safer and more respectful online discourse.

## 3. Data
The dataset used for classifying offensive tweets is split into three subsets: training, testing, and development (validation). The training set contains 8,886 tweets and is used to train the machine learning algorithms and the MARBERTv2 transformer model. The test set comprises 2,540 tweets and serves as an unseen dataset to evaluate the models’ generalization performance. The development set, with 1,270 tweets, is used for hyperparameter tuning and preventing overfitting. The tweets in the dataset contain noise, mentions, hashtags, and emojis, making the classification task challenging.
### 3.1 Preprocessing
Data preprocessing is an important step in classification tasks. Many unnecessary tokens may not help in the given task; in fact, they may have a negative impact on the final results. We performed the following pre-processing steps on the dataset:

- **Arabic Letter Normalization:** We unified letters that may appear in different forms in the text.
- **Punctuation Normalization:** We removed all punctuation from the text.
- **Digit Normalization:** We removed all numbers.
- **Hashtag Segmentation:** We removed the spaces and "#" from hashtags, leaving only the words.
- **Diacritic Removal:** Diacritics were removed.
- **Removal of Symbols and Non-Arabic Words:** Symbols and non-Arabic words were removed.
- **Removal of Repeated Characters or Emojis:** Repeated characters or emojis occurring more than two times were removed.
- **Emojis:** Emojis are important in the text, so we used a dataset containing emojis and their names. Since these names are in English, we utilized the Google Translate API to translate them into Arabic. Each emoji was replaced with its corresponding Arabic name.
![Figure 1: A sample from the dataset after applying the preprocessing](https://github.com/Amr-abdelsamee/Arabic-Offensive-Language-Detection/blob/main/screenshots/data.png)

### 3.2 Data Augmentation
The dataset mentioned in Section 3 has an imbalance, which could introduce biases in the model's results. To address this, we employed two methods for balancing the data:
- **Undersampling:** We made both classes the same size as the smaller one.
- **Oversampling:** We expanded the smaller class to match the size of the larger one by duplicating some instances.

It's important to note that these methods have limitations. Oversampling might lead to overfitting, where the model memorizes duplicated data instead of learning useful patterns. On the other hand, undersampling could remove crucial data, potentially impacting the model's generalization ability. We used three datasets—imbalanced, undersampled, and oversampled—and fed them to all the models to compare the differences between the sampled and imbalanced data.

### 4. Models
During the development of the offensive tweet classification model, we explored various hyperparameters and architecture choices to optimize the performance of each algorithm and model. For the machine learning algorithms (Decision Tree, SVC, Linear SVC, Random Forest, SGD, and KNN), we tuned hyperparameters such as the number of estimators and maximum depth. Different combinations of these hyperparameters were tested to find the best configuration that yielded the highest accuracy and generalization ability.

### 4.1 ML Algorithms
We utilized the following algorithms for text classification:

- Decision Tree.
- Support Vector Classifier (SVC).
- Linear Support Vector Classifier (Linear SVC).
- Random Forest.
- Stochastic Gradient Descent (SGD).
- K-Nearest Neighbors (KNN).

In Section 6, we present the performance of each model on each of the three datasets: the imbalanced, the undersampled, and the oversampled data.

### 4.2 Transformers
We employed three different transformer models on our datasets to facilitate a comparison of their respective results. Furthermore, we compared the outcomes of these transformer models with the results obtained from the previous methods. The transformer models we utilized are:

- bert-base-multilingual-cased
- ARABERTv02
- MARBERTv2

All of these transformer models exhibited superior performance compared to the previous methods we employed.

## 5. Results
We have compiled a table summarizing the outcomes obtained from all the previously employed approaches. Upon analysis, it was determined that the MARBERTv2 transformer model demonstrated the highest performance, achieving an f1-score of 82.2%. Conversely, the K-Nearest Neighbors (KNN) model yielded the least favorable result with an f1-score of 60.9% this is for the unbalanced data. For the Undersampled data MARBERTv2 transformer model demonstrated the highest performance, achieving an f1-score of 80.9% and the Decision tree model yielded the least favorable result with an f1-score of 63.7%. For the Oversampled data MARBERTv2 transformer model demonstrated the highest performance with f1-score of 80.9% and the Decision tree model yielded the least favorable result with an f1-score of 62.5%.

The following table presents a summary of the results:

![Figure 2: A comparison between the different approaches we used](https://github.com/Amr-abdelsamee/Arabic-Offensive-Language-Detection/blob/main/screenshots/results.png)


