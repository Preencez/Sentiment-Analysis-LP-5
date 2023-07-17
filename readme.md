

Covid_ 19 Tweet Sentiment Analysis

![NLP](https://www.seacom.it/wp-content/uploads/2021/06/Natural-Language-Processing.png)

The primary objective of this project is to create a machine learning model that can evaluate the sentiment (positive, neutral, or negative) of Twitter posts specifically related to vaccinations. The model will analyze the content of these tweets and classify them based on their sentiment, providing valuable insights into public opinions and attitudes towards vaccinations on social media platforms.

![COVID-19 Vaccine](https://www.labrepco.com/wp-content/uploads/2020/09/vaccine-covid-19-1.jpg)



![Sentiment Analysis](./images/sentiments_positive_negative_neutral.jpg)

Dataset

The dataset comprises of tweets that have been collected and categorized using Dataset. The tweets are labeled as positive (👍), neutral (🤐), or negative (👎). In order to protect privacy, usernames and web addresses have been excluded from the dataset.

Objective
The primary objective is to develop a machine learning model capable of accurately predicting the sentiment of tweets related to vaccinations. The goal is to create a model that can effectively determine whether the sentiment of a given tweet is positive, neutral, or negative, providing valuable insights into public opinions and attitudes towards vaccinations.

Three mmodels were trainined and the best model was picked 

After training your model, you should evaluate its performance using appropriate evaluation metrics. Once your model is trained and evaluated, you can use it to make predictions on new or unseen data, such as the tweets provided in the Test.csv file.

To participate in the challenge, you are encouraged to share your approach, code, and results on the Zindi platform. This will allow you to actively engage in the challenge and compare your model's performance with other participants.

The available files for download include:

Train.csv: Contains labeled tweets that can be used for training your model.
Test.csv: Contains tweets that you need to classify using your trained model.
SampleSubmission.csv: Serves as an example of the submission file format. Ensure that the ID names are correct, and the 'label' column values range between -1 and 1.
NLP_Primer_twitter_challenge.ipynb: This starter notebook can assist you in making your initial submission for the challenge.
The train dataset consists of the following features:

tweet_id: A unique identifier for each tweet.
safe_tweet: The text content of the tweet with sensitive information (usernames, URLs) removed.
label: The sentiment of the tweet, with -1 representing negative sentiment, 0 for neutral, and 1 for positive sentiment.
agreement: The percentage of agreement among the three reviewers for the given label.

Modelling
![Machine Learning Process](https://raw.githubusercontent.com/msambaraju/blog-usa/master/images/2019/03/Machine_Learning.png)

Pre-trained Models and Fine-tuning
![Pre-trained] (https://devblogs.nvidia.com/wp-content/uploads/2020/03/solution-architecture.png)

The sentiment classification models used for analyzing sentiment in vaccination-related tweets are
1. ROBERTA
2. BERT 
3. DISTILBERT
These models have been fine-tuned specifically for sentiment classification and are capable of understanding the context of language and extracting meaningful representations from text.

To utilize these models, the chosen model is loaded based on the user's preference using the corresponding identifier from the Hugging Face library. During the modeling process, parameter tuning and fine-tuning techniques were employed to optimize the models' performance.

Hyperparameters such as learning rate, batch size, and number of epochs were adjusted to achieve the best possible results in accurately classifying tweets into positive, neutral, or negative sentiment categories regarding COVID-19 vaccinations. The ultimate goal is to effectively categorize tweets based on sentiment while ensuring the analysis is conducted in the English language.

![hyperparameter Tunning](https://images.akira.ai/glossary/akira-ai-hyperparameter-tuning-ml-models.png)


Evaluation
RMSE is computed by taking the square root of the mean of the squared differences between the predicted values and the actual values.
It offers a single numerical value that reflects the overall performance of the model, where lower values indicate higher accuracy.
The RMSE metric enables straightforward interpretation of prediction errors in the same unit as the target variable. This means that the RMSE value can be directly related to the scale of the variable being predicted, making it easier to assess the magnitude of the model's errors.

Deployment

To deploy the model, follow these steps outlined here

https://github.com/Preencez/Sentiment-Analysis-LP-5

huggingface.co/Preencez/finetuned-Sentiment-classfication-ROBERTA-model

Contact 
For any inquiries or questions regarding the project, you can contact:

Name: Faith Berida

Role: Data Analyst

Organization: Azubi Africa

LinkedIn: https://www.linkedin.com/in/faith-toyin-berida-513097a2/

Medium:https://princesstoy07.medium.com/analyzing-sentiment-analysis-on-public-opinion-regarding-covid-vaccines-using-pre-trained-75910ddfae31