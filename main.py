from transformers import pipeline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#model = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")


class Sentiment_tweets:
    def __init__(self,name_file):
        self.tweets = pd.read_excel(f'{name_file}.xlsx')['Tweet']
        self.model = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")
        self.inference = self.__inference__()

    def __inference__(self):
       labels = self.model(self.tweets.tolist())
       return [label['label'] for label in labels]

    def plot(self):
        labels = pd.Series(self.inference).value_counts()
        fig, ax = plt.subplots(figsize =(16, 20))
        ax.barh(labels.keys(), labels)
        plt.show()


if __name__ == '__main__':
    Sentiment_tweets('Data Analysis').plot()
