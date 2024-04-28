import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from collections import Counter
import cmd

def plot_distribution(series, plt_title, figsize, bins=14):
  plt.figure(figsize = figsize)
  plt.title(plt_title)
  series = np.array(series, dtype=int)
  sn.distplot(series, hist=True, bins=bins, kde=True, kde_kws={'bw':.2})
  plt.axvline(series.mean(),color='midnightblue',label='Mean')    
  plt.axvline(np.median(series),color='blue',label='Median')
  plt.axvline(series.max(),color='indigo',label='Max')
  plt.axvline(series.min(),color='crimson',label='Min')
  plt.axvline(np.quantile(series, 0.25),color='red',label='First quartile - 25%')
  plt.axvline(np.quantile(series, 0.75),color='orangered',label='Third quartile - 75%')
  plt.legend()
  plt.show()

def stripplot(data, title, figsize):
  plt.figure(figsize=figsize)
  plt.title(title)
  y = data.columns[0]
  x = data.columns[1]
  sn.stripplot(x=x, y=y, data=data, jitter=False, dodge=True)
  sn.boxplot(x=x, y=y, data=data)
  plt.xticks(rotation = 90)
  plt.show()

def print_count(series, title, show=True):
  counts = Counter(series)
  countlist = []
  for c in counts:
    countlist.append(str(c) + ':' +str(counts[c]))
  if show:
    print('\n'+title)
    cmd.Cmd().columnize(countlist, displaywidth=80)

  return counts

def piecountplot(series, title):
  plt.title(title)
  series.value_counts().plot(kind='pie', autopct='%1.2f%%')
  plt.legend()

def piepurchaseplot(dictionary, title):
  plt.title(title)
  plt.pie(dictionary.values(), labels = dictionary.keys(), autopct='%1.2f%%')
  plt.legend()

def countplot(series, title, figsize):
  plt.figure(figsize=figsize)
  plt.title(title)
  sn.countplot(series)
  plt.show()
  
def barplot(series, title, figsize):
  plt.figure(figsize=figsize)
  plt.title(title)
  plt.bar(series.index, series.values, color="royalblue")
  plt.show()

def allplots(data, title, kind='countplot', figsize=(10,2.5)):
  if kind == 'countplot':
    countplot(data, title, figsize)
  elif kind == 'piecountplot':
    piecountplot(data, title)
  elif kind == 'dist_plot':
    plot_distribution(data, title, figsize)
  elif kind == 'barplot':
    barplot(data, title, figsize)
  elif kind == 'piepurchaseplot':
    piepurchaseplot(data, title)
  elif kind == 'stripplot':
    stripplot(data, title, figsize)