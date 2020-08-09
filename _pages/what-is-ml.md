---
title: "What is Machine Learning"
permalink: /what-is-machine-learning/
header:
  image: "/images/ai-s.jpeg"
---
Machine Learning is subset of the field of Artificial Intelligence where computer systems are used to perform a specific task without given explicit instructions.  

>The main concept to grasp with machine learning is you don’t program the rules, instead you give the machine learning algorithm lots of examples and it learns how to do the task on it’s own.

Here’s a picture to help explain where machine learning lies with relation to other terms you’ve possibly heard of.

![output]({{ site.url }}{{ site.baseurl }}/images/ml-terms.png)

…. And the terms don’t stop.  Here are more machine learning terms and jargon.

![output]({{ site.url }}{{ site.baseurl }}/images/ml-jargon.png)

There’s no need to worry about these for now, you’ll pick them up if you choose to dig deeper into machine learning.

## What is machine learning is used for?

Machine learning is currently limited to fairly narrow tasks. Here are a few applications of machine learning.

###Vision

Not that long ago some scientists thought a computer could never distinguish images of cats from dogs and now this is considered pretty trivial. With YOLO you can detect multiple objects in real-time.

{% include video id="MPU2HistivI" provider="youtube" %}

Other vision machine learning examples are facial recognition, medical diagnosis, self-driving cars, inspecting sites from drones, law enforcement (Clearview AI), 

###Audio

Perhaps the most obvious example of audio machine learning is speech recognition. All major commercial speech recognition applications use deep learning; Apple Siri, Alexa, Google Assistant.

This example show’s an AI assistant booking an appointment and having a ‘real’ conversation.

{% include video id="lXUQ-DdSDoE" provider="youtube" %}

Not my work but there is now a trained machine learning model that can be used to take any song as an mp3 and split it into two new audio files, one being the instrumental and the other being the vocals.  The model can be used for free in my preferred development tool Google Colab. 

###Recommendation Systems 

Ever wondered how Google, Amazon and Netflix seem to know what you might want to buy or watch next.  All of these companies use machine learning to work out what you’re likely to want to do based on your prior buying, search and watching history.
Part of the reason these tech giants were (and are still) so successful is down to their investment in machine learning research, but *machine learning can be used in any sized business right now.*

###Tabular Data

Tabular data is basically spreadsheet data.  To build a machine learning model from tabular data you need data that might look something like the arrangement below. It’s possible to learn from many different data types, both numeric and categorical.  The dependant variable is the outcome you’re trying to learn. In this case policy_lapse is 1 if the insurance policy lapses otherwise it is 0.

![output]({{ site.url }}{{ site.baseurl }}/images/tabular-binary-data.png)

To ‘train’ the machine learning model you need as many examples as possible.  The features should be relevant to the outcome that you’re trying to predict.

The outcome will be a mathematical ‘model’ that can make new predictions based on data it hasn’t seen before.

![output]({{ site.url }}{{ site.baseurl }}/images/tabular-model-new-prediction.png)

A good tabular dataset for a beginner data scientist is the Titanic Problem 

https://www.kaggle.com/c/titanic

These problems are known as binary classification problems as the outcome is 1 or 0.  We can also build models to predict discrete continuous numbers.  If we started out from this dataset, we could train the model to predict the premium.  This type of problem is called a regression problem.

![output]({{ site.url }}{{ site.baseurl }}/images/tabular-regression.png)

The final type of tabular data problem we’ll look at is the multiclass problem.  This time the dependant variable can be classified a one of a set number of regions.  This type of problem is a Multiclass Classification problem.

![output]({{ site.url }}{{ site.baseurl }}/images/tabular-multiclass.png)

##How Does Machine Learning Work?

The backbone of AI is basically math. Specifically, Calculus, Linear Algebra, Probability and Statistics.  So firstly all data whether it is visual, audio or tabular must be converted entirely into numbers.  Here’s an example showing how a greyscale image of a handwritten digit 8 is converted into numbers.

![output]({{ site.url }}{{ site.baseurl }}/images/mnist-digit-8.png)

The next step is to choose a machine learning algorithm and train the model.  Neural Networks or Deep Learning is one of the most common techniques used.

As the model trains it is basically nudging weights and biases around to minimise the error (or loss).  This video demonstrates the learning process in a Neural Network.

{% include video id="Ilg3gGewQ5U" provider="youtube" %}

Despite the complexity under the hood Python libraries such as Google’s Tensorflow and Facebook’s PyTorch make it possible for anyone to build deep learning machine learning models. 

##Text Data

We haven’t yet looked at text data.  Computers are great at storing text data but they don’t ‘understand’ it as such.  This is starting to change. We can now relatively easily train machine learning models to recognize the sentiment of text.  Probably the most famous dataset often used by newbies is the IMDB movie reviews dataset. This dataset is contains over 50,000 reviews. Each review has been manually labelled as a positive or negative review. If we train a machine learning model of this data we can take a new review and work out if it is positive or negative.

The Tensor Flow embedding projector is a cool way to introduce the concept of Word Embeddings. After training a model on the IMDB data set we can visually see how the model has determined what works might lean towards positive and those that might lean towards negative.

https://projector.tensorflow.org/

Pre-trained word embeddings are freely available to use from Google (Word2Vec), Facebook (FASText). I wrote a ‘fun’ study of the GloVe word embeddings.  This data set was trained on Wikipiedia text by Stanford University.

https://spcman.github.io/getting-to-know-julia/nlp/word-embeddings/

Example using text data are SPAM detectors and the automatic detection of social media posts to see if they are abusive in some way or not.

Work related projects

Project 1 : Can we use the subject and note text to predict the correct note type and sub type of notes not seen in the training process? 

Data is real but has been anonymized.

Project 2: The second work related note book was written in Julia. It’s not machine learning such, but more a look at monte-carlo simulation on portfolios using risk profiles similar to ours.

https://spcman.github.io/getting-to-know-julia/monte%20carlo/monte-carlo-investment-earnings/

## Where are we heading?

Like it or not AI is going to have more and more impact on our daily lives. Respected thinkers liken the changes to a 'second industrial revolution' or a 'new electricity'. This time, however, the white-collar and professional workers may be more affected.  In the medical industry we’re already seeing that AI is better than humans a detecting cancer from images. It’s likely that more general medical diagnosis will be better via AI in the not too distant future. The legal sector is also likely to be affected as tasks done by paralegals will soon be done really well using AI too … and this is just the start.

AI has proved ‘better’ than human intelligence at a number of tasks most famously at the game of Go.  The story behind Move 37 is quite fascinating.

{% include video id="vI9BllT7ovg" provider="youtube" %}

Although AI can already be ‘better’ than humans at a number of tasks it is still limited to the individual tasks that it has been trained to do.  Artificial General Intelligence (AGI) is a term used to describe a machine that has the capacity to learn any intellectual task that a human can do. When invented this will be the tipping point of something really big! It’s anticipated that such a machine would quickly learn to be more intelligent that humans having a status of ‘superintelligent’. It’s something that troubles many scientists as there is no limit on intelligence and what that that might mean.  Nobody knows how far this is actually away but it could less than 60 years away....

## The dark side of AI

Aside from the obvious ‘Terminator’ scenario, deep fakes videos are relatively easy to make right now.

{% include video id="gLoI9hAX9dw" provider="youtube" %}

These videos are not restricted to World leaders, soon someone could be impersonating you!

## Machine learning must be easy?

Unfortunately, not quite, well at least not yet!

Getting the models to work requires a fair bit of skill and experimentation. As a machine learning practitioner you really need to be able to write code to run ‘experiments’.  You’ll ideally need to have a good grasp of math too; in particular the areas of linear algebra, statistics, pobability and calculus.  There are also many algorithms, terms, languages, libraries to choose from and to know how to use properly. If you have an idea for a machine learning project sometimes data may also become a barrier.  Accurate models generally need many training data examples to learn the task.  For most applications you generally need over 500 labelled examples in a data set for a decent prediction accuracy.

...But learning AI is certainly doable and once you get started it’s a real positive addiction to see what you can learn and do!

About the math! Don’t be put off by this, you can get started and build pretty amazing models without knowing much math at all.  Start at the top and work down to the math later if you like.

## Machine Learning in my business

I recently stumbled on this talk by Andrew Ng.  I did his brilliant course on Machine Learning a while ago.  He explains how now may be the time to try out some use-cases in your own business.

{% include video id="j2nGxw8sKYU" provider="youtube" %}


