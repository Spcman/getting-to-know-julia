---
title: "What is Machine Learning"
permalink: /what-is-machine-learning/
header:
  image: "/images/ai-s.jpeg"
---
Machine Learning is subset of the field of Artificial Intelligence where computer systems are used to perform a specific task without given explicit instructions.  

>In other words, with machine learning you don’t tell the program the rules, instead you give the machine learning algorithm lots of examples and it learns how to do the task on it’s own.

Here’s a picture to help explain where machine learning lies with relation to other terms you’ve possibly heard of.

![output]({{ site.url }}{{ site.baseurl }}/images/ml-terms.png)

…. And the terms don’t stop.  Here are more machine learning terms and jargon.

![output]({{ site.url }}{{ site.baseurl }}/images/ml-jargon.png)

There’s no need to worry about these for now, you’ll pick them up if you choose to dig deeper into machine learning.

## Machine Learning can be traced back to 1959, why is AI taking off right now?

<style>
#myDIV {
  width: 100%;
  padding: 10px 0;
  text-align: left;
  background-color: lightblue;
  margin-top: 20px;
  display: none;
}
</style>

<button onclick="myFunction()">Reveal Answers</button>

<div id="myDIV">
	<ul>
		<li>Copious amounts of data</li>
		<li>Computing power</li>
	</ul>
</div>

<script>
function myFunction() {
  var x = document.getElementById("myDIV");
  if (x.style.display === "none") {
    x.style.display = "block";
  } else {
    x.style.display = "none";
  }
}
</script>


## What is machine learning is used for?

Machine learning is currently limited to fairly narrow tasks. Here are a few applications of machine learning.

### Vision

Not that long ago some scientists thought a computer could never distinguish images of cats from dogs and now this is considered pretty trivial. With YOLO you can detect multiple objects in real-time.

{% include video id="MPU2HistivI" provider="youtube" %}

Other vision machine learning examples are facial recognition, medical diagnosis, self-driving cars, inspecting sites from drones, law enforcement (Clearview AI).

### Audio

Perhaps the most obvious example of audio machine learning is speech recognition. All major commercial digital assistants use deep learning; Apple Siri, Amazon Alexa, Google Assistant.

This example show’s an AI assistant (Google Duplex) booking an appointment and having a ‘real’ conversation.

{% include video id="lXUQ-DdSDoE" provider="youtube" %}

0:55

Not my work but there is now a trained machine learning model that can be used to take any song as an mp3 and split it into two new audio files, one being the instrumental and the other being the vocals.  The model can be used for free in my preferred development environment Google Colab. 

### Recommendation Systems 

Ever wondered how Google, Amazon and Netflix seem to know what you might want to buy or watch next.  All of these companies use machine learning to work out what you’re likely to want based on your prior buying, search and watching history. Part of the reason these tech giants were (and still are) so successful is down to their investment in machine learning research, but **machine learning can be used in any sized business right now.**

### Tabular Data

Tabular data is basically spreadsheet data.  To build a machine learning model from tabular data you need data that might look something like the arrangement below. It’s possible to learn from many different data types, both numeric and categorical.  The dependant variable is the outcome you’re trying to learn. In this case policy_lapse is 1 if the insurance policy lapses otherwise it is 0.

![output]({{ site.url }}{{ site.baseurl }}/images/tabular-binary-data.png)

To ‘train’ the machine learning model you need as many examples as possible.  The features should be relevant to the outcome that you’re trying to predict.

The outcome will be a mathematical ‘model’ that can make new predictions on new data rows it hasn’t seen before.

![output]({{ site.url }}{{ site.baseurl }}/images/tabular-model-new-prediction.png)

A good tabular dataset for a beginner data scientist is the [Titanic Kaggle Competition](https://www.kaggle.com/c/titanic)

These problems are known as binary classification problems as the outcome is 1 or 0.  We can also build models to predict discrete continuous numbers.  If we started out from this dataset, we could train the model to predict the premium.  This type of problem is called a regression problem.

![output]({{ site.url }}{{ site.baseurl }}/images/tabular-regression.png)

The final type of tabular data problem we’ll look at is the multiclass problem.  This time the dependant variable can be classified a one of a set number of regions.  This type of problem is a Multiclass Classification problem.

![output]({{ site.url }}{{ site.baseurl }}/images/tabular-multiclass.png)

## How Does Machine Learning Work?

The backbone of AI is basically math. Specifically, Calculus, Linear Algebra, Probability and Statistics.  So firstly all data whether it is visual, audio or tabular must be converted entirely into numbers.  Let's say we were teaching the computer how to recognise hand-written digits.  Here’s how we would convert the greyscale image '8' into numbers.

![output]({{ site.url }}{{ site.baseurl }}/images/mnist-digit-8.png)

The next step is to choose a machine learning algorithm and train the model.  A Neural Network (Deep Learning) is one of the most common techniques used.

As the model trains it is basically nudging weights and biases around to minimise the error (or loss).  This video demonstrates the learning process in a Neural Network.

{% include video id="Ilg3gGewQ5U" provider="youtube" %}

Despite the complexity under the hood Python libraries such as Google’s Tensorflow and Facebook’s PyTorch make it possible for anyone to build deep learning machine learning models. 

## Text Data (Natural Language Processing)

We haven’t yet looked at text data.  Computers are great at storing text data but they don’t ‘understand’ it as such. We can now relatively easily train machine learning models to recognize the sentiment of text.  Probably the most famous dataset used by newbies for this task is the IMDB movie reviews dataset. This dataset contains over 50,000 reviews. Each review has been manually labelled as a positive or negative review. If we train a machine learning model on this data we can take a new review and work out if it is positive or negative.

The [Tensor Flow embedding projector](https://projector.tensorflow.org/) is a cool way to introduce the concept of Word Embeddings. After training a model on the IMDB data set we can visually see how the model has determined what words might lean towards positive and those that might lean towards negative.

Pre-trained word embeddings can greatly improve the accuracy of text based machine learning models. There are ones freely available to use from Google (Word2Vec), Facebook (fastText) and Stanford University (GloVe). I wrote a ‘fun’ article called [Word Embedding with Dracula](https://spcman.github.io/getting-to-know-julia/nlp/word-embeddings/) on my blog.

Other example using text data are Spam detectors, chat bots and the automatic detection of social media posts to see if they are abusive in some way or not.

## Work related projects

**Project 1 : Can we use the subject and note text to predict the correct note type and sub type of notes not seen in the training process?**

Data is real but has been anonymized.

**[Project 2](https://spcman.github.io/getting-to-know-julia/monte%20carlo/monte-carlo-investment-earnings/): The second work related note book was written in Julia. It’s not machine learning such, but more a look at monte-carlo simulation on portfolios using risk profiles similar to ours.**

## Where are we heading?

Like it or not AI is going to have more and more impact on our daily lives. Respected thinkers liken the changes to a 'second industrial revolution' or a 'new electricity'. This time, however, the white-collar and professional workers may be more affected.  In the medical industry we’re already seeing that AI is better than humans a detecting cancer from images. It’s likely that more general medical diagnosis will be better via AI in the not too distant future. The legal sector is also likely to be affected as tasks done by paralegals will soon be done really well using AI too … and this is just the start.

AI has proved ‘better’ than human intelligence at a number of tasks most famously at the game of Go.  The story behind Move 37 is quite fascinating.

{% include video id="vI9BllT7ovg" provider="youtube" %}

You can see the entire documentary here

{% include video id="WXuK6gekU1Y" provider="youtube" %}

Although AI can already be ‘better’ than humans at a number of tasks it is still limited to the individual tasks that it has been trained to do.  Artificial General Intelligence (AGI) is a term used to describe a machine that has the capacity to learn any intellectual task that a human can do. When invented this will be the tipping point of something really big! It’s anticipated that such a machine would quickly learn to be more intelligent that humans having a status of ‘superintelligent’. It’s something that troubles many scientists as there is no limit on intelligence and what that that might mean.  Nobody knows how far this is actually away but it could less than 60 years away....

## The dark side of AI

Aside from the obvious ‘Terminator’ scenario, deep fakes videos are relatively easy to make right now.

{% include video id="gLoI9hAX9dw" provider="youtube" %}

## Machine Learning in my business

Aimed at business execs, Andrew Ng did this talk suggesting that now is the time for businesses to start looking at AI use-cases to gain competative advantage.  I did Andrew's brilliant course on Machine Learning a while ago.

{% include video id="j2nGxw8sKYU" provider="youtube" %}

1.17min
