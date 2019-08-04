---
layout: archive
title: Up and Running! My First Post
date: 2019-08-04
categories: [blog]
tags: [Jekyll]
header:
  image: "/images/ai-s.jpeg"
---
This website is hosted for free on [GitHub Pages with Jekyll](https://help.github.com/en/articles/about-github-pages-and-jekyll)

Big shout out to Michael Rose for the theme [Minimal Mistakes](https://mmistakes.github.io/minimal-mistakes/) and DataOptimal who made this video on how to get started and put everything together.

{% include video id="qWrcgHwSG8M" provider="youtube" %}

{% for category in site.categories %}
  <h3>{{ category[0] }}</h3>
  <ul>
    {% for post in category[1] %}
      <li><a href="{{ post.url }}">{{ post.title }}</a></li>
    {% endfor %}
  </ul>
{% endfor %}