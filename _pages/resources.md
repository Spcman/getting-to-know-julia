---
layout: archive
permalink: /resources/
title: "Julia Resources"
author_profile: true
tags: [julia, resources]
header:
  image: "/images/resources.jpeg"
---

{% include base_path %}
{% include group-by-array collection=site.posts field="tags" %}

{% for tag in group_names %}
  {% assign posts = group_items[forloop.index0] %}
  <h2 id="{{ tag | slugify }}" class="archive__subtitle">{{ tag }}</h2>
  {% for post in posts %}
    {% include archive-single.html %}
  {% endfor %}
{% endfor %}

Add resources for Julia!!!!