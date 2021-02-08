# NLP-2020-EmotionGIF
Source code of competition in **[EmotionGIF 2020 (ACL 2020 Workshop)](https://sites.google.com/view/emotiongif-2020/home?authuser=0)**.
We won the third place and report can be referred [here](https://drive.google.com/file/d/1eLZHL8JqSSlwX43e-IXu352Xz0hMe3ME/view).
## Challenge
Given the labeled training data, you will need to recommend GIF categories for all the tweets in the unlabeled evaluation dataset. 
For each tweet, you need to recommend exactly 6 categories. 
## Dataset
Our dataset is collected from Twitter, and includes 40K samples. 
We provide three files: training data, development data that can be used during practice, and evaluation data used for ranking the submissions.
## Metric
The metric that will be used to evaluate entries is Mean Recall at k, with k=6 (MR@6). 
