# LT2212 V20 Assignment 1

Put any documentation here including any answers to the questions in the 
assignment on Canvas.

General:
n = 1000(in the test groud) n = 1 is the default value that Asad added
m = 5



Answer for Part3:
After applying td-idf method, it can be observed especially with help of visualization that the distances between the most of the data pairs become less. Although there is no change in absolute relationships.
Considered the way how td-idf funtions, it is intuitive that the two factors: td and idf are two competing factors and restrict each other to mitigate the effect of extrem data fragments. 

Anser for Bonus part:
In this part, I applied the MultinomialNB() model to train the classifier. Naive Bayes turns out to be ideal for this task. Both with frequncy data and td-idf data, a accuracy of over 0.9 can be reached. By each shuffle and retraining, the performance varies slightly. There is no significant difference in terms of performance when use both approach. 

