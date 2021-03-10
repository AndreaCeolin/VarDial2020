# VarDial2020

This folder contains the material developed for the RDI shared task at VarDial 2020 by Team Phlyers, and described in:

> Ceolin, A. & Zhang, H. (2020). *Discriminating between standard Romanian and Moldavian tweets using filtered character ngrams*. In Proceedings of the 7th Workshop on NLP for Similar Languages, Varieties and Dialects, 265-272.

Here is a list of the files.

1. **data**: This folder contains the training and development data of the shared task, which are presented and described in Butnaru and Ionescu (2019) and Găman et al. (2020).

2. **models.py**: This file contains the Naive Bayes and SVM models developed for the task.

3. **eval_news.py**: This script runs the models developed for the task on the news development set. 

4. **eval_tweets.py**: This script runs the models developed for the task on the tweets development set.

5. **cnn.ipynb**: This Jupyter Notebook contains the CNN developed for the task.

6. **trained_cnn_model**: This file contains the parameters of the CNN.

7. **CeolinZhang2020.pdf**: This is the proceedings paper that describes the task and our contribution.

Note that some of the numbers produced by the script are greater than those contained in the proceedings paper. This is mostly due to some bugs in the TFIDF transformation that was fixed only after the submission. However, this only affected the models that were not used to participate in the task.

References:

> Butnaru, A. M., & Ionescu, R. T. (2019). MOROCO: The Moldavian and Romanian dialectal corpus. arXiv preprint arXiv:1901.06543.

> Gaman, M., Hovy, D., Ionescu, R. T., Jauhiainen, H., Jauhiainen, T., Linden, K., Ljubešić, N., Partanen, N., Purschke, C., Scherrer, Y. and Zampieri M.  (2020). A report on the VarDial evaluation campaign 2020. In Proceedings of the 7th Workshop on NLP for Similar Languages, Varieties and Dialects. International Committee on Computational Linguistics.








