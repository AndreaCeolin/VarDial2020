== Data Format ==

The training data contains the following files:

	train.txt - training set
	train.labels - training labels
	dev-source.txt - development/validation set (with samples from the source genre: news)
	dev-source.labels - development/validation labels (for samples from the source genre: news)
	dev-target.txt - development/validation set (with samples from the target genre: tweets)
	dev-target.labels - development/validation labels (for samples from the target genre: news)

Each line in the *.txt files is tab-delimited in the format:

	text-sample<tab>dialect-label

Each line in the *.labels files is in the format:

	dialect-label

== Task Description ==

In the Romanian Dialect Identification (RDI) shared task, participants have to train a model on samples collected the news domain and evaluate it on tweets. Therefore, participants have to build a model for a cross-genre binary classification by dialect task, in which a classification model is required to discriminate between the Moldavian (MD) and the Romanian (RO) dialects across different text genres (news versus tweets). The task is closed, therefore, participants are not allowed to use external data to train their models.

For training, we provide participants with the MOROCO data set [1] which contains Moldavian (MD) and Romanian (RO) samples of text collected from the news domain. The training set contains 33564 samples.

The development set is composed of two parts:
	
	- a development set of 5923 samples from the source genre (news), which was used as the private test set in the MRC Shared Task of VarDial 2019.
	- a development set of 215 samples from the target genre (tweets).

The training and development samples from the source genre are also available at https://github.com/butnaruandrei/MOROCO. The repository includes some code to load the data (in slightly different format) and to evaluate the results / produce confusion matrices.

For the VarDial 2020 evaluation campaign, the test set is formed of a new set of samples, which is not part of MOROCO. The samples are collected from Twitter.

All samples are preprocessed in order to replace named entities with a special tag: $NE$.

References:
[1] Andrei M. Butnaru, Radu Tudor Ionescu. MOROCO: The Moldavian and Romanian Dialectal Corpus. In proceedings of ACL, pp. 688-698, 2019.

== Evaluation ==

The test data (to be released later) will only contain sentences without their dialect labels. Participants will be required to submit the labels for these test instances. The exact details of the submission file format will be provided later.
