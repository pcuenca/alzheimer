# Early Detection of Alzheimer's Disease

Bringing detection closer to people.

This project was undertaken as part of the AI Saturdays sessions on machine learning & deep learning that took place in Madrid in 2019.

Our goal was to assess different models, combine them when appropriate and make a simple iOS app that could potentially be used by real people in the privacy of their homes.

## Dataset

We have used the Pitt dataset from `DementiaBank` in the `TalkBank` database. Specifically, we have trained our models with the `Boston Cookie Theft` image description task, since that's the most balanced subset.  The dataset consists of 552 recordings with accompanying transcripts(*), all of them describing the test image in the context of a clinical interview with each participant.

(*) One of the transcripts does not correspond with the voice recording and was excluded from the set, making it a total of 551 interviews.

## Language models

Several NLP models were tried, with and without syntactic tagging and markup annotations. Algorithms we tried include:

- SVM
- NB-SVM
- Random Forest
- ULMFiT
- BERT
- Ad-hoc CNN

[Results summary to be provided]

## Audio models

The recordings were processed to isolate the intervention of participants from those of interviewers. Spectrograms were produced and then fed to a CNN network (xResNet-34).

The audio classifer was considered as a standalone classifier. Unfortunately we ran out of time and could not create a combined classifier with both text and sound data. That's one of the lines of work we are considering for a future update.

## iOS App

The iOS App displays the Cookie Theft image and invites users to describe it. Their voice is converted to text and fed into a NLP classifier. All operations are carried out inside the app and no data is sent to any cloud service.

## Next Steps

- We believe the availability of data to be the main bottleneck. Any serious attempt to provide useful tools for real people will be contingent on acquiring much more data. We plan to contact medical institutions and non-profit organizations to set up a data collection task. A single picture description task cannot be generalized to a general-purpose classification analysis, we need more robust databases.
- Combine audio with text. This could start with something simple such as introducing pause information into the text stream.
- Find an efficient way to convert complex models such as ULMFiT to CoreML. The path Pytorch -> ONNX -> CoreML is tricky and unreliable.
- Use LSTMs in addition to CNNs for audio.
- Evaluate sensible audio data augmentation transformations. Deal with recordings with high length variance.
- Disseminate results to increase awareness and try to set up pilot tests in controlled environments. These could help in the data acquisiton front.

## Participants

- Daniel Amador.
- Jesús Gómez.
- Pedro Cuenca.
