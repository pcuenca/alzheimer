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

## References

* Becker, J. T., Boller, F., Lopez, O. L., Saxton, J., & McGonigle, K. L. (1994). The natural history of Alzheimer's disease: description of study cohort and accuracy of diagnosis. Archives of Neurology, 51(6), 585-594.    
* Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv    preprint arXiv:1810.04805.    
* Howard, J., & Ruder, S. (2018). Universal language model fine-tuning for text classification. arXiv preprint arXiv:1801.06146.    
* Hernández-Domínguez, L., Ratté, S., Sierra-Martínez, G., & Roche-Bergua, A. (2018). Computer-based evaluation of Alzheimer’s disease and mild cognitive impairment patients during a picture description task. Alzheimer's & Dementia: Diagnosis, Assessment & Disease Monitoring, 10, 260-268.     
* López-de-Ipiña, K., Alonso, J. B., Barroso, N., Faundez-Zanuy, M., Ecay, M., Solé-Casals, J., ... & Ezeiza, A. (2012, December). New approaches for Alzheimer’s disease diagnosis based on automatic spontaneous speech analysis and emotional temperature. In International Workshop on Ambient Assisted Living (pp. 407-414). Springer, Berlin, Heidelberg.    
* MacWhinney, B. (2000). The CHILDES Project: Tools for Analyzing Talk. 3rd Edition. Mahwah, NJ: Lawrence Erlbaum Associates.    
* Mirheidari, B., Blackburn, D., Reuber, M., Walker, T., & Christensen, H. (2016, September). Diagnosing people with dementia using automatic conversation analysis. In Proceedings of Interspeech (pp. 1220-1224). ISCA.    
* Mirheidari, B., Blackburn, D., Walker, T., Venneri, A., Reuber, M., & Christensen, H. (2018). Detecting Signs of Dementia Using Word Vector Representations. Proc. Interspeech 2018, 1893-1897.
* Mirheidari, B., Blackburn, D., Walker, T., Reuber, M., & Christensen, H. (2019). Dementia detection using automatic analysis of conversations. Computer Speech & Language, 53, 65-79.    
* Yancheva, M., Fraser, K., & Rudzicz, F. (2015). Using linguistic features longitudinally to predict clinical scores for Alzheimer’s disease and related dementias. In Proceedings of SLPAT 2015: 6th Workshop on Speech and Language Processing for Assistive Technologies (pp. 134-139).    


## Participants

- Daniel Amador.
- Jesús Gómez.
- Pedro Cuenca.

