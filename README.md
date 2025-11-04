# TODOs

- [x] Ask the organizers if the data augmentation done (column `lgbt`) is considered unconstrained or constrained. 
> It is considered constrained
- [x] Try different BERT models for different languages (each language is a separate submission)
> I tried a spanish tuned model for the spanish language but it underperforms wrt to the multilingual (0.619 F1 vs 0.740)
> I tried with an italian specific model for the italian language. This improved the performances (0.9672 F1 vs 0.9280)
- [ ] Deep dive into feature fusing:
  - [ ] Is it possible to fuse an additional information source?
  - [ ] Is gating the best way to fuse informations?
- [ ] LGBT recognition pipeline:
  - [ ] the `lgbt` column is LLM generated. Can it be improved? (majority voting between LLM). Right now DeepSeek assigned the label
  - [ ] Most tweet present irony, and DeepSeek is not able to detect irony well (eg. "sisi, se tu sei ricco allora io sono etero"). This kind of sentences are labelled as non-lgbt. Is there a way to detect irony?

## Results analysis
### Italian language
The italian language can be considered as "solved" as only very few examples are misclassified (4 out of 326). To further improve classification I suggest looking into the LLM labeling, by changing the prompt or by introducing majority voting. We could also analyze when the model missclassify the tweet, as it could also be a problem of unclear labelling in the provided dataset. For example
`"@USER @USER Per qualcuno sono gay,per altri transgender, per latri ancora LGBT. Per so froci e rottinculo! Bloccatemi vigliacchi di Twitter"`
This tweet is labelled as reclamatory, but our model predicts it isn't.

We know that when the model missclassify, it is very sure about its answer. For example the confidence for the previous example is `0.989`

### Spanish language
The spanish language performs worse, but no actual studies has been done on it yet
