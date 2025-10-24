# TODOs

- [ ] Ask the organizers if the data augmentation done (column `lgbt`) is considered unconstrained or constrained.
- [ ] Try different BERT models for different languages (each language is a separate submission)
- [ ] Deep dive into feature fusing:
  - [ ] Is it possible to fuse an additional information source?
  - [ ] Is gating the best way to fuse informations?
- [ ] LGBT recognition pipeline:
  - [ ] the `lgbt` column is LLM generated. Can it be improved? (majority voting between LLM). Right now DeepSeek assigned the label
  - [ ] Most tweet present irony, and DeepSeek is not able to detect irony well (eg. "sisi, se tu sei ricco allora io sono etero"). This kind of sentences are labelled as non-lgbt. Is there a way to detect irony?

...
