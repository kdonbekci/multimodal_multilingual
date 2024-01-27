The repository is structured as follows:

The directory `data/` contains the pickle files with the cached metadata of both [CMU-MOSEI](http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/) and [CH-SIMS](https://aclanthology.org/2020.acl-main.343/) datasets. The notebook `Dataset Preparation.ipynb` contains illustrative example of how to extract the Self-supervised Embeddings used in the project ([Wav2Vec2-XLS-R](https://huggingface.co/facebook/wav2vec2-xls-r-300m), [XLM-RoBERTa](https://huggingface.co/FacebookAI/xlm-roberta-large), [FAb-Net](https://www.robots.ox.ac.uk/~vgg/research/unsup_learn_watch_faces/fabnet.html)). 

The raw video files are currently stored on the Kedi server in the Spoken Language Processing Lab.
```
/local/speech/data/CMU-MOSEI.tar
/local/speech/data/CH-SIMS.tar
```
Downloading and extracting the tarballs inside the `data/` directory should allow you to run the notebook without trouble. You might have errors instantiating the HuggingFace models, it might help deleting the appropriate directory inside `models/` and rerunning to force a fresh download of the checkpoint.

The directory `Self-Supervised-Embedding-Fusion-Transformer` contains the source code for a project that our approach is based off of. Specifically, the Inter-Modality Attention (IMA) block that fuses contextualized embeddings by appending a trainable `CLS` token to the audio and visual embeddings, and using said token's embeddings to efficiently extract an utterance-level representation first independently from each modality and then fused with other modalities. 

The Jupyter notebook `Modeling.ipynb` has a proof-of-concept of this IMA layer as well as other parts of the architecture. Currently I am working on the notebook that trains the model on the actual datasets. 

Feel free to reach out to me at kd2939@columbia.edu 

