# 16S_sentence_embedding_taxonomy
Comparison between qiime2 methods of classifying 16S reads and NLP sentence embedding projection. The objective is to keep relative performance the same but substantially reduce the time to classify reads.

Sure! Here's an example README file for a Jupyter notebook that uses NLP embeddings to classify 16S rRNA sequences:

---

# 16S rRNA Sequence Classification Using NLP Embeddings

This Jupyter notebook demonstrates how to use natural language processing (NLP) embeddings to classify 16S rRNA sequences based on their similarity to known reference sequences.

## Installation

Before running this notebook, you will need to install the following Python packages:

- `numpy`
- `pandas`
- `scikit-learn`
- `sentence-transformers`

You can install these packages using `pip`:

```
pip install numpy pandas scikit-learn sentence-transformers
```

## Usage

To use this notebook, you will need a file containing 16S rRNA sequences in FASTA format, as well as a reference database of 16S rRNA sequences and their taxonomic classifications.

1. Import the necessary Python packages:

   ```
   import numpy as np
   import pandas as pd
   from sklearn.metrics.pairwise import cosine_similarity
   from sentence_transformers import SentenceTransformer
   ```

2. Load the 16S rRNA sequences into a pandas DataFrame:

   ```
   df = pd.read_csv('16S_sequences.fasta', header=None, names=['id', 'sequence'])
   ```

3. Load the reference database into a pandas DataFrame:

   ```
   ref_db = pd.read_csv('reference_database.csv')
   ```

4. Use a pre-trained NLP embedding model to encode the 16S rRNA sequences and reference sequences as dense vectors:

   ```
   model = SentenceTransformer('bert-base-nli-mean-tokens')
   df['embedding'] = model.encode(df['sequence'].tolist())
   ref_db['embedding'] = model.encode(ref_db['sequence'].tolist())
   ```

5. For each 16S rRNA sequence, find the most similar reference sequence based on cosine similarity:

   ```
   similarities = cosine_similarity(df['embedding'].tolist(), ref_db['embedding'].tolist())
   max_similarities = np.amax(similarities, axis=1)
   max_indices = np.argmax(similarities, axis=1)
   df['most_similar_index'] = max_indices
   df['most_similar_similarity'] = max_similarities
   df = df.join(ref_db.set_index('id'), on='most_similar_index')
   ```

6. Compare the predicted taxonomic classifications to the actual classifications using Qiime2 tools or other relevant software.

## Credits

This notebook was created by Tyler Myers. The pre-trained NLP embedding model was developed by the [sentence-transformers](https://github.com/UKPLab/sentence-transformers) project.
