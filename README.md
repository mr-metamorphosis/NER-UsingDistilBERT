# Named Entity Recognition (NER) with Hugging Face Transformers

This repository provides an example of how to fine-tune a pre-trained Transformer model for Named Entity Recognition (NER) using the Hugging Face Transformers library. The dataset used in this example is the CoNLL-2003 dataset, which includes annotations for four types of entities: Person, Organization, Location, and Miscellaneous.

## Table of Contents

1. [Installation](#installation)
2. [Dataset Overview](#dataset-overview)
3. [Code Explanation](#code-explanation)
4. [Model Training](#model-training)
5. [Evaluation and Results](#evaluation-and-results)
6. [Usage](#usage)
7. [File Structure](#file-structure)
8. [Acknowledgments](#acknowledgments)
9. [License](#license)

## #Installation

To get started, you need to install the required libraries:

```bash
pip install -U transformers
pip install -U accelerate
pip install -U datasets
pip install seqeval
pip install evaluate

### 1. Installation

To get started, you need to install the required libraries:
bash
```
!pip install -U transformers
!pip install -U accelerate
!pip install -U datasets
!pip install seqeval
!pip install evaluate
```

### 2. Dataset Overview

The CoNLL-2003 dataset is used in this example. It is a standard dataset for NER tasks, containing the following entity types:
PER: Person
ORG: Organization
LOC: Location
MISC: Miscellaneous
The dataset is available in the datasets library and can be loaded easily with the following command:

```
from datasets import load_dataset
data = load_dataset('conllpp')
```
### 4. Code Explanation

Data Preparation
      <li> 4.1 Load and Explore the Dataset:
      ``` 
      data = load_dataset('conllpp')
      ```
     <li> 4.2 Inspect the Dataset Structure:
      ```
      data['train'].features
      ```
     <li> 4.3 Convert NER Tags to Human-Readable Format:
       - Define mapping from tag indices to tag names and vice versa.
      ```
      tags = data['train'].features['ner_tags'].feature
      index2tag = {idx:tag for idx, tag in enumerate(tags.names)}
      tag2index = {tag:idx for idx, tag in enumerate(tags.names)}\
      ```
      <li> 4.4 Add Readable NER Tag Names:
       - Create a new column with human-readable NER tag names.

       ```
       def create_tag_names(batch):
          tag_name = {'ner_tags_str': [tags.int2str(idx) for idx in batch['ner_tags']]}
          return tag_name
       data = data.map(create_tag_names)

       ```
Then proceeded with the model building that includes 
- Tokenization
- Aligning Labels with Tokens
- Tokenize and Align Labels

then,
Creating a Data Collation and Metrics system :
<ol type ='num'> 
<li> Data Collation: 
      <ul><li> Use a data collator for token classification to handle padding and batching. </li></ul>
</li>

       



