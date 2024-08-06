import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification


# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')


# Functions
def load_data(path):
    """Read the data text from a given path"""
    with open(path, 'r', encoding = 'utf-8') as f:
        lines = f.readlines()
    return lines

def tokenizer_text(text):
    """Tokenize the text using padding and truncation so that the tokens have the same length"""
    return tokenizer(text,
                     padding = 'max_length',
                     truncation = True,
                     max_length = 512,
                     return_tensors='tf'
                     )

# Initialize path to the corpus text
CORPUS_TEXT_PATH = '/Users/kenlam/Desktop/Data science/ML projects/RAG_resume/model/corpus_video.txt'


# Tokenize the texts 
raw_texts = load_data(CORPUS_TEXT_PATH)
tokenized_texts = [tokenizer_text(text) for text in raw_texts] 

# Transform these tokens into Tensorflow's DataSet
def to_tf_dataset(tokenized_texts):
    """Transform the tokens into Tensorflow's Dataset type"""
    input_ids = [text['input_ids'] for text in tokenized_texts]
    attention_mask = [text['attention_mask'] for text in tokenized_texts]

    dataset = tf.data.Dataset.from_tensor_slices(({
        'input_ids': tf.convert_to_tensor(input_ids),
        'attention_mask': tf.convert_to_tensor(attention_mask),
    }))

    return dataset.shuffle(1000).batch(4)

train_dataset = to_tf_dataset(tokenized_texts)