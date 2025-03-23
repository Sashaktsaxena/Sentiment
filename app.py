

from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Initialize FastAPI app
app = FastAPI()

# Load Model and Tokenizer
MODEL_PATH = "best_model1.h5"
TOKENIZER_PATH = "tokenizer.pickle"

print("Loading model and tokenizer...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as handle:
    tokenizer = pickle.load(handle)

print("Model and tokenizer loaded successfully!")

class TextInput(BaseModel):
    text: str

def test_sentiment(text, max_len=150):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
    prediction = float(model.predict(padded, verbose=0)[0][0])

    sentiment = "Positive" if prediction >= 0.5 else "Negative"
    confidence = prediction * 100 if prediction >= 0.5 else (1 - prediction) * 100

    return {"sentiment": sentiment, "confidence": f"{confidence:.2f}%"}

@app.post("/predict/")
def predict_sentiment(input_text: TextInput):
    return test_sentiment(input_text.text)
