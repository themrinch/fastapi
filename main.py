from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel


class Item(BaseModel):
    text: str


app = FastAPI()
classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)


@app.get("/")
async def root():
    return {'model': 'Text Classifier'}


#@app.get("/predict/")
#def predict():
#    return classifier('I like dogs!')[0]


@app.post("/predict/")
def predict(item: Item):
    """Text Classifier"""
    return classifier(item.text)[0]
