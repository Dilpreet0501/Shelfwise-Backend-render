from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np
import pickle

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open('book_recommendation_model.pkl', 'rb') as file:
    model = pickle.load(file)
    pivot_table = model['pivot_table']
    similarity_matrix = model['similarity_matrix']
    new_books = model['new_books']

class BookRequest(BaseModel):
    book_name: str

@app.get("/")
def read_root():
    return {"message": "API is Running"}

def recommend(book_name: str) -> List[Dict[str, Any]]:
    
    try:
        index = np.where(pivot_table.index == book_name)[0][0]
    except IndexError:
        raise HTTPException(status_code=404, detail=f"Book '{book_name}' not found in the pivot table.")

   
    similar_books = sorted(list(enumerate(similarity_matrix[index])), key=lambda x: x[1], reverse=True)[1:11]

    
    data = []
    for idx, similarity in similar_books:
        temp_df = new_books[new_books['Book-Title'] == pivot_table.index[idx]]
        item = {
            "title": temp_df['Book-Title'].values[0],
            "author": temp_df['Book-Author'].values[0],
            "image_url": temp_df['Image-URL-M'].values[0],
            "similarity": similarity
        }
        data.append(item)

    return data

@app.post("/predict")
def predict(book_request: BookRequest):
    book_name = book_request.book_name
    if not book_name:
        raise HTTPException(status_code=400, detail="Book name not provided")

    recommendations = recommend(book_name)
    return {"recommendations": recommendations}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
