from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np
import pandas as pd
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
    ratings_with_name = model['ratings_with_name']

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
        item = {}
        temp_df = new_books[new_books['Book-Title'] == pivot_table.index[idx]]
        if not temp_df.empty:
            item['title'] = temp_df['Book-Title'].values[0]
            item['author'] = temp_df['Book-Author'].values[0]
            item['image_url'] = temp_df['Image-URL-M'].values[0]

            ratings_df = ratings_with_name[ratings_with_name['Book-Title'] == pivot_table.index[idx]]
            item['average_rating'] = ratings_df['Book-Rating'].mean()

            item['similarity'] = similarity
            data.append(item)

    return data

@app.post("/predict")
def predict(book_request: BookRequest):
    book_name = book_request.book_name
    if not book_name:
        raise HTTPException(status_code=400, detail="Book name not provided")

    recommendations = recommend(book_name)
    return {"recommendations": recommendations}

@app.post("/booksget")
def get_books(book_request: BookRequest):
    book_name = book_request.book_name
    if not book_name:
        raise HTTPException(status_code=400, detail="Book name not provided")
    data=[]
    item = {}
    temp_df = new_books[new_books['Book-Title'] == book_name]
    if not temp_df.empty:
        item['title'] = temp_df['Book-Title'].values[0]
        item['author'] = temp_df['Book-Author'].values[0]
        item['image_url'] = temp_df['Image-URL-M'].values[0]

        ratings_df = ratings_with_name[ratings_with_name['Book-Title'] == book_name]
        item['average_rating'] = ratings_df['Book-Rating'].mean()
        data.append(item)
    return {"getbooks": data}
    

@app.get("/high-rated")
def get_high_rated_books() -> List[Dict[str, Any]]:
    high_rated_books = ratings_with_name.groupby('Book-Title').agg(
        average_rating=('Book-Rating', 'mean')
    ).reset_index()

    high_rated_books = high_rated_books.sort_values(by='average_rating', ascending=False).head(10)

    data = []
    for _, row in high_rated_books.iterrows():
        item = {}
        temp_df = new_books[new_books['Book-Title'] == row['Book-Title']]
        if not temp_df.empty:
            item['title'] = temp_df['Book-Title'].values[0]
            item['author'] = temp_df['Book-Author'].values[0]
            item['image_url'] = temp_df['Image-URL-M'].values[0]
            item['average_rating'] = row['average_rating']
            data.append(item)

    return {"high_rated_books": data}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
