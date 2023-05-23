from typing import Union
from fastapi import FastAPI

app = FastAPI()


@app.get('/')
async def read_root():
    return {"Welcome message": "Welcome to Names API"}


@app.get('/names/{number}')
async def get_names(number: int, seed: Union[str, None] = None):
    return {"number": number, "seed": seed}
