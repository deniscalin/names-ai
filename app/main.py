from typing import Union
from fastapi import FastAPI

app = FastAPI()


@app.get('/')
async def read_root():
    return {"Welcome message": "Welcome to the Names AI API"}


@app.get('/names/{number}')
async def get_names(number: int, seed: Union[int, None] = None):
    return {"number": number, "seed": seed}
