from typing import Union
from fastapi import FastAPI
from app.inference import load_params, generate_names


app = FastAPI()


@app.get('/')
async def read_root():
    return {"Welcome message": "Welcome to the Names AI API"}


@app.get('/names/{number}')
async def get_names(number: int, seed: Union[int, None] = 1):
    C, layers, itos = await load_params()
    names = generate_names(C, layers, itos, number, seed)
    return {'names': names}
