from typing import Union
from fastapi import FastAPI
from inference import load_params, generate_names


app = FastAPI()


@app.get('/')
async def read_root():
    return {"Welcome message": "Welcome to the Names AI API"}


@app.get('/names/{number}')
async def get_names(number: int, seed: Union[int, None] = None):
    params, itos = await load_params()
    names = generate_names(params, itos, number, seed)
    return {'names': names}
