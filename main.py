from fastapi import FastAPI
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from safetensors import safe_open
import torch

app = FastAPI()

model_path = './model'
def initialize_model():
    weights = safe_open(f"{model_path}/model.safetensors", framework="pt")

    # Получаем тензоры из weights
    weights_dict = {key: weights.get_tensor(key) for key in weights.keys()}

    # Загрузка конфигурации и токенизатора
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)

    # Загрузка модели с конфигурацией
    model = GPT2LMHeadModel.from_pretrained(model_path)

    # Загрузка весов модели
    model.load_state_dict(weights_dict)
    return (tokenizer, model)

tokenizer, model = initialize_model()

# Описание входных данных для API
class QuestionRequest(BaseModel):
    question: str

# Функция для генерации ответа
def generate_response(prompt: str, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=max_length, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Точка входа для API
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    question = request.question
    answer = generate_response(question)
    return {"answer": answer}

if __name__ == '__main__':
    app.run(debug=True)