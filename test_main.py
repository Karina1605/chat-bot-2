from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_ask_question():
    response = client.post("/ask", json={"question": "How are you?"})
    
    # Проверяем статус-код ответа
    assert response.status_code == 200
    
    # Проверяем, что ответ содержит поле 'answer'
    assert "answer" in response.json()
    
    # Дополнительно можно проверить, что ответ не пустой
    assert response.json()['answer'] != ""
