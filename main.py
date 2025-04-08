from fastapi import FastAPI, Request, HTTPException
import dill
import pandas as pd

app = FastAPI()

# Загрузка обученной модели и пользовательского препроцессора при старте приложения
try:
    with open('data/models/ga_pipeline_final.pkl', 'rb') as f:
        model_data = dill.load(f)
        model = model_data['model_pipeline']
    with open('data/models/custom_preprocessor.pkl', 'rb') as f:
        custom_preprocessor = dill.load(f)
except FileNotFoundError:
    print("Ошибка: Не найдены файлы с моделью или препроцессором.")
    model = None
    custom_preprocessor = None

@app.get("/version")
async def version():
    return model_data['metadata']

@app.post("/predict")
async def predict(request: Request):
    if model is None or custom_preprocessor is None:
        raise HTTPException(status_code=500, detail="Модель или препроцессор не загружены.")

    body = await request.json()
    input_data = pd.DataFrame([body])

    # Применяем пользовательский препроцессор
    processed_input_data = custom_preprocessor.transform(input_data)

    # Делаем предсказание
    prediction = model.predict(processed_input_data)[0]
    return {
        "session_id": body["session_id"],
        "prediction": int(prediction)
    }