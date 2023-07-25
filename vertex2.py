from fastapi import Request, FastAPI
import os

app = FastAPI()


@app.get('/')
def get_root():
    return {'message': 'Welcome to the spam detection API: ham if you good, spam if you bad'}

@app.get('/health_check')
def health():
    return 200

@app.post(os.environ['AIP_PREDICT_ROUTE'])
async def predict(request: Request):
    print("----------------- PREDICTING -----------------")
    body = await request.json()
    instances = body["instances"]
    outputs = instances
    print("----------------- OUTPUTS -----------------")
    print(outputs)

    return {"predictions": outputs}
