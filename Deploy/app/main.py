from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
from app.model.model import predict


app = FastAPI()


@app.get("/")
def home():
    return {"health_check": "OK"}


@app.post("/upload/")
async def upload_image(file: UploadFile):
    file_content = await file.read()

    # Run inference on the uploaded file content
    result = predict(file_content)
    return JSONResponse(result)
