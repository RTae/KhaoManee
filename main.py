from models.inception_resnet_v1 import InceptionResnetV1
from starlette.responses import StreamingResponse
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from models.mtcnn import MTCNN
from pydantic import BaseModel
from typing import Optional
from detect import detect
from recognition import recognition
import torch
import io

# Looking for device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Face detection model MTCNN
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20, keep_all=True,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

# InceptionResnetV1 for extract feature to make embedding vector
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

tags_metadata = [
    {
        "name": "detection",
        "description": "Face Dectection API : Set data to 'true' will return json of bounding boxes and landmarks ,if you set landmark to 'true'. By the way if you set data to 'false' it will return image with draw bounding box",
    },
    {
        "name": "recognition",
        "description": "Face recognition API : This api can recognition only some person in Thailand Government, by set data to 'true' will return json of bounding boxes. By the way you set data to 'false' it will return image with draw bounding box",
    },
]

# FastAPI
app = FastAPI(
    title="KhaoManee",
    description="KhaoManee is the api that use for face dectection or face recognition ",
    version="1.0.4",
    openapi_tags=tags_metadata
)

# Face Detection
@app.post('/detectface', tags = ['detection'])
async def detectface(file: UploadFile = File(...), landmark: Optional[str] = "false", data = "false"):
    contents = await file.read()
    result = detect(contents, mtcnn, landmark_state=landmark, data=data)
    if result is not None:
        if data.lower() == 'true':
            return JSONResponse(content=result) 
        else:
            return StreamingResponse(io.BytesIO(result), media_type='image/png')
    
    else :
        return StreamingResponse(io.BytesIO(contents), media_type='image/png')

# Face recognition in Thailand government
@app.post('/recognitiongov', tags = ['recognition'])
async def recognitiongov(file: UploadFile = File(...), data = "false"):
    contents = await file.read()
    result = recognition(contents, mtcnn, resnet, device, data=data)
    if result is not None:
        if data.lower() == 'true':
            return JSONResponse(content=result) 
        else:
            return StreamingResponse(io.BytesIO(result), media_type='image/png')
    else :
        return StreamingResponse(io.BytesIO(contents), media_type='image/png')
