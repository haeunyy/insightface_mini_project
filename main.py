import io
import os
import time
from fastapi.staticfiles import StaticFiles
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from insightface.app import FaceAnalysis
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
import insightface
from sqlalchemy.orm import Session
import models 
from PIL import Image
from mediapipe.tasks.python import vision
from database import engine, sessionlocal
from util import BytesIoImageOpen, createBytesIo





templates = Jinja2Templates(directory="templates")

app = FastAPI()
app.mount("/static", StaticFiles(directory='static', html=True), name="static") 

module = FaceAnalysis(allowed_modules=['detection', 'recognition'], providers=['CPUExecutionProvider'])
module.prepare(ctx_id=0, det_size=(640, 640))

models.Base.metadata.create_all(bind=engine) # DB models 테이블생성 




def get_db():
    db = sessionlocal()
    try:
        yield db # DB 일시 중지
    finally:
        db.close()





async def process_image(contents):
    """
        UploadFile로 받은 이미지를 추론하여 벡터 배열값을 반환합니다. 


    """
    nparr = np.frombuffer(contents, np.uint8) 
    cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    faces = module.get(cv_img)

    try:
        feat = np.array(faces[0].normed_embedding, dtype=np.float32)
    except:
        return None
    
    return feat




@app.get("/")
async def home(request: Request, db: Session = Depends(get_db)):
    return templates.TemplateResponse("index.html", {"request": request})



        
now = time

@app.post("/upload", response_class=JSONResponse)
async def upload_image(request: Request, file: UploadFile, db: Session = Depends(get_db)):
    """
    로그인 기능 


    """
    contents = await file.read()
    feat1 = await process_image(contents)

    user_list = db.query(models.User_info)

    for user in user_list:
        feat2 = await process_image(user.cap_image)

        if feat1 is None and feat2 is None:
            return {"result" : 'error'}
            
        sims = np.dot(feat1, feat2)

        if sims > 0.55:
            
            work_user = models.Working_hour(mem_id = user.id, start_time = now.strftime('%Y-%m-%d %H:%M:%S'))
            db.add(work_user)
            db.commit()
            return {"user" : user.mem_name} 

    return {"result" : 'error'}


@app.post("/regist", response_class=JSONResponse)
async def save(file: UploadFile, name:str=Form(...), phone:str=Form(...), db: Session = Depends(get_db),):
    """
    사원 등록

    """
    file_read = await file.read()

    if process_image(file_read) is None:
        return {"result" : 'error'}

    try:
        new_user = models.User_info(mem_name = name, phone = phone, cap_image = file_read)

        db.add(new_user)
        db.commit()

        return {"user" : new_user.mem_name} 
        
    except :
        return {"result" : 'error'}




@app.post("/exit", response_class=JSONResponse)
async def upload_image(request: Request, file: UploadFile, db: Session = Depends(get_db)):
    """
    퇴근 등록

    """
    contents = await file.read()
    feat1 = await process_image(contents)

    user_list = db.query(models.User_info)

    for user in user_list:
        feat2 = await process_image(user.cap_image)

        if feat1 is None and feat2 is None:
            return {"result" : 'error'}
            
        sims = np.dot(feat1, feat2)

        if sims > 0.55:
            work_user = db.query(models.Working_hour).filter(models.Working_hour.mem_id == user.id).first()
            work_user.quit_time = now.strftime('%Y-%m-%d %H:%M:%S')
            db.commit()
            
            return {"user" : user.mem_name} 

    return {"result" : 'error'}





