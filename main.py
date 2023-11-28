import io
import os
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



#  환경 설정


####  FastAPI 및 템플릿 설정 ####
app = FastAPI()
templates = Jinja2Templates(directory="templates")

####  정적 파일 및 디렉토리 설정 ####
statics_path = os.path.join("templates", "statics", "images")
app.mount("/static", StaticFiles(directory=statics_path, html=True), name="static")

#### DB 설정 ####
from database import engine, sessionlocal

models.Base.metadata.create_all(bind=engine) # DB models 테이블생성 
def get_db():
    db = sessionlocal()
    try:
        yield db # DB 일시 중지
    finally:
        db.close()

####  폴더 생성 ####
if not os.path.exists(statics_path):  
    os.makedirs(statics_path)

#### 
async def capture_and_save_image(image: UploadFile):
    image_path = os.path.join(statics_path, "temp_file.png")

    with open(image_path, "wb") as f:
        f.write(await image.read())  # Use await to read the uploaded file content
    return image_path


###
# 코드
##
@app.get("/")
async def home(request: Request, db: Session = Depends(get_db)):
    return templates.TemplateResponse("index.html", {"request": request})


async def process_image(contents):
    nparr = np.frombuffer(contents, np.uint8) 
    cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    app_face = FaceAnalysis(allowed_modules=['detection', 'recognition'], providers=['CPUExecutionProvider'])
    app_face.prepare(ctx_id=0, det_size=(640, 640))
    
    faces = app_face.get(cv_img)

    try:
        print(faces)
        feat = np.array(faces[0].normed_embedding, dtype=np.float32)
        print(feat)
        return feat
    except:
        return None
        


@app.post("/upload", response_class=JSONResponse)
async def upload_image(request: Request, file: UploadFile, db: Session = Depends(get_db)):
    contents = await file.read()
    feat1 = await process_image(contents)

    user_list = db.query(models.User_info)

    for user in user_list:
        feat2 = await process_image(user.cap_image)

        # result = {
        #     "user_list": user_list,
        #     "message": "Data processed successfully",
        #     "output": "Your processed output here",
        # }
        
        if feat1 is None and feat2 is None:
            print(f'feat1 is None and feat2 is None:=====feat1:{feat1}, feat2:{feat2},: error뭔가 잘못됨')
            return {"data" : '인식되지 않았습니다. 다시 시도해주세요.'}
            
        sims = np.dot(feat1, feat2)

        if sims >= 0.55:
            print(f'if sims >= 0.55:================{sims}, {user.mem_name}')
            
            # return { "user" : user.mem_name }
            return {"user" : user.mem_name} 
        # templates.TemplateResponse("index.html", {"request": request, "result_data": user})
        else:
            print(f'if else======================={sims},: 회원이 아님')

    return {"data" : '인식되지 않았습니다. 다시 시도해주세요.'}


# 사원등록
@app.post("/regist", response_class=JSONResponse)
async def save(file: UploadFile, name:str=Form(...), phone:str=Form(...), db: Session = Depends(get_db),):

    file_read = await file.read()
    print(file_read)

    if process_image(file_read) is None:
        return '정면을 응시해주세요.'

    try:
        new_user = models.User_info(mem_name = name, phone = phone, cap_image = file_read)

        db.add(new_user)
        db.commit()

        # 이미지를 데이터베이스에 추가
        return {"user" : new_user.mem_name} 
        
    except :
        return {"data" : '인식되지 않았습니다. 다시 시도해주세요.'}





