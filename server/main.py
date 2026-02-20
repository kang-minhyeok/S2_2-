import cv2
import os
import numpy as np
import time
import torch
import datetime
from collections import deque, Counter
from fastapi import FastAPI, File, UploadFile, Query, BackgroundTasks, Depends, Request, Form,HTTPException, Response
from fastapi.responses import FileResponse
from fastapi.responses import RedirectResponse
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from pydantic import BaseModel
import subprocess
os.environ["OPENCV_VIDEOIO_DEBUG"] = "1" # 제일 윗줄에 넣으세요
# --- [데이터베이스 설정 구간] ---
DATABASE_URL = "mysql+pymysql://root:0727@localhost:3306/safety_db?charset=utf8mb4"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 비밀번호 암호화 설정
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

# 회원가입 시 받을 데이터 규격 (일반user)
class UserCreate(BaseModel):
    id: str
    password: str
    residentFront: str
    residentBack: str
    name: str
    phone: str

# 관리자 회원가입 시 데이터 규격 (admin)
class AdminSignupRequest(BaseModel):
    id: str
    password: str
    name: str
    phone: str
    residentFront: str
    residentBack: str
    orgCode: str

# 앱에서 서버로 신고 정보를 보낼 때의 형식
class ReportCreate(BaseModel):
    name: str
    phone_number: str
    ssn: str
    content: str

# 로그인 시 정보 확인용 class
class UserLogin(BaseModel):
    id: str
    password: str

# 비밀번호 암호화 함수
def get_password_hash(password):
    return pwd_context.hash(password)

# 영상 감지 내역
class DetectionResult(Base):
    __tablename__ = "detection_results"

    id = Column(Integer, primary_key=True, index=True) # DB 관리용 번호
    object_id = Column(Integer) # 감지된 객체 id
    detected_color = Column(String(20)) # 감지된 color
    video_name = Column(String(100)) # video 이름
    detected_at = Column(DateTime, default=datetime.datetime.now) # 분석된 시간


# users 테이블 정의
class User(Base):
    __tablename__ = "users"

    user_no = Column(Integer, primary_key=True, index=True) # DB 관리용 번호
    id = Column(String(50), unique=True, nullable=False)   # 로그인 아이디
    password = Column(String(255), nullable=False)          # 암호화된 비밀번호
    residentFront = Column(String(6), nullable=False)           # 주민번호 앞자리
    residentBack = Column(String(255), nullable=False)          # 암호화된 주민번호 뒷자리
    name = Column(String(50), nullable=False)               # 사용자 이름
    phone = Column(String(20), nullable=False)       # 휴대폰 번호
    role = Column(String(20), default="USER")     # 역할 "USER"(일반), "ADMIN"(관계자)
    affiliation = Column(String(100), nullable=True) # 관계자일 경우 소속명 저장
    created_at = Column(DateTime, default=datetime.datetime.now) # 가입일

# 소속 코드 관리 테이블
class Affiliation(Base):
    __tablename__ = "affiliations"

    id = Column(Integer, primary_key=True, index=True) # DB 관리용 번호
    code = Column(String(50), unique=True, nullable=False)  # 인증 코드
    name = Column(String(100), nullable=False)              # 소속명 (예: xx경찰서, oo소방서)

# 신고 내역 테이블
class IncidentReport(Base):
    __tablename__ = "incident_reports"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), nullable=False) # 신고자 성함
    phone_number = Column(String(20), nullable=False) # 전화번호
    ssn = Column(String(255), nullable=False) # 주민번호 (암호화 저장)
    content = Column(String(500), nullable=True) # 신고 상세 내용
    video_path = Column(String(200), nullable=True) # 분석할 영상 경로
    created_at = Column(DateTime, default=datetime.datetime.now)



# 테이블 자동 생성 (DB 탭에서 확인 가능)
Base.metadata.create_all(bind=engine)


app = FastAPI()

# 1. GPU 장치 설정 및 로드
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✅ 실행 장치: {device} ")

# Nano 모델을 GPU 메모리로 로드
model = YOLO('yolov8n-pose.pt').to(device)

# 2. 전역 메모리 설정
# 시계열 분석을 위한 color_buffer
# 프레임 skip시 전 프레임의 객체박스 위치를 기록할 latest_track_data
color_buffer = {}
latest_track_data = {}


"""
표준 Hue 범위 기반 색상 판별 로직
"""
def detect_color_name(roi):

    if roi is None or roi.size == 0: return "Unknown"
    # 속도를 위해 24x24 리사이징
    small_roi = cv2.resize(roi, (24, 24), interpolation=cv2.INTER_AREA)
    data = small_roi.reshape((-1, 3)).astype(np.float32)

    # K-Means (K=2)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(data, 2, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
    dom_bgr = centers[np.argmax(np.bincount(labels.flatten()))]
    h, s, v = cv2.cvtColor(np.uint8([[dom_bgr]]), cv2.COLOR_BGR2HSV)[0][0]

    # 표준 HSV 색상 정의
    if v < 75: return "Black"
    if s < 50: return "White" if v > 200 else "Gray"
    if h < 8 or h > 170: return "Red"
    if 155 <= h <= 170: return "Pink"
    if 8 <= h < 16: return "Orange"
    if 16 <= h < 38: return "Yellow"
    if 38 <= h < 88: return "Green"
    if 88 <= h < 125: return "Blue"
    if 125 <= h < 155: return "Purple"
    return "Color"



"""
시계열 분석
10프레임을 비교해 객체의 색상 판별해 오차를 줄임
"""
def get_smoothed_color(obj_id, new_color):

    if obj_id not in color_buffer:
        color_buffer[obj_id] = deque(maxlen=10)
    color_buffer[obj_id].append(new_color)
    if len(color_buffer[obj_id]) < 4: return new_color
    return Counter(color_buffer[obj_id]).most_common(1)[0][0]



"""
영상 분석할 때 사용하는 함수입니다.
"""
# [수정] 기존 analyze_video의 핵심 로직을 함수로 분리
def process_video_analysis(report_id: int, content: str = None):
    # 별도의 DB 세션을 생성합니다. (백그라운드 작업용)
    db = SessionLocal()

    global latest_track_data, color_buffer
    ts = int(time.time())

    # [주의] 실제 서비스에서는 신고 내용에 포함된 비디오 경로를 사용해야 합니다.
    # 일단 현재 코드의 test.mp4를 유지합니다.
    in_path, out_path = "test.mp4", f"out_{ts}_{report_id}.mp4"
    cap, out = None, None

    latest_track_data = {}
    color_buffer = {}
    final_results = {}

    try:
        color_map = {"검은": "Black", "흰": "White", "빨간": "Red", "파란": "Blue",
                     "노란": "Yellow", "초록": "Green", "보라": "Purple", "회": "Gray", "분홍": "Pink"}
        target_color = next((v for k, v in color_map.items() if content and k in content), "")

        cap = cv2.VideoCapture(in_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if not out.isOpened():
            print(f"❌ [비상] 비디오 파일을 열 수 없습니다! 경로: {out_path}")
            print(f"현재 작업 디렉토리: {os.getcwd()}")
        else:
            print(f"✅ 비디오 파일 생성 시작: {out_path}")
        frame_count = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            frame_count += 1

            if frame_count % 3 == 0:
                # YOLO 추적 로직 실행
                results = model.track(frame, persist=True, verbose=False, conf=0.3, device=device)
                new_tracks = {}

                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                    ids = results[0].boxes.id.cpu().numpy().astype(int)
                    kpts = results[0].keypoints.xy.cpu().numpy() if results[0].keypoints is not None else []

                    for i, obj_id in enumerate(ids):
                        if obj_id in latest_track_data and frame_count % 15 != 0:
                            new_tracks[obj_id] = latest_track_data[obj_id]
                            new_tracks[obj_id]["box"] = boxes[i]
                            continue

                        # 포즈 기반 ROI 추출 및 색상 감지
                        roi = None
                        if i < len(kpts) and all(kpts[i][j][1] > 0 for j in [5, 6]):
                            pk = kpts[i]
                            sw = abs(pk[5][0] - pk[6][0])
                            ry1, ry2 = int(min(pk[5][1], pk[6][1])), int(min(pk[5][1], pk[6][1]) + sw * 0.5)
                            rx1, rx2 = int(min(pk[5][0], pk[6][0])), int(max(pk[5][0], pk[6][0]))
                            roi = frame[ry1:ry2, rx1:rx2]

                        if roi is not None and roi.size > 0:
                            stable_color = get_smoothed_color(obj_id, detect_color_name(roi))
                            new_tracks[obj_id] = {"box": boxes[i], "color": stable_color}
                            final_results[obj_id] = stable_color

                latest_track_data = new_tracks

            # 결과 그리기 및 저장
            for obj_id, data in latest_track_data.items():
                if target_color == "" or target_color.lower() in data["color"].lower():
                    x1, y1, x2, y2 = data["box"]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, f"ID:{obj_id} {data['color']}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            out.write(frame)

        # --- [DB 저장 단계] ---
        try:
            for obj_id, color in final_results.items():
                new_record = DetectionResult(
                    object_id=int(obj_id),
                    detected_color=color,
                    video_name=out_path
                    # 여기에 report_id 컬럼이 있다면 추가: report_id=report_id
                )
                db.add(new_record)

            # [추가] 신고서 테이블에도 분석된 비디오 경로 업데이트
            report = db.query(IncidentReport).filter(IncidentReport.id == report_id).first()
            if report:
                report.video_path = out_path

            db.commit()
            print(f"✅ 신고번호 {report_id}: {len(final_results)}건 탐지 기록 저장 완료")
        except Exception as e:
            db.rollback()
            print(f"❌ 분석 결과 저장 실패: {e}")

    finally:
        if out: out.release()
        if cap: cap.release()
        # [핵심] 웹 재생용(H.264)으로 강제 변환
        web_out_path = f"web_{out_path}"
        try:
            # ffmpeg를 사용하여 브라우저가 좋아하는 libx264 코덱으로 인코딩
            subprocess.run([
                'ffmpeg', '-i', out_path,
                '-vcodec', 'libx264',
                '-acodec', 'aac',
                '-movflags', 'faststart',
                '-y', web_out_path
            ], check=True)

            # 변환 성공 시, DB에 저장할 경로를 웹용 파일로 변경
            final_save_path = web_out_path
        except Exception as e:
            print(f"❌ ffmpeg 변환 실패: {e}")
            final_save_path = out_path # 실패 시 원본이라도 유지

        # DB 업데이트 (Jinja2 변수명에 맞춰 저장)
        report = db.query(IncidentReport).filter(IncidentReport.id == report_id).first()
        if report:
            report.video_path = final_save_path
            db.commit()


# DB 세션을 가져오는 헬퍼 함수
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

"""
앱이나 웹에서 탐지 기록을 조회할 때 사용하는 API입니다.
특정 색상만 골라서 보고 싶을 때 /logs?color=Red 처럼 요청할 수 있습니다.
"""
@app.get("/logs")
async def get_detection_logs(color: str = Query(None)):
    """
    앱이나 웹에서 탐지 기록을 조회할 때 사용하는 API입니다.
    특정 색상만 골라서 보고 싶을 때 /logs?color=Red 처럼 요청할 수 있습니다.
    """
    db = SessionLocal()
    try:
        # 1. DB에서 쿼리 생성
        query = db.query(DetectionResult)

        # 2. 색상 필터링이 들어온 경우 처리
        if color:
            query = query.filter(DetectionResult.detected_color == color)

        # 3. 최신순으로 정렬하여 데이터 가져오기 (최신 발견이 위로)
        logs = query.order_by(DetectionResult.detected_at.desc()).all()

        # 4. 앱/웹이 이해하기 쉬운 리스트 형식으로 반환
        return [
            {
                "id": log.id,
                "object_id": log.object_id,
                "color": log.detected_color,
                "video_url": f"http://서버주소/download/{log.video_name}", # 영상 링크
                "detected_at": log.detected_at.strftime("%Y-%m-%d %H:%M:%S")
            } for log in logs
        ]
    except Exception as e:
        return {"error": str(e)}
    finally:
        db.close()




# 회원가입 API
@app.post("/register")
async def register_user(
        request: Request,
        id: str = Form(...),            # 앱의 JSON 대신 웹의 Form 데이터를 받습니다.
        password: str = Form(...),
        residentFront: str = Form(...),
        residentBack: str = Form(...),
        name: str = Form(...),
        phone: str = Form(...),
        db: Session = Depends(get_db)
):
    # 2. 아이디 중복 확인 (기존 로직 동일)
    existing_user = db.query(User).filter(User.id == request.id).first()
    if existing_user:
        return HTMLResponse(content="<script>alert('이미 사용 중인 아이디입니다.'); history.back();</script>", status_code=400)

    # 1. 기존 로직 그대로 사용 (암호화)
    hashed_pwd = pwd_context.hash(password)
    hashed_res_back = pwd_context.hash(residentBack)

    # 2. DB에 저장할 사용자 객체 생성
    new_user = User(
        id=id,
        password=hashed_pwd,
        residentFront=residentFront,
        residentBack=hashed_res_back,
        name=name,
        phone=phone
    )

    # 3. DB에 저장 (SQLAlchemy가 INSERT 쿼리를 자동 생성)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return RedirectResponse(url="/", status_code=303)

# 신고 접수 API
@app.post("/report/submit")
async def submit_report(
        report: ReportCreate,
        background_tasks: BackgroundTasks,
        db: Session = Depends(get_db)
):
    # 1. 주민번호 암호화 저장
    hashed_ssn = pwd_context.hash(report.ssn)

    new_report = IncidentReport(
        name=report.name,
        phone_number=report.phone_number,
        ssn=hashed_ssn,
        content=report.content
    )

    db.add(new_report)
    db.commit()
    db.refresh(new_report)

    # 2. [핵심] 비디오 분석 함수를 백그라운드 작업으로 등록
    # 사용자가 보낸 신고내용(content)을 함께 넘겨 특정 색상을 찾게 합니다.
    background_tasks.add_task(process_video_analysis, report_id=new_report.id, content=report.content)

    return {
        "status": "success",
        "message": "신고 접수가 완료되었습니다. 비디오 분석이 백그라운드에서 시작됩니다.",
        "report_id": new_report.id
    }

# 관리자(admin) 회원가입 API
@app.post("/signup/admin")
def signup_admin(request: Request,
                 id: str = Form(...),
                 password: str = Form(...),
                 name: str = Form(...),
                 phone: str = Form(...),
                 residentFront: str = Form(...),
                 residentBack: str = Form(...),
                 orgCode: str = Form(...), # [핵심] 관리자용 추가 필드
                 db: Session = Depends(get_db)
):

    # 1. 소속 코드 검증
    # 입력한 코드가 DB에 있는지 확인하고, 없으면 에러 발생
    affiliation_data = db.query(Affiliation).filter(Affiliation.code == orgCode).first()

    if not affiliation_data:
        return HTMLResponse(content="<script>alert('유효하지 않은 소속 코드입니다.'); history.back();</script>", status_code=400)

    # 2. 아이디 중복 확인
    existing_user = db.query(User).filter(User.id == id).first()
    if existing_user:
        return HTMLResponse(content="<script>alert('이미 사용 중인 아이디입니다.'); history.back();</script>", status_code=400)

    # 3. 비밀번호 해싱
    hashed_password = pwd_context.hash(password)
    hashed_res_back = pwd_context.hash(residentBack)

    # 4. DB 저장
    # role은 "ADMIN", affiliation은 코드에 해당하는 이름(ex: oo경찰서)으로 자동 저장
    new_user = User(
        id=id,
        password=hashed_password,      # 암호화된 비밀번호 저장
        name=name,
        phone=phone,
        residentFront=residentFront,
        residentBack=hashed_res_back,
        role="ADMIN",                  # 관리자 권한 부여
        affiliation=affiliation_data.name # 소속명 입력
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return RedirectResponse(url="/", status_code=303)


# 1. 정적 파일(CSS/JS) 연결
app.mount("/static", StaticFiles(directory="static"), name="static")

# 2. Jinja2 템플릿 엔진 설정
templates = Jinja2Templates(directory="templates")

# 메인 페이지 (http://IP:8000/ 접속 시)
@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request, db: Session = Depends(get_db)):
    # 브라우저가 가져온 쿠키에서 "session_user"가 있는지 확인합니다.
    user_id = request.cookies.get("session_user")
    user_info = None
    if user_id:
        # 2. ID가 있다면 DB에서 사용자 전체 정보를 조회 (이름, 소속, 권한 등)
        user_info = db.query(User).filter(User.id == user_id).first()
    # 템플릿 파일명과 함께 데이터를 넘겨줍니다.
    return templates.TemplateResponse("home.html", {
        "request": request,
        "user": user_info
    })


# 로그인 기능(앱)
@app.post("/login")
async def login(
        request: Request,
        id: str = Form(...), # [수정] user_data 대신 직접 Form 데이터를 받습니다.
        password: str = Form(...),
        db: Session = Depends(get_db)
):
    # 1. DB에서 해당 아이디의 사용자 찾기
    user = db.query(User).filter(User.id == id).first()

    # 2. 사용자가 없거나 비밀번호가 틀린 경우
    if not user or not pwd_context.verify(password, user.password):
        # 웹 환경에서는 에러 페이지로 리다이렉트하거나 에러 파라미터를 붙여 리턴하는 게 좋습니다.
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": True
        })

    # 3. 로그인 성공 시 관리자 페이지로 이동 (Redirect)
    response = RedirectResponse(url="/", status_code=303)
    # "session_user"라는 이름으로 아이디를 브라우저에 저장합니다. (유효기간 1시간)
    response.set_cookie(key="session_user", value=user.id, httponly=True, max_age=3600)
    # [중요] 여기에 세션 정보를 쿠키 등으로 저장하는 로직이 추가되어야 나중에 /admin 페이지에 들어갈 수 있습니다.
    return response


# 1. 로그인 페이지 보여주기 (GET)
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: bool = False, logout: bool = False):
    return templates.TemplateResponse("login.html", {
        "request": request,
        "error": error,
        "logout": logout
    })

# 1. 로그아웃으로 메인화면으로 이동 (post)
@app.post("/logout")
async def logout():
    response = RedirectResponse(url="/", status_code=303)
    response.delete_cookie("session_user") # 쿠키 삭제로 로그아웃 처리
    return response

# 1. 회원가입 유형 선택 페이지
@app.get("/signup/select", response_class=HTMLResponse)
async def signup_select_page(request: Request):
    return templates.TemplateResponse("signup/select.html", {"request": request})

# 2. 일반 회원가입 양식 페이지
@app.get("/signup/user", response_class=HTMLResponse)
async def signup_user_page(request: Request):
    return templates.TemplateResponse("signup/user.html", {"request": request})

# 3. 관계자용 회원가입 양식 페이지
@app.get("/signup/admin", response_class=HTMLResponse)
async def signup_admin_page(request: Request):
    return templates.TemplateResponse("signup/admin.html", {"request": request})

# 관리자 화면 연결
@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request, db: Session = Depends(get_db)):

    # 1. 로그인 여부 확인
    user_id = request.cookies.get("session_user")
    if not user_id:
        # 로그인이 안 되어 있다면 로그인 페이지로 튕김
        return RedirectResponse(url="/login", status_code=303)

    # 2. 사용자 정보 및 권한 조회
    user = db.query(User).filter(User.id == user_id).first()

    # 3. 관리자(ADMIN) 권한 체크
    # 유저가 없거나, 역할이 ADMIN이 아니라면 '접근 거부' 페이지 리턴
    if not user or user.role != "ADMIN":
        return templates.TemplateResponse("access-denied.html", {"request": request})

    # 4. 권한 통과 시: IncidentReport 테이블에서 모든 데이터를 최신순으로 가져옴
    reports = db.query(IncidentReport).order_by(IncidentReport.id.desc()).all()

    # 5. admin_dashboard.html로 렌더링
    return templates.TemplateResponse("admin_dashboard.html", {
        "request": request,
        "reports": reports,
        "user": user # 상단에 이름 표시용
    })

# 관리자 화면 접근 거부 (일반 user 또는 로그인을 안한상태)
@app.get("/access-denied", response_class=HTMLResponse)
async def access_denied(request: Request):
    return templates.TemplateResponse("access-denied.html", {"request": request})

# CCTV 모니터링 페이지
@app.get("/admin/cctv/{report_id}", response_class=HTMLResponse)
async def admin_cctv_page(report_id: int, request: Request, db: Session = Depends(get_db)):

    # 관리자 권한 체크
    user_id = request.cookies.get("session_user")
    user = db.query(User).filter(User.id == user_id).first()
    if not user or user.role != "ADMIN":
        return templates.TemplateResponse("access-denied.html", {"request": request})

    # DB에서 신고 내역 조회
    report = db.query(IncidentReport).filter(IncidentReport.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="영상을 찾을 수 없는 신고 내역입니다.")

    return templates.TemplateResponse("admin_cctv.html", {
        "request": request,
        "report": report, # HTML로 신고 정보 전달
        "user": user
    })

# 신고 내역 상세 보기
@app.get("/admin/{report_id}", response_class=HTMLResponse)
async def admin_report_detail(report_id: int, request: Request, db: Session = Depends(get_db)):
    # 권한 체크 (admin만 가능)
    user_id = request.cookies.get("session_user")
    user = db.query(User).filter(User.id == user_id).first()
    if not user or user.role != "ADMIN":
        return templates.TemplateResponse("access-denied.html", {"request": request})

    # 특정 신고 내역 조회
    report = db.query(IncidentReport).filter(IncidentReport.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="신고 내역을 찾을 수 없습니다.")

    # 상세 페이지 렌더링
    return templates.TemplateResponse("admin_detail.html", {
        "request": request,
        "report": report,
        "user": user
    })



# 분석된 영상 파일을 브라우저에서 볼 수 있게 해주는 경로
@app.get("/video/{video_name}")
async def get_video(video_name: str):
    file_path = os.path.join(os.getcwd(), video_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404)

    return FileResponse(file_path, media_type="video/mp4")