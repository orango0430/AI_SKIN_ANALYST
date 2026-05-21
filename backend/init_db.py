from database import engine, Base
import models  # 테이블 모델 임포트

def init_db():
    print("DB 테이블 생성 중...")
    Base.metadata.create_all(bind=engine)
    print("완료!")

if __name__ == "__main__":
    init_db()
