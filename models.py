
from sqlalchemy import Column, ForeignKey, Integer, TEXT, LargeBinary
from database import Base
# from pydantic import BaseModel


class User_info(Base):
    __tablename__ = 'user_info'

    id = Column(Integer, primary_key=True)
    phone = Column(TEXT)
    mem_name = Column(TEXT)
    cap_image = Column(LargeBinary)



class Working_hour(Base):
    __tablename__ = 'working_hour'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('user_info.id'))
    start_time = Column(TEXT)
    quit_time = Column(TEXT)