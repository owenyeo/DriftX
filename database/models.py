from sqlalchemy import Column, Integer, Float, String, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class InferenceLog(Base):
    __tablename__ = "inference_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    input_data = Column(JSON)
    output_data = Column(JSON, nullable=True)
    latency = Column(Float)
    status_code = Column(Integer)
    timestamp = Column(DateTime)