from datetime import datetime
import uuid
from sqlalchemy import Column, String, Integer, DateTime, Boolean

from app.infra.db.session import Base


class EvalRecord(Base):
    __tablename__ = "eval_records"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    run_id = Column(String, nullable=False)
    document_id = Column(String, nullable=False)
    user_id = Column(String, nullable=False)
    mode = Column(String, nullable=False)
    sample_index = Column(Integer, default=-1)
    question = Column(String, nullable=True)
    answer = Column(String, nullable=True)
    ground_truth = Column(String, nullable=True)
    faithfulness = Column(Integer, nullable=True)
    answer_relevancy = Column(Integer, nullable=True)
    context_precision = Column(Integer, nullable=True)
    context_recall = Column(Integer, nullable=True)
    passed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
