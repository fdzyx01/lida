from sqlalchemy import create_engine, Column, String, BigInteger, Date, JSON, Integer, Text, event, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, declared_attr
from datetime import datetime
from lida.config import DATABASE_URL
import uuid

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def generate_unique_id():
    return uuid.uuid4().int & (1 << 63) - 1


class TimestampMixin:
    @declared_attr
    def create_time(cls):
        return Column(DateTime, nullable=False, default=datetime.utcnow)

    @declared_attr
    def update_time(cls):
        return Column(DateTime, nullable=False, default=datetime.utcnow)

    @staticmethod
    def before_update_listener(mapper, connection, target):
        target.update_time = datetime.utcnow()

class MytimeStampMixin:

    @declared_attr
    def created_at(cls):
        return Column(DateTime, nullable=False, default=datetime.utcnow)

    @declared_attr
    def updated_at(cls):
        return Column(DateTime, nullable=False, default=datetime.utcnow)

    @staticmethod
    def before_update_listener(mapper, connection, target):
        target.updated_at = datetime.utcnow()


# 在所有继承TimestampMixin的类上注册before_update监听器
event.listen(TimestampMixin, 'before_update', TimestampMixin.before_update_listener)

# 在所有继承TimestampMixin的类上注册before_update监听器
event.listen(MytimeStampMixin, 'before_update', MytimeStampMixin.before_update_listener)

class Chat(TimestampMixin, Base):
    __tablename__ = "t_chat"

    id = Column(BigInteger, primary_key=True, index=True, nullable=False, default=generate_unique_id)
    user_id = Column(BigInteger, nullable=True)
    name = Column(String(45), nullable=True)
    data_filename = Column(String(45), nullable=True)
    dataset_name = Column(String(45), nullable=True)
    dataset_description = Column(String(200), nullable=True)
    field_names = Column(JSON, nullable=True)
    fields = Column(JSON, nullable=True)
    extra_hint_interest = Column(String(255), nullable=True)


class Goal(TimestampMixin, Base):
    __tablename__ = "t_goal"

    id = Column(BigInteger, primary_key=True, index=True, nullable=False, default=generate_unique_id)
    chat_id = Column(BigInteger, nullable=True)
    index = Column(Integer, nullable=True)
    question = Column(String(255), nullable=True)
    visualization = Column(String(255), nullable=True)
    rationale = Column(String(1000), nullable=True)
    is_auto = Column(Integer, nullable=True)
    library = Column(String(50), nullable=True)
    code = Column(Text, nullable=True)
    explanation = Column(String(255), nullable=True)
    

class Edit(TimestampMixin, Base):
    __tablename__ = "t_edit"

    id = Column(BigInteger, primary_key=True, index=True, nullable=False, default=generate_unique_id)
    goal_id = Column(BigInteger, nullable=True)
    index = Column(Integer, nullable=True)
    edit = Column(String(150), nullable=True)


class Explain(TimestampMixin, Base):
    __tablename__ = "t_explain"

    id = Column(BigInteger, primary_key=True, index=True, nullable=False, default=generate_unique_id)
    goal_id = Column(BigInteger, nullable=True)
    explanation = Column(JSON, nullable=True)


class Evaluate(TimestampMixin, Base):
    __tablename__ = "t_evaluate"

    id = Column(BigInteger, primary_key=True, index=True, nullable=False, default=generate_unique_id)
    goal_id = Column(BigInteger, nullable=True)
    evaluate = Column(JSON, nullable=True)


class Recommend(TimestampMixin, Base):
    __tablename__ = "t_recommend"

    id = Column(BigInteger, primary_key=True, index=True, nullable=False, default=generate_unique_id)
    goal_id = Column(BigInteger, nullable=True)
    code = Column(Text, nullable=True)


class User(TimestampMixin, Base):
    __tablename__ = "users"

    id = Column(BigInteger, primary_key=True, index=True, nullable=False, default=generate_unique_id)
    username = Column(String(50), unique=True, nullable=False)
    password = Column(String(255), nullable=False)


class TaskManagement(MytimeStampMixin, Base):
    __tablename__ = "task_management"

    task_id = Column(Integer, primary_key=True, autoincrement=True, comment='任务ID')
    task_name = Column(String(255), nullable=False, comment='任务名称')
    task_details = Column(Text, nullable=True, comment='任务详情')
    # created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    # updated_at = Column(DateTime, nullable=True)
    chat_id = Column(String(255), nullable=True, comment='ChatID')

# 新增 JsonDataStorage 模型
class JsonDataStorage(MytimeStampMixin, Base):
    __tablename__ = "json_data_storage"

    id = Column(Integer, primary_key=True, autoincrement=True, comment='唯一ID')
    chat_id = Column(String(255), nullable=False, comment='记录ID（如您提供的id字段）')
    json_table1 = Column(JSON, nullable=False, comment='存储整个JSON对象')
    json_table2 = Column(JSON, nullable=False, comment='存储整个JSON对象')
    # created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    # updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)




# 创建数据库表（如果表不存在）
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
