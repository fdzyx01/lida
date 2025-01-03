from http.client import HTTPException
import json
import os
import logging
from datetime import timedelta
from json import JSONDecodeError
from typing import Dict, List

import requests
from fastapi import FastAPI, UploadFile, Form, Depends, Query
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import traceback

from llmx import llm, providers
from sqlalchemy import delete
from sqlalchemy.orm import Session

from .auth import authenticate_user, ACCESS_TOKEN_EXPIRE_MINUTES, create_access_token, create_user, get_current_user
from .entity import SessionLocal, Chat, Goal, Explain, Evaluate, Edit, TaskManagement, User, Recommend
from pydantic import BaseModel

from .models import Token
from ..datamodel import GoalWebRequest, SummaryUrlRequest, TaskCreateRequest, TextGenerationConfig, UploadUrl, VisualizeEditWebRequest, \
    VisualizeEvalWebRequest, VisualizeExplainWebRequest, VisualizeRecommendRequest, VisualizeRepairWebRequest, \
    VisualizeWebRequest, InfographicsRequest, VisualizeConclusionRequest, DescribeData, UserCreate, VisWebRequest
from ..components import Manager
from lida.ollamaTextGenerator import OllamaTextGenerator

# instantiate model and generator
# textgen = llm()
# 这里设置固定为ollama了
textgen = OllamaTextGenerator()
logger = logging.getLogger("lida")
api_docs = os.environ.get("LIDA_API_DOCS", "False") == "True"

lida = Manager(text_gen=textgen)
app = FastAPI()
# allow cross origin requests for testing on localhost:800* ports only
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000", "http://localhost:8001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
api = FastAPI(root_path="/api", docs_url="/docs" if api_docs else None, redoc_url=None)
app.mount("/api", api)

root_file_path = os.path.dirname(os.path.abspath(__file__))
static_folder_root = os.path.join(root_file_path, "ui")
files_static_root = os.path.join(root_file_path, "files/")
data_folder = os.path.join(root_file_path, "files/data")
os.makedirs(data_folder, exist_ok=True)
os.makedirs(files_static_root, exist_ok=True)
os.makedirs(static_folder_root, exist_ok=True)

# mount lida front end UI files
app.mount("/", StaticFiles(directory=static_folder_root, html=True), name="ui")
api.mount("/files", StaticFiles(directory=files_static_root, html=True), name="files")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# def check_model

@api.post("/visualize")
async def visualize_data(req: VisualizeWebRequest,
                         db: Session = Depends(get_db),
                         current_user: User = Depends(get_current_user)) -> dict:
    """Generate goals given a dataset summary"""
    try:
        db_chat = db.query(Chat).filter(Chat.id == req.chat_id, Chat.user_id == current_user.id).first()
        if db_chat is None:
            return {"status": False,
                    "message": "Chat not found with the given chat_id"}
        # print(req.textgen_config)
        charts = lida.visualize(
            summary=req.summary,
            goal=req.goal,
            textgen_config=req.textgen_config if req.textgen_config else TextGenerationConfig(),
            library=req.library, return_error=True)
        print("found charts: ", len(charts), " for goal: ")
        if len(charts) == 0:
            return {"status": False, "message": "No charts generated"}

        # save to database
        db_goal = db.query(Goal).filter(Goal.chat_id == req.chat_id, Goal.question == req.goal.question).first()
        if db_goal is None:
            db_goal = Goal(chat_id=req.chat_id,
                           index=req.goal.index,
                           question=req.goal.question,
                           visualization=req.goal.visualization,
                           rationale=req.goal.rationale,
                           is_auto=0,
                           library=req.library,
                           code=charts["code"])
        else:
            db_goal.library = req.library
            db_goal.code = charts[0].code
        db.commit()
        db.refresh(db_goal)

        return {"status": True, "charts": charts,
                "message": "Successfully generated charts.",
                "goal_id": f"{db_goal.id}"}

    except Exception as exception_error:
        logger.error(f"Error generating visualization goals: {str(exception_error)}")
        return {"status": False,
                "message": f"Error generating visualization goals. {str(exception_error)}"}


@api.post("/visualize/edit")
async def edit_visualization(req: VisualizeEditWebRequest,
                             db: Session = Depends(get_db),
                             current_user: User = Depends(get_current_user)) -> dict:
    """Given a visualization code, and a goal, generate a new visualization"""
    try:
        db_goal = db.query(Goal).filter(Goal.chat_id == req.chat_id, Goal.id == req.goal_id).first()
        if db_goal is None:
            return {"status": False,
                    "message": "Goal not found with the given chat_id and question"}

        textgen_config = req.textgen_config if req.textgen_config else TextGenerationConfig()
        charts = lida.edit(
            code=req.code,
            summary=req.summary,
            instructions=req.instructions,
            textgen_config=textgen_config,
            library=req.library, return_error=True)

        # charts = [asdict(chart) for chart in charts]
        if len(charts) == 0:
            return {"status": False, "message": "No charts generated"}

        # save to database
        db_goal.library = req.library
        db_goal.code = charts[0].code
        result = db.execute(delete(Edit).where(Edit.goal_id == db_goal.id))
        if isinstance(req.instructions, str):
            db_edit = Edit(goal_id=db_goal.id,
                           index=0,
                           edit=req.instructions)
            db.add(db_edit)
        elif isinstance(req.instructions, list):
            index = 0
            for instruction in req.instructions:
                db_edit = Edit(goal_id=db_goal.id,
                               index=index,
                               edit=instruction)
                index += 1
                db.add(db_edit)
        db.commit()

        return {"status": True, "charts": charts,
                "message": f"Successfully edited charts."}

    except Exception as exception_error:
        logger.error(f"Error generating visualization edits: {str(exception_error)}")
        print(traceback.print_exc())
        return {"status": False,
                "message": f"Error generating visualization edits."}


@api.post("/visualize/repair")
async def repair_visualization(req: VisualizeRepairWebRequest) -> dict:
    """ Given a visualization goal and some feedback, generate a new visualization that addresses the feedback"""

    try:

        charts = lida.repair(
            code=req.code,
            feedback=req.feedback,
            goal=req.goal,
            summary=req.summary,
            textgen_config=req.textgen_config if req.textgen_config else TextGenerationConfig(),
            library=req.library,
            return_error=True
        )

        if len(charts) == 0:
            return {"status": False, "message": "No charts generated"}
        return {"status": True, "charts": charts,
                "message": "Successfully generated chart repairs"}

    except Exception as exception_error:
        logger.error(f"Error generating visualization repairs: {str(exception_error)}")
        return {"status": False,
                "message": f"Error generating visualization repairs."}


@api.post("/visualize/explain")
async def explain_visualization(req: VisualizeExplainWebRequest,
                                db: Session = Depends(get_db),
                                current_user: User = Depends(get_current_user)) -> dict:
    """Given a visualization code, provide an explanation of the code"""
    textgen_config = req.textgen_config if req.textgen_config else TextGenerationConfig(
        n=1,
        temperature=0)

    try:
        db_goal = db.query(Goal).filter(Goal.chat_id == req.chat_id, Goal.id == req.goal_id).first()
        if db_goal is None:
            return {"status": False,
                    "message": "Goal not found with the given chat_id and question"}

        explanations = lida.explain(
            code=req.code,
            textgen_config=textgen_config,
            library=req.library)

        # save to database
        db_explain = db.query(Explain).filter(Explain.goal_id == db_goal.id).first()
        if db_explain is None:
            db_explain = Explain(goal_id=db_goal.id,
                                 explanation={"explanations": explanations[0]})
            db.add(db_explain)
        else:
            db_explain.explanation = {"explanations": explanations[0]}
        db.commit()

        return {"status": True, "explanations": explanations[0],
                "message": "Successfully generated explanations"}

    except Exception as exception_error:
        logger.error(f"Error generating visualization explanation: {str(exception_error)}")
        return {"status": False,
                "message": f"Error generating visualization explanation."}


@api.post("/visualize/evaluate")
async def evaluate_visualization(req: VisualizeEvalWebRequest,
                                 db: Session = Depends(get_db),
                                 current_user: User = Depends(get_current_user)) -> dict:
    """Given a visualization code, provide an evaluation of the code"""

    try:
        db_goal = db.query(Goal).filter(Goal.chat_id == req.chat_id, Goal.question == req.goal.question).first()
        if db_goal is None:
            return {"status": False,
                    "message": "Goal not found with the given chat_id and goal"}

        evaluations = lida.evaluate(
            code=req.code,
            goal=req.goal,
            textgen_config=req.textgen_config if req.textgen_config else TextGenerationConfig(
                n=1,
                temperature=0),
            library=req.library)[0]

        # save to database
        db_evaluate = db.query(Evaluate).filter(Evaluate.goal_id == db_goal.id).first()
        if db_evaluate is None:
            db_evaluate = Evaluate(goal_id=db_goal.id,
                                   evaluate={"evaluations": evaluations})
            db.add(db_evaluate)
        else:
            db_evaluate.evaluate = {"evaluations": evaluations}
        db.commit()

        return {"status": True, "evaluations": evaluations,
                "message": "Successfully generated evaluation"}

    except Exception as exception_error:
        logger.error(f"Error generating visualization evaluation: {str(exception_error)}")
        return {"status": False,
                "message": f"Error generating visualization evaluation. {str(exception_error)}"}


@api.post("/visualize/recommend")
async def recommend_visualization(req: VisualizeRecommendRequest,
                                  db: Session = Depends(get_db),
                                  current_user: User = Depends(get_current_user)) -> dict:
    """Given a dataset summary, generate a visualization recommendations"""

    try:
        db_goal = db.query(Goal).filter(Goal.chat_id == req.chat_id, Goal.id == req.goal_id).first()
        if db_goal is None:
            return {"status": False,
                    "message": "Goal not found with the given chat_id and question"}

        textgen_config = req.textgen_config if req.textgen_config else TextGenerationConfig()
        charts = lida.recommend(
            summary=req.summary,
            code=req.code,
            textgen_config=textgen_config,
            library=req.library,
            return_error=True)

        if len(charts) == 0:
            return {"status": False, "message": "No charts generated"}

        # save to database
        db.query(Recommend).filter(Recommend.goal_id == req.goal_id).delete()
        for chart in charts:
            db_recommend = Recommend(goal_id=req.goal_id,
                                     code=chart["code"])
            db.add(db_recommend)
        db.commit()

        return {"status": True, "charts": charts,
                "message": "Successfully generated chart recommendation"}

    except Exception as exception_error:
        logger.error(f"Error generating visualization recommendation: {str(exception_error)}")
        return {"status": False,
                "message": f"Error generating visualization recommendation."}


@api.post("/visualize/conclusion")
async def conclusion_visualization(req: VisualizeConclusionRequest) -> dict:
    """Given a dataset summary, generate a visualization recommendations"""

    try:
        textgen_config = req.textgen_config if req.textgen_config else TextGenerationConfig()
        charts = lida.conclusion(
            summary=req.summary,
            code=req.code,
            goal=req.goal,
            hint=req.hint,
            library=req.library,
            textgen_config=textgen_config
        )

        if len(charts) == 0:
            return {"status": False, "message": "No charts generated"}
        return {"status": True, "charts": charts,
                "message": "Successfully generated chart conclusion"}

    except Exception as exception_error:
        logger.error(f"Error generating visualization conclusion: {str(exception_error)}")

        import traceback
        traceback.print_exception(exception_error)
        return {"status": False,
                "message": f"Error generating visualization conclusion."}


@api.post("/text/generate")
async def generate_text(textgen_config: TextGenerationConfig) -> dict:
    """Generate text given some prompt"""

    try:
        completions = textgen.generate(textgen_config)
        return {"status": True, "completions": completions.text}
    except Exception as exception_error:
        logger.error(f"Error generating text: {str(exception_error)}")
        return {"status": False, "message": f"Error generating text."}


@api.post("/goal")
async def generate_goal(req: GoalWebRequest,
                        db: Session = Depends(get_db),
                        current_user: User = Depends(get_current_user)) -> dict:
    """Generate goals given a dataset summary"""
    try:
        db_chat = db.query(Chat).filter(Chat.id == req.chat_id, Chat.user_id == current_user.id).first()
        if db_chat is None:
            return {"status": False,
                    "message": "Chat not found with the given chat_id"}
        textgen_config = req.textgen_config if req.textgen_config else TextGenerationConfig()
        goals = lida.goals(req.summary, n=req.n, textgen_config=textgen_config, hint=req.extra_hint_interest)

        # save to database
        db.query(Goal).filter(Goal.chat_id == req.chat_id).delete()
        db_chat.extra_hint_interest = req.extra_hint_interest
        for goal in goals:
            db_goal = Goal(chat_id=req.chat_id,
                           index=goal.index,
                           question=goal.question,
                           visualization=goal.visualization,
                           rationale=goal.rationale,
                           is_auto=1)
            db.add(db_goal)
        db.commit()

        return {"status": True, "data": goals,
                "message": f"Successfully generated {len(goals)} goals"}
    except Exception as exception_error:
        logger.error(f"Error generating goals: {str(exception_error)}")
        # Check for a specific error message related to context length
        if "context length" in str(exception_error).lower():
            return {
                "status": False,
                "message": "The dataset you uploaded has too many columns. Please upload a dataset with fewer columns and try again."
            }

        # For other exceptions
        return {
            "status": False,
            "message": f"Error generating visualization goals. {exception_error}"
        }


@api.post("/summarize")
async def upload_file(file: UploadFile = Form(...),
                      data: str = Form(...),
                      chat_id: str = Form(...),
                      db: Session = Depends(get_db),
                      current_user: User = Depends(get_current_user)) -> dict:
    try:
        json_data = json.loads(data)
        data: DescribeData = DescribeData(**json_data)
    except JSONDecodeError:
        return {"status": False,
                "message": f"Data type not matched. {DescribeData.__dict__}"}
    print(file.filename)
    print(data)
    # return {}

    """ Upload a file and return a summary of the data """
    # allow csv, excel, json
    allowed_types = ["text/csv", "application/vnd.ms-excel", "application/json"]

    # print("file: ", file)
    # check file type

    if file.content_type not in allowed_types:
        return {"status": False,
                "message": f"Uploaded file type ({file.content_type}) not allowed. Allowed types are: csv, excel, json"}

    try:

        # save file to files folder
        file_location = os.path.join(data_folder, file.filename)
        # open file without deleting existing contents
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())

        # summarize
        textgen_config = TextGenerationConfig(n=1, temperature=0)
        summary, unused_hint = lida.summarize(
            data=file_location,
            file_name=file.filename,
            summary_method="llm",
            summary_hint=data,
            textgen_config=textgen_config)

        # save to database
        if chat_id != "new":
            db_chat = db.query(Chat).filter(Chat.id == chat_id, Chat.user_id == current_user.id).first()
            if db_chat is None:
                return {"status": False,
                        "message": "Chat not found with the given chat_id"}
            db_chat.data_filename = summary['file_name']
            db_chat.dataset_name = summary['name']
            db_chat.dataset_description = summary['dataset_description']
            db_chat.field_names = {"field_names": summary['field_names']}
            db_chat.fields = {"fields": summary['fields']}
        else:
            db_chat = Chat(user_id=current_user.id,
                           name=summary['name'],
                           data_filename=summary['file_name'],
                           dataset_name=summary['name'],
                           dataset_description=summary['dataset_description'],
                           field_names={"field_names": summary['field_names']},
                           fields={"fields": summary['fields']})
            db.add(db_chat)
        db.commit()
        db.refresh(db_chat)

        ret = {"status": True, "summary": summary, "data_filename": file.filename,
               "chat_id": f"{db_chat.id}"}
        if unused_hint:
            ret["warning"] = {
                "message": ",".join(unused_hint) + " fields description unmatched. ",
                "data": unused_hint
            }
        return ret
    except Exception as exception_error:
        logger.error(f"Error processing file: {str(exception_error)}")
        return {"status": False, "message": f"Error processing file."}


# upload via url
@api.post("/summarize/url")
async def upload_file_via_url(req: SummaryUrlRequest) -> dict:
    """ Upload a file from a url and return a summary of the data """
    url = req.url
    textgen_config = req.textgen_config if req.textgen_config else TextGenerationConfig(
        n=1, temperature=0)
    file_name = url.split("/")[-1]
    file_location = os.path.join(data_folder, file_name)

    # download file
    url_response = requests.get(url, allow_redirects=True, timeout=1000)
    open(file_location, "wb").write(url_response.content)
    try:

        summary = lida.summarize(
            data=file_location,
            file_name=file_name,
            summary_method="llm",
            textgen_config=textgen_config)
        return {"status": True, "summary": summary, "data_filename": file_name}
    except Exception as exception_error:
        # traceback.print_exc()
        logger.error(f"Error processing file: {str(exception_error)}")
        return {"status": False, "message": f"Error processing file."}


# convert image to infographics


@api.post("/infographer")
async def generate_infographics(req: InfographicsRequest) -> dict:
    """Generate infographics using the peacasso package"""
    try:
        result = lida.infographics(
            visualization=req.visualization,
            n=req.n,
            style_prompt=req.style_prompt
            # return_pil=req.return_pil
        )
        return {"status": True, "result": result, "message": "Successfully generated infographics"}
    except Exception as exception_error:
        logger.error(f"Error generating infographics: {str(exception_error)}")
        return {"status": False,
                "message": f"Error generating infographics. {str(exception_error)}"}


# list supported models


@api.get("/models")
def list_models() -> dict:
    return {"status": True, "data": providers, "message": "Successfully listed models"}


@api.get("/getChatList")
def get_chats(skip: int = 0, limit: int = 10, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)) -> dict:
    chats = db.query(Chat).filter(Chat.user_id == current_user.id).offset(skip).limit(limit).all()
    chat_list = [{"id": f"{chat.id}",
                  "name": chat.name,
                  "create_time": chat.create_time,
                  "update_time": chat.update_time} for chat in chats]

    return {"status": True, "chats": chat_list, "message": "Successfully listed chats!"}


@api.get("/getChatInfo/{chat_id}")
def get_chat_info(chat_id: str,
                  db: Session = Depends(get_db),
                  current_user: User = Depends(get_current_user)) -> dict:
    chat = db.query(Chat).filter(Chat.id == chat_id, Chat.user_id == current_user.id).first()
    if not chat:
        return {"status": False, "message": "No such chat found."}

    file_location = os.path.join(data_folder, chat.data_filename)
    lida.open_data(data=file_location)
    summary = {
        "name": chat.dataset_name,
        "file_name": chat.data_filename,
        "dataset_description": chat.dataset_description,
        "fields": chat.fields.get("fields"),
        "field_names": chat.field_names.get("field_names")
    }
    goals = db.query(Goal).filter(Goal.chat_id == chat_id).all()
    goal_list = [{"index": goal.index,
                  "question": goal.question,
                  "visualization": goal.visualization,
                  "rationale": goal.rationale,
                  "is_auto": goal.is_auto,
                  "id": f"{goal.id}"} for goal in goals]

    return {"status": True, "summary": summary,
            "data": goal_list, "message": "Successfully get chatInfo!"}


@api.post("/getGoalInfo")
async def get_chat_info(req: VisWebRequest,
                        db: Session = Depends(get_db),
                        current_user: User = Depends(get_current_user)) -> dict:
    goal = db.query(Goal).join(Chat, Chat.id == Goal.chat_id)\
        .filter(Goal.id == req.goal_id, Chat.user_id == current_user.id).first()
    if not goal:
        return {"status": False, "message": "No such goal found."}

    charts = lida.vis(
        code_specs=[goal.code],
        summary=req.summary,
        library=goal.library,
        return_error=True,
    )

    edits = db.query(Edit).filter(Edit.goal_id == goal.id).all()
    instructions = [{"index": edit.index,
                     "edit": edit.edit} for edit in edits]

    exp = eva = rec = None
    explain = db.query(Explain).filter(Explain.goal_id == goal.id).first()
    evaluate = db.query(Evaluate).filter(Evaluate.goal_id == goal.id).first()
    recommends = db.query(Recommend).filter(Recommend.goal_id == goal.id).all()

    if explain:
        exp = explain.explanation["explanations"]
    if evaluate:
        eva = evaluate.evaluate["evaluations"]
    if recommends:
        recommend_code_specs = [recommend.code for recommend in recommends]
        rec = lida.vis(
            code_specs=recommend_code_specs,
            summary=req.summary,
            library=goal.library,
            return_error=True,
        )

    return {"status": True, "charts": charts,
            "instructions": instructions,
            "explanations": exp,
            "evaluations": eva,
            "recommend": rec,
            "message": "Successfully get chatInfo!"}

# 创建任务
@api.post("/tasks/create", response_model=dict)
async def create_task(
    req: TaskCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> dict:
    """Create a new task with the provided details."""

    try:
        # 创建新的任务条目
        db_task = TaskManagement(
            task_name=req.task_name,
            task_details=req.task_details,
            chat_id=req.chat_id,
        )
        
        # 添加到数据库会话并提交
        db.add(db_task)
        db.commit()
        db.refresh(db_task)

        return {
            "status": True,
            "data": {
                "task_id": db_task.task_id,  # 使用正确的字段名
                "name": db_task.task_name,
                "details": db_task.task_details,
                "chat_id": db_task.chat_id,
                "created_at": db_task.created_at,
                "updated_at": db_task.updated_at
            },
            "message": "Successfully created task!"
        }

    except Exception as exception_error:
        logging.error(f"Error creating task: {str(exception_error)}")
        return {
            "status": True,
            "message": f"Error creating task. {exception_error}"
        }

# 查询所有任务
@api.get("/tasks/getAll", response_model=dict)
def get_all_tasks(
    skip: int = Query(0, description="Skip number of tasks"),
    limit: int = Query(10, description="Number of tasks to return"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> dict:
    """Retrieve a list of all tasks with pagination."""
    
    try:

        # 查询所有任务并应用分页
        tasks = db.query(TaskManagement).offset(skip).limit(limit).all()
        
        # 构建任务列表，确保使用正确的属性名
        task_list = [
            {
                "task_id": task.task_id,  # 使用正确的属性名 task_id
                "task_name": task.task_name,
                "task_details": task.task_details,
                "chat_id": task.chat_id,
                "created_at": task.created_at,
                "updated_at": task.updated_at
            } 
            for task in tasks
        ]

        # 返回包含状态、任务列表和消息的字典
        return {"status": True, "tasks": task_list, "message": "Successfully listed all tasks!"}

    except Exception as exception_error:
        logging.error(f"Error fetching tasks: {str(exception_error)}")
        return {
            "status": False,
            "message": f"Error fetching tasks. {exception_error}"
        }
    

# 根据名字查询任务
@api.get("/tasks/getName", response_model=dict)
def get_task_by_name(
    task_name: str,
    skip: int = Query(0, description="Skip number of tasks"),
    limit: int = Query(10, description="Number of tasks to return"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> dict:
    """Retrieve a list of all tasks with pagination."""
    
    try:

        # 构建查询基础
        query = db.query(TaskManagement)

        # 如果提供了 task_name 参数，则应用过滤
        if task_name:
            query = query.filter(TaskManagement.task_name.ilike(f"%{task_name}%"))

        # 应用分页参数
        paginated_tasks = query.offset(skip).limit(limit).all()
        
        # 构建任务列表，确保使用正确的属性名
        task_list = [
            {
                "task_id": task.task_id,  # 使用正确的属性名 task_id
                "task_name": task.task_name,
                "task_details": task.task_details,
                "chat_id": task.chat_id,
                "created_at": task.created_at,
                "updated_at": task.updated_at
            } 
            for task in paginated_tasks
        ]

        # 返回包含状态、任务列表和消息的字典
        return {"status": True, "tasks": task_list, "message": "Successfully listed name tasks!"}

    except Exception as exception_error:
        logging.error(f"Error fetching tasks: {str(exception_error)}")
        return {
            "status": False,
            "message": f"Error fetching tasks. {exception_error}"
        }


# 登录获取令牌
@api.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)) -> dict:
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        return {"status": False, "message": "Incorrect username or password"}
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user.username}, expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer"}


@api.post("/register")
async def register_user(user: UserCreate, db: Session = Depends(get_db)) -> dict:
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        return {"status": False, "message": "Username already registered"}
    create_user(db, user.username, user.password)
    return {"status": True, "message": "Successfully registered user!"}
