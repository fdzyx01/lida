import csv
from http.client import HTTPException
import io
import json
import os
import logging
from datetime import timedelta
from json import JSONDecodeError
from typing import Dict, List, Optional

import requests
from fastapi import FastAPI, UploadFile, Form, Depends, Query, File, status
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import traceback

from llmx import llm, providers
from sqlalchemy import delete, desc
from sqlalchemy.orm import Session

from .auth import authenticate_user, ACCESS_TOKEN_EXPIRE_MINUTES, create_access_token, create_user, get_current_user
from .entity import JsonDataStorage, SessionLocal, Chat, Goal, Explain, Evaluate, Edit, TaskManagement, User, Recommend
from pydantic import BaseModel

from .models import Token
from ..datamodel import ChatUpdateRequest, GoalUpdateExplanationRequest, GoalWebRequest, JsonDataStorageCreateRequest, SummaryUrlRequest, TaskCreateRequest, TaskNameUpdateRequest, TextGenerationConfig, UploadUrl, VisualizeEditWebRequest, \
    VisualizeEvalWebRequest, VisualizeExplainWebRequest, VisualizeRecommendRequest, VisualizeRepairWebRequest, \
    VisualizeWebRequest, InfographicsRequest, VisualizeConclusionRequest, DescribeData, UserCreate, VisWebRequest
from ..components import Manager, picture_result_generate
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
    allow_origins=["*"],
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
        image_base64 = charts[0].raster
        image_url = f'data:image/jpeg;base64,{image_base64}'
        try:
            content = picture_result_generate(image_url).content
        except:
            content = "error"
        charts[0].picture_result = content
        # print(content)
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
                           code=charts["code"],
                           raster=charts[0].raster,
                           picture_result=charts[0].picture_result 
                           )
        else:
            db_goal.library = req.library
            db_goal.code = charts[0].code
            db_goal.raster = charts[0].raster
            db_goal.picture_result = charts[0].picture_result
        db.commit()
        db.refresh(db_goal)

        return {"status": True, "charts": charts,
                "message": "Successfully generated charts.",
                "goal_id": f"{db_goal.id}"}

    except Exception as exception_error:
        logger.error(f"Error generating visualization goals: {str(exception_error)}")
        return {"status": False,
                "message": f"Error generating visualization goals. {str(exception_error)}"}

class VisualizeUpdatePictureInput(BaseModel):
    chat_id: str
    goal_id: str
    raster: Optional[str] = None
    picture_result: Optional[str] = None

@api.post("/visualize/update_picture")
async def update_picture(req: VisualizeUpdatePictureInput,
                             db: Session = Depends(get_db),
                             current_user: User = Depends(get_current_user)) -> dict:
    try:
        db_goal = db.query(Goal).filter(Goal.chat_id == req.chat_id, Goal.id == req.goal_id).first()
        if db_goal is None:
            return {"status": False,
                    "message": "Goal not found with the given chat_id and question"}
        if req.raster is not None:
            db_goal.raster = req.raster
        if req.picture_result is not None:
            db_goal.picture_result = req.picture_result
        db.commit()
        db.refresh(db_goal)
        return {"status": True,
                "message": "Successfully updated picture."}
    except Exception as exception_error:
        logger.error(f"Error updating picture: {str(exception_error)}")
        return {"status": False,
                "message": f"Error updating picture. {str(exception_error)}"}


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
        image_base64 = charts[0].raster
        image_url = f'data:image/jpeg;base64,{image_base64}'
        try:
            content = picture_result_generate(image_url).content
        except:
            content = "error"
        charts[0].picture_result = content
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
            db_chat.name = summary['name']
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
        with db.begin():
            db_chat = Chat(user_id=current_user.id)
            db.add(db_chat)
            db.flush()

            # 创建新的任务条目
            db_task = TaskManagement(
                task_name=req.task_name,
                task_details=req.task_details,
                chat_id=db_chat.id, # 此处 chat_id 已经由数据库自动生成并赋值
            )

            
            # 添加到数据库会话并提交
            db.add(db_task)
            db.flush()

            return {
                "status": True,
                "data": {
                    "task_id": db_task.task_id,  # 使用正确的字段名
                    "name": db_task.task_name,
                    "details": db_task.task_details,
                    "chat_id": str(db_task.chat_id),
                    "created_at": db_task.created_at,
                    "updated_at": db_task.updated_at
                },
                "message": "Successfully created task!"
            }

    except Exception as exception_error:
        db.rollback()
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

        # 查询所有任务并应用分页和排序
        tasks_query = db.query(TaskManagement).order_by(desc(TaskManagement.created_at))
        tasks = tasks_query.offset(skip).limit(limit).all()

        # 计算总的任务条数（不带分页）
        total_tasks = tasks_query.count()
        
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
        return {"status": True, "tasks": task_list, "total": total_tasks, "message": "Successfully listed all tasks!"}

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
        # 计算总的任务条数
        total_tasks = query.count()

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
        return {"status": True, "tasks": task_list, "total": total_tasks, "message": "Successfully listed name tasks!"}

    except Exception as exception_error:
        logging.error(f"Error fetching tasks: {str(exception_error)}")
        return {
            "status": False,
            "message": f"Error fetching tasks. {exception_error}"
        }

# 根据chat_id修改任务表task_name
@api.put("/tasks/updateTaskNameByChatId", response_model=dict)
def update_task_name_by_chat_id(
    req: TaskNameUpdateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> dict:
    """Update the task_name field for all records matching the provided chat_id."""
    try:
        if not current_user:
            raise HTTPException(status_code=401, detail="User not authenticated")

        logging.info(f"Received request to update task_name for chat_id={req.chat_id}")

        with db.begin():
            # 查询与 chat_id 匹配的所有记录
            tasks = (
                db.query(TaskManagement)
                .filter(TaskManagement.chat_id == req.chat_id)
                .all()
            )

            if not tasks:
                logging.warning(f"No matching tasks found for chat_id={req.chat_id}")
                return {"status": False, "message": "No matching tasks found."}

            # 更新每个匹配记录的 task_name 字段
            updated_count = 0
            for task in tasks:
                task.task_name = req.new_task_name
                updated_count += 1

        logging.info(f"Successfully updated {updated_count} tasks for chat_id={req.chat_id}")

        return {
            "status": True,
            "message": f"Successfully updated {updated_count} tasks.",
            "updated_count": updated_count
        }

    except HTTPException as http_error:
        logging.error(f"HTTP error for chat_id {req.chat_id}: {str(http_error)}")
        raise http_error
    except Exception as exception_error:
        db.rollback()
        error_message = f"Error updating task_name for chat_id {req.chat_id}: {str(exception_error)}"
        logging.error(error_message)
        return {"status": False, "message": error_message}


# 根据chat_id查询最新的一条goals表中的记录
@api.get("/goals/getLatestIdByChatId", response_model=dict)
def get_latest_id_by_chat_id(
    chat_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> dict:
    """Retrieve the latest goal ID filtered by chat ID."""
    
    try:
        # 查询与 chat_id 匹配的最新记录的 id
        latest_goal = (
            db.query(Goal)
            .filter(Goal.chat_id == chat_id)
            .order_by(Goal.update_time.desc())  # 或者使用 create_time.desc()
            .first()
        )
        
        if not latest_goal:
            return {"status": False, "message": "No goals found for the given chat ID."}

        # 返回包含状态、单个目标记录和消息的字典
        return {
            "status": True,
            "goal": {
                "id": latest_goal.id,
                "chat_id": latest_goal.chat_id,
                "index": latest_goal.index,
                "question": latest_goal.question,
                "visualization": latest_goal.visualization,
                "rationale": latest_goal.rationale,
                "is_auto": bool(latest_goal.is_auto),  # Assuming is_auto is stored as integer in the database
                "library": latest_goal.library,
                "code": latest_goal.code,
                "explanation": latest_goal.explanation,
                "create_time": latest_goal.create_time.isoformat() if latest_goal.create_time else None,
                "update_time": latest_goal.update_time.isoformat() if latest_goal.update_time else None
            },
            "message": "Successfully retrieved the latest goal!"
        }

    except Exception as exception_error:
        logging.error(f"Error fetching latest goal: {str(exception_error)}")
        return {
            "status": False,
            "message": f"Error fetching latest goal.{exception_error}"
        }

# 创建json_data_storage 两条json数据
@api.post("/jsonDataStorage/create", response_model=dict)
def create_json_data_storage(
    req:JsonDataStorageCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> dict:
    """Create a new JSON data storage record."""
    
    try:
        if not current_user:
            raise HTTPException(status_code=401, detail="User not authenticated")
        
         # 验证输入参数
        if not req.chat_id or not isinstance(req.chat_id, str):
            raise ValueError("Invalid chat_id")
        
        if not req.json_table1 or not isinstance(req.json_table1, dict):
            raise ValueError("Invalid json_table1")
        
        if not req.json_table2 or not isinstance(req.json_table2, dict):
            raise ValueError("Invalid json_table2")

        with db.begin():  # 使用上下文管理器确保事务管理
            # 创建新的 JsonDataStorage 实例并填充数据
            new_record = JsonDataStorage(
                chat_id=req.chat_id,
                json_table1=req.json_table1,
                json_table2=req.json_table2
            )

            # 添加到数据库会话
            db.add(new_record)
            db.flush() # 刷新新记录，获取其自增ID

            # 提交后刷新实例以获取最新数据
            db.refresh(new_record)

        return {
            "status": True,
            "data": {
                "id": new_record.id,  # 使用正确的字段名
                "chat_id": new_record.chat_id,
                "json_table1": new_record.json_table1,
                "json_table2": new_record.json_table2,
                "created_at": new_record.created_at,
                "updated_at": new_record.updated_at
            },
            "message": "Successfully created jsonDataStorage record!"
        }

    except HTTPException as http_error:
        logging.error(f"HTTP error creating JsonDataStorage record for user {current_user.id}: {str(http_error)}")
        raise http_error
    except Exception as exception_error:
        db.rollback()
        error_message = f"Error creating JsonDataStorage record: {str(exception_error)}"
        logging.error(error_message)
        return {
            "status": False,
            "message": error_message
        }
    
# 根据更新goal表中 goal_id和chat_id都匹配的记录，更新explanation字段为success
@api.put("/goals/updateExplanation", response_model=dict)
def update_goal_explanation(
    req: GoalUpdateExplanationRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> dict:
    """Update the explanation field for a specific goal record based on goal_id and chat_id."""
    try:
        if not current_user:
            raise HTTPException(status_code=401, detail="User not authenticated")

        with db.begin():  # 使用上下文管理器确保事务管理
            # 查询与 goal_id 和 chat_id 匹配的记录
            goal = (
                db.query(Goal)
                .filter(Goal.id == req.id, Goal.chat_id == req.chat_id)
                .first()
            )
            
            if not goal:
                return {"status": False, "message": "No matching goal found."}

            # 更新 explanation 字段
            goal.explanation = "success"

            # 强制刷新会话以确保更改被追踪
            db.flush()

         # 构建返回的响应数据
        updated_goal_data = {
            "id": goal.id,
            "chat_id": goal.chat_id,
            "index": goal.index,
            "question": goal.question,
            "visualization": goal.visualization,
            "rationale": goal.rationale,
            "is_auto": bool(goal.is_auto),  # Assuming is_auto is stored as integer in the database
            "library": goal.library,
            "code": goal.code,
            "explanation": goal.explanation,
            "create_time": goal.create_time.isoformat() if goal.create_time else None,
            "update_time": goal.update_time.isoformat() if goal.update_time else None
            
        }

        return {
            "status": True,
            "message": "Successfully updated the explanation.",
            "goal": updated_goal_data
        }

    except HTTPException as http_error:
        logging.error(f"HTTP error for goal_id {req.id} and chat_id {req.chat_id}: {str(http_error)}")
        raise http_error
    except Exception as exception_error:
        db.rollback()
        error_message = f"Error updating explanation for goal_id {req.id} and chat_id {req.chat_id}: {str(exception_error)}"
        logging.error(error_message)
        return {
            "status": False,
            "message": error_message
        }
    
# 根据chat_id查询出表中explanation字段为success的goal记录 并且查询 t_chat表中的相关信息
@api.get("/goals/getSuccessGoalsByChatId", response_model=dict)
def get_success_goals_by_chat_id(
    chat_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> List[dict]:
    """Fetch all goal records with explanation 'success' for a specific chat_id."""
    try:
        if not current_user:
            raise HTTPException(status_code=401, detail="User not authenticated")

        logging.info(f"Received request to fetch success goals for chat_id={chat_id}")

        # 查询 Chat 表中的信息
        chat_info = db.query(Chat).filter(Chat.id == chat_id).first()
        if not chat_info:
            raise HTTPException(status_code=404, detail=f"Chat with id {chat_id} not found")

        # 构建查询基础并获取所有目标
        goals = db.query(Goal).filter(Goal.chat_id == chat_id, Goal.explanation == "success").all()

        logging.info(f"Found {len(goals)} success goals for chat_id={chat_id}")
        
        # 构建返回的响应数据
        response_data = [
            {
                "goal": {
                    "index": goal.index,
                    "question": goal.question,
                    "visualization": goal.visualization,
                    "rationale": goal.rationale,
                },
                
                "name": chat_info.name,
                "file_name": chat_info.data_filename,
                "dataset_description": chat_info.dataset_description,
                "field_names": chat_info.field_names,
                "fields": chat_info.fields,
                
            }
            for goal in goals
        ]

        # 前端不要status和message
        return {
            "summary": response_data,
        }

    except HTTPException as http_error:
        logging.error(f"HTTP error for chat_id {chat_id}: {str(http_error)}")
        raise http_error
    except Exception as exception_error:
        error_message = f"Error fetching success goals for chat_id {chat_id}: {str(exception_error)}"
        logging.error(error_message)
        raise HTTPException(status_code=500, detail=error_message)

# 根据chat_id去修改t_chat表中的fields字段
@api.put("/chats/updateByChatId", response_model=dict)
def update_chat_by_chat_id(
    chat_update_request: ChatUpdateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> dict:
    """Update the fields field for a specific chat record based on chat_id."""
    
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not authenticated")

    chat_id = chat_update_request.chat_id
    fields_update = chat_update_request.fields

    # 查询指定 chat_id 的 Chat 记录
    chat = db.query(Chat).filter(Chat.id == chat_id).first()
    if not chat:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Chat with id {chat_id} not found")

    try:
        # 更新 Chat 记录中的 'fields' 字段
        chat.fields = fields_update

        # 提交更改
        db.commit()
        db.refresh(chat)
        
    except Exception as db_error:
        error_message = f"Database error updating fields for chat with id {chat_id}: {str(db_error)}"
        logging.error(error_message)
        db.rollback()  # 回滚事务以保持数据库一致性
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_message
        )

    return {"status": "success", "message": "Fields updated successfully"}


# 解析csv文件数据  并且把解析的数据插入到 table1字段中
@api.post("/jsonDataStorage/parseCsvData", response_model=dict)
async def parse_csv_data_and_store(
    chat_id: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> dict:
    """Upload a CSV file and store its parsed data along with the chat_id in the database."""
    try:
        if not current_user:
            raise HTTPException(status_code=401, detail="User not authenticated")

        # 确保文件是CSV格式
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Invalid file format. Only CSV files are allowed.")

        # 异步读取文件内容并创建一个 StringIO 对象
        contents = await file.read()
        csv_file = io.StringIO(contents.decode('utf-8'))

        # 使用 DictReader 逐行解析 CSV 文件
        reader = csv.DictReader(csv_file)
        parsed_data = []
        max_rows = 100  # 设置最大行数为100
        row_count = 0

        for row in reader:
            if row_count >= max_rows:
                break
            parsed_row = {key: value for key, value in row.items()}
            parsed_data.append(parsed_row)
            row_count += 1

        # 获取所有列名（即使只读取了部分行）
        columns = reader.fieldnames if reader.fieldnames else []

        # 构建返回的响应数据
        response_data = {
            "data": parsed_data,
            "columns": columns  # 包含实际的列名
        }

        # 确保插入符合预期的数据结构
        wrapped_parsed_data = {"data": parsed_data}

       # 使用上下文管理器确保事务管理
        try:
            with db.begin():  # 使用上下文管理器自动管理事务
                # 插入到数据库中
                new_record = JsonDataStorage(
                    chat_id=chat_id,
                    json_table1=wrapped_parsed_data,  # 将解析的数据插入到 json_table1 字段
                    json_table2={}  # 可以保持为空或根据需要插入其他数据
                )
                db.add(new_record)
                db.flush()  # 刷新以获取自增ID和其他可能的默认值
                db.refresh(new_record)  # 刷新实例以获取最新数据
        except Exception as db_error:
            error_message = f"Database error processing CSV file for chat_id {chat_id}: {str(db_error)}"
            logging.error(error_message)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_message)

        return response_data  # 直接返回字典，FastAPI 会自动将其转换为 JSON 响应

    except HTTPException as http_error:
        logging.error(f"HTTP error for chat_id {chat_id}: {str(http_error)}")
        raise http_error
    except Exception as e:
        await db.rollback()  # 使用异步数据库操作（如果支持）
        error_message = f"Error processing CSV file for chat_id {chat_id}: {str(e)}"
        logging.error(error_message)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_message)

# 登录获取令牌
@api.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)) -> dict:
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        return {"status": False, "message": "Incorrect username or password"}
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user.username}, expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer", "user_id": str(user.id)}


@api.post("/register")
async def register_user(user: UserCreate, db: Session = Depends(get_db)) -> dict:
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        return {"status": False, "message": "Username already registered"}
    create_user(db, user.username, user.password)
    return {"status": True, "message": "Successfully registered user!"}



class deleteTaskInput(BaseModel):
    taskIdList: List[int]

@api.post("/deleteTask")
async def deleteTask(
    req: deleteTaskInput,
    db: Session = Depends(get_db), 
    current_user: User = Depends(get_current_user)
): 
    try:
        # 开始事务
        with db.begin():  # 或者使用 db.begin_nested()，如果你需要嵌套事务
            chatIdList = [task.chat_id for task in db.query(TaskManagement).filter(TaskManagement.task_id.in_(req.taskIdList)).all()]
            for chatId in chatIdList:
                chat = db.query(Chat).filter(Chat.id == chatId, Chat.user_id == current_user.id).first()
                if (chat is None):
                    return {"status": False, "message": f"存在不属于当前用户的记录，删除失败，事务已回滚！"}
                # 修正逻辑，确保 chatid 是一个列表
                db.delete(chat)
                goal = db.query(Goal).filter(Goal.chat_id == chat.id).first()
                if goal:
                    db.delete(goal)
                    # 删除与 goal 相关的其他表数据
                    edit = db.query(Edit).filter(Edit.goal_id == goal.id).first()
                    if edit:
                        db.delete(edit)
                    explain = db.query(Explain).filter(Explain.goal_id == goal.id).first()
                    if explain:
                        db.delete(explain)
                    evaluate = db.query(Evaluate).filter(Evaluate.goal_id == goal.id).first()
                    if evaluate:
                        db.delete(evaluate)
                    recommend = db.query(Recommend).filter(Recommend.goal_id == goal.id).first()
                    if recommend:
                        db.delete(recommend)
                
                db.query(TaskManagement).filter(TaskManagement.chat_id == str(chat.id)).delete()
            
    except Exception as e:
        # 捕获数据库异常并回滚事务
        db.rollback()
        return {"status": False, "message": f"删除失败，事务已回滚: {str(e)}"}

    return {"status": True, "message": "Successfully deleted chat record!"}



@api.get("/select_json_data_storage")
async def select_json_data_storage(
    chat_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        json_data_storage = db.query(JsonDataStorage).filter(JsonDataStorage.chat_id == str(chat_id)).first()
    except Exception as e:
        return {"status": False, "message": f"Failed to select json_data_storage: {str(e)}"}
    return {"status": True, "json_data_storage": json_data_storage, "message": "Successfully get json_data_storage!"}

class JsonDataStorageInput(BaseModel):
    chat_id: str
    json_table2: dict
@api.post("/update_json_data_storage")
async def update_json_data_storage(
    req: JsonDataStorageInput,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        with db.begin():
            # 查询符合条件的第一条记录
            record = db.query(JsonDataStorage).filter(JsonDataStorage.chat_id == str(req.chat_id)).first()
            if record is None:
                return {"status": False, "message": "没有该 chat_id 对应的记录！"}
            # 更新第一条记录的字段
            record.json_table2 = req.json_table2
            db.add(record)  # 将修改的记录添加到会话中（可选，通常修改后 SQLAlchemy 会自动跟踪）
    except Exception as e:
        db.rollback()  # 回滚事务
        return {"status": False, "message": f"Failed to update json_data_storage: {str(e)}"}
    # 返回成功消息
    return {"status": True, "message": "Successfully updated json_data_storage!"}



@api.get("/select_table_and_fields_by_chatid")
async def select_table_and_fields_by_chatid(
    chat_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        # 查询 chat
        chat = db.query(Chat).filter(Chat.id == str(chat_id)).first()
        chat_data = None
        if chat:
            # 处理 chat.fields 和 chat.field_names 为 None 的情况
            chat_fields = chat.fields if chat.fields is not None else []
            chat_field_names = chat.field_names if chat.field_names is not None else []
            chat_data = {
                "field_names": chat_field_names,
                "fields": chat_fields
            }

        # 查询 jsonData
        jsonData = db.query(JsonDataStorage).filter(JsonDataStorage.chat_id == str(chat_id)).first()
        json_data = None
        if jsonData:
            # 处理 jsonData.json_table1 为 None 的情况
            json_table1 = jsonData.json_table1 if jsonData.json_table1 is not None else {}
            json_data = {
                "json_table1": json_table1
            }

        # 返回结果
        return {
            "status": True,
            "data": {
                "chat": chat_data,
                "jsonData": json_data
            }
        }
    except Exception as e:
        return {"status": False, "message": f"Failed to select table and fields: {str(e)}"}
    

@api.get("/ishava_task_byName")
async def ishava_task_byName(
    task_name: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        task_management = db.query(TaskManagement).filter(TaskManagement.task_name == str(task_name)).first()
        if task_management is None:
            return {"status": False, "data": {"message": "任务名称不存在！", "task_management": None}}
        else:
            return {"status": True,  "data": {"message": "任务名称已存在！", "task_management": task_management}}
    except Exception as e:
        return {"status": False, "message": f"Failed to select table and fields: {str(e)}"}
