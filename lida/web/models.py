from pydantic import BaseModel, Json


class ChatCreate(BaseModel):
    name: str


class Token(BaseModel):
    access_token: str
    token_type: str
