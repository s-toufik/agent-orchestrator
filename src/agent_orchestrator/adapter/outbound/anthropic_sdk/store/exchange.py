from pydantic import BaseModel


class Exchange(BaseModel):
    user: str
    assistant: str
