from pydantic import BaseModel,Field
from typing import Annotated 
class Message(BaseModel):
    message: Annotated[str, Field(..., description='The message to be written', example='You have earned a 9 lakhs bonus this year!')]