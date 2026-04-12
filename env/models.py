from pydantic import BaseModel

class Observation(BaseModel):
    ticket_text: str
    history: list

class Action(BaseModel):
    category: str  # billing, tech, general

class Reward(BaseModel):
    value: float