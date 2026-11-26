from pydantic import BaseModel, Field, ConfigDict
from typing import Literal

class PredictionRequest(BaseModel):
    model_config=ConfigDict(extra="forbid", allow_inf_nan=False)
    age: int = Field(ge=17, le=120)
    campaign: int = Field(ge=1)
    pdays: int = Field(ge=0)
    previous: int = Field(ge=0)
    euribor3m: float
    emp_var_rate: float = Field(alias="emp.var.rate")
    cons_price_idx: float = Field(alias="cons.price.idx")
    cons_conf_idx: float = Field(alias="cons.conf.idx")
    nr_employed: float = Field(alias="nr.employed")
    job: Literal["admin.", "blue-collar", "entrepreneur", "housemaid",
                 "management", "retired", "self-employed", "services",
                 "student", "technician", "unemployed", "unknown"]
    marital: Literal["divorced", "married", "single", "unknown"]
    education: Literal[
        "illiterate", "basic.4y", "basic.6y", "basic.9y",
        "high.school", "professional.course", "university.degree", "unknown"
    ]
    default: Literal["no", "yes", "unknown"]
    housing: Literal["no", "yes", "unknown"]
    loan: Literal["no", "yes", "unknown"]
    contact: Literal["cellular", "telephone"]
    month: Literal["apr", "aug", "dec", "jul", "jun", "mar", "may", "nov", "oct", "sep"]
    day_of_week: Literal["fri", "mon", "thu", "tue", "wed"]
    poutcome: Literal["failure", "nonexistent", "success"]

class PredictionResponse(BaseModel):
    prediction: int
    probability: float

class ModelSwitchRequest(BaseModel):
    model_config=ConfigDict(extra="forbid")
    model_name: str