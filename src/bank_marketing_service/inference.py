from fastapi import FastAPI
import os 
from contextlib import asynccontextmanager
from bank_marketing_service.registry import load_model, get_model_path, save_active_model_name, read_active_model_name
from fastapi import Request, HTTPException
import pandas as pd 
from bank_marketing_service.schemas import PredictionRequest, PredictionResponse, ModelSwitchRequest
from bank_marketing_service.preprocessing import prepare_features
import logging
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from pathlib import Path
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from bank_marketing_service.settings import Settings

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    app.state.settings = settings
    logging.basicConfig(level=settings.log_level)
    registry_dir = settings.model_registry_dir
    app.state.model = None
    app.state.model_name = None
    try:
        saved_name = read_active_model_name(registry_dir)
        configured_path = settings.active_model_path.strip()
        if saved_name is not None:
            path = get_model_path(registry_dir, saved_name)
        elif configured_path:
            path = Path(configured_path)
        else:
            path = None
        
        if path is None:
            logger.warning("Service was started without active model")
        else:
            loaded_model = load_model(path)
            
            if not isinstance(loaded_model, Pipeline):
                raise TypeError("Stored model is not a Pipeline")
            
            check_is_fitted(loaded_model)
            app.state.model = loaded_model
            app.state.model_name = Path(path).name 
            logger.info("Active model loaded: %s", app.state.model_name)

    except Exception:
        logger.exception("Failed to load active model at startup")
    
    yield

app = FastAPI(lifespan=lifespan)

@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    errors = []

    for error in exc.errors():
        errors.append({
            "loc": list(error["loc"]),
            "msg": error["msg"],
            "type": error["type"],
        })
    return JSONResponse(status_code=422, content={"detail": errors})

@app.get("/health")
def health(request: Request):
    if request.app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "active_model": request.app.state.model_name}

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request):

    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        table = pd.DataFrame([payload.model_dump(by_alias=True)])
        table_prepared = prepare_features(table)
        prediction = model.predict(table_prepared)
        probability = model.predict_proba(table_prepared)
        
        return {
            "prediction": int(prediction[0]),
            "probability": float(probability[0, 1])
        }
    
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed")



@app.post("/models/activate")
def switch_model(payload: ModelSwitchRequest, request: Request):
    settings = request.app.state.settings
    try:
        model_path = get_model_path(settings.model_registry_dir, payload.model_name)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid model name")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model file not found")
    
    try:
        new_model = load_model(model_path)
    except Exception:
        logger.exception("Failed to load model: %s", payload.model_name)
        raise HTTPException(status_code=500, detail="Failed to load model")

    if not isinstance(new_model, Pipeline):
        logger.error("Invalid model type: %s", payload.model_name)
        raise HTTPException(status_code=500, detail="Invalid model type")
    
    try:
        check_is_fitted(new_model)
    except NotFittedError:
        logger.exception("Model is not fitted: %s", payload.model_name)
        raise HTTPException(status_code=500, detail="Model is not fitted")

    try:
        save_active_model_name(settings.model_registry_dir, payload.model_name)
    except OSError:
        logger.exception("Failed to save active model: %s", payload.model_name)
        raise HTTPException(status_code=500, detail="Failed to save active model")
    
    request.app.state.model = new_model
    request.app.state.model_name = payload.model_name
    logger.info("Active model switched to %s", payload.model_name)
    
    return {    
        "active_model": payload.model_name
    }