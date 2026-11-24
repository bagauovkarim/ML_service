from sklearn.model_selection import train_test_split
from bank_marketing_service.dataset import load_training_data
from bank_marketing_service.preprocessing import get_preprocessor_scaled
from bank_marketing_service.registry import save_model
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import logging
import os
import requests
import time

logger = logging.getLogger(__name__)


def main():
    X, y = load_training_data(os.environ["TRAIN_DATA_PATH"])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=float(os.environ["VALIDATION_SIZE"]), shuffle=False
    )

    pipeline = Pipeline([
        ("prep", get_preprocessor_scaled()),
        ("model", LogisticRegression(random_state=int(os.environ["MODEL_RANDOM_STATE"]), C=float(os.environ["MODEL_C"]), 
                                     solver=os.environ["MODEL_SOLVER"], max_iter=int(os.environ["MODEL_MAX_ITER"]), l1_ratio=float(os.environ["MODEL_L1_RATIO"])))
    ])

    pipeline.fit(X_train, y_train)

    probs = pipeline.predict_proba(X_val)[:, 1]

    auc = roc_auc_score(y_val, probs)

    threshhold = float(os.environ["MIN_ROC_AUC"])

    logger.info("Validation ROC-AUC: %.3f; minimum: %.3f", auc, threshhold)

    if (auc < threshhold):
        logger.error("Error: auc < threshold")
        raise SystemExit(1)

    logger.info("Model passed validation")

    model_path = save_model(pipeline, os.environ["MODEL_REGISTRY_DIR"])
    logger.info("Model saved: %s", model_path)
    activate_inference(model_path)
    logger.info("Model switched: %s", model_path)



def activate_inference(model_path):
    max_attempts = int(os.environ["ACTIVATION_MAX_ATTEMPTS"])
    delay_seconds = float(os.environ["ACTIVATION_RETRY_DELAY_SECONDS"])


    url = os.environ["INFERENCE_URL"].rstrip("/") + "/models/activate"

    if max_attempts < 1:
        raise ValueError("ACTIVATION_MAX_ATTEMPTS must be at least 1")
    if delay_seconds < 0:
        raise ValueError("ACTIVATION_RETRY_DELAY_SECONDS cannot be negative")

    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.post(url, json={"model_name": model_path.name}, timeout=float(os.environ["INFERENCE_TIMEOUT_SECONDS"]))
            response.raise_for_status()
            return
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            if attempt == max_attempts:
                logger.exception("Could not activate model after %d attempts", max_attempts)
                raise
            logger.warning(
                "Activation attempt %d/%d failed; retrying in %s seconds",
                attempt, max_attempts, delay_seconds
            )
            time.sleep(delay_seconds)
         


if __name__ == "__main__":
    logging.basicConfig(level=os.environ["LOG_LEVEL"])
    main()