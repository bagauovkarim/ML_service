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
from bank_marketing_service.settings import Settings

logger = logging.getLogger(__name__)


def main():
    settings = Settings()
    logging.basicConfig(level=settings.log_level)

    X, y = load_training_data(settings.train_data_path)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=settings.validation_size, shuffle=False
    )

    pipeline = Pipeline([
        ("prep", get_preprocessor_scaled()),
        ("model", LogisticRegression(random_state=settings.model_random_state, C=settings.model_c,           
                                     solver=settings.model_solver, max_iter=settings.model_max_iter, l1_ratio=settings.model_l1_ratio))
    ])

    pipeline.fit(X_train, y_train)

    probs = pipeline.predict_proba(X_val)[:, 1]

    auc = roc_auc_score(y_val, probs)

    threshhold = settings.min_roc_auc

    logger.info("Validation ROC-AUC: %.3f; minimum: %.3f", auc, threshhold)

    if (auc < threshhold):
        logger.error("Error: auc < threshold")
        raise SystemExit(1)

    logger.info("Model passed validation")

    model_path = save_model(pipeline, settings.model_registry_dir)
    logger.info("Model saved: %s", model_path)
    activate_inference(model_path, settings)
    logger.info("Model switched: %s", model_path)



def activate_inference(model_path, settings: Settings):
    max_attempts = settings.activation_max_attempts
    delay_seconds = settings.activation_retry_delay_seconds


    url = str(settings.inference_url).rstrip("/") + "/models/activate"

    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.post(url, json={"model_name": model_path.name}, timeout=settings.inference_timeout_seconds)
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
    main()