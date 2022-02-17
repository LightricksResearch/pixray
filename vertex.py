import logging
import os
import sys
from pathlib import Path
import tempfile, shutil
import joblib
from fastapi import Request, FastAPI
import pixray

READY = False
app = FastAPI()
logger = logging.getLogger(__name__)

AIP_HEALTH_ROUTE = os.environ.get("AIP_HEALTH_ROUTE", "/health_check")
AIP_PREDICT_ROUTE = os.environ.get("AIP_PREDICT_ROUTE", "/predict")
MODEL_ARTIFACTS = os.environ.get("AIP_STORAGE_URI", ".")


def create_temporary_copy(src_path):
    _, tf_suffix = os.path.splitext(src_path)
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"tempfile{tf_suffix}")
    shutil.copy2(src_path, temp_path)
    return temp_path


logger.info(f"Downloading artifacts from:{MODEL_ARTIFACTS}")
# local_files = download_folder(MODEL_ARTIFACTS)
# logger.info(f"Downloading artifacts done. Downloaded files: {local_files}")

# model = load_joblib(local_files['model.joblib'])
logger.info("model loaded")

# user_item_sparse = load_rec_matrix(local_files['user_item_csv.csv'])
logger.info("loaded ALS matrix")

READY = True


@app.get(AIP_HEALTH_ROUTE)
def check_health():
    if READY:
        return 200
    return 500


@app.post(AIP_PREDICT_ROUTE)
async def predict(request: Request):
    body = await request.json()
    prompts = body["instances"]
    pixray.reset_settings()
    pixray.add_settings(prompts=prompts)
    settings = pixray.apply_settings()
    pixray.do_init(settings)
    run_complete = False
    while run_complete == False:
        run_complete = pixray.do_run(settings, return_display=True)
        output_file = os.path.join(settings.outdir, settings.output)
        temp_copy = create_temporary_copy(output_file)

    ret_val = {"predictions": Path(os.path.realpath(temp_copy))}
    logger.debug(ret_val, file=sys.stderr)
    return ret_val
