import logging
import os
import sys
from pathlib import Path
import tempfile, shutil
import joblib
from fastapi import Request, FastAPI
from fastapi.responses import FileResponse

import pixray
import uvicorn

READY = False
app = FastAPI()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler("spam.log")
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

AIP_HEALTH_ROUTE = os.environ.get("AIP_HEALTH_ROUTE", "/health_check")
AIP_PREDICT_ROUTE = os.environ.get("AIP_PREDICT_ROUTE", "/predict")
MODEL_ARTIFACTS = os.environ.get("AIP_STORAGE_URI", ".")


def create_temporary_copy(src_path):
    _, tf_suffix = os.path.splitext(src_path)
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"tempfile{tf_suffix}")
    shutil.copy2(src_path, temp_path)
    return temp_path


#
# logger.info(f"Downloading artifacts from:{MODEL_ARTIFACTS}")
# # local_files = download_folder(MODEL_ARTIFACTS)
# # logger.info(f"Downloading artifacts done. Downloaded files: {local_files}")
#
# # model = load_joblib(local_files['model.joblib'])
# logger.info("model loaded")
#
# # user_item_sparse = load_rec_matrix(local_files['user_item_csv.csv'])
# logger.info("loaded ALS matrix")
def run(prompt,**kwargs):
    pixray.reset_settings()
    pixray.add_settings(prompts=prompt, quality="draft", custom_loss="aesthetic",**kwargs)
    settings = pixray.apply_settings()
    pixray.do_init(settings)
    run_complete = False
    i = 0
    while not run_complete:
        run_complete = pixray.do_run(settings, return_display=True)
        output_file = os.path.join(settings.outdir, settings.output)
        temp_copy = create_temporary_copy(output_file)
        logger.info(f"output newer version of result {i}")
        i += 1
    ret_val =  Path(os.path.realpath(temp_copy))
    return ret_val


run("test",iterations=1)
READY = True


@app.get(AIP_HEALTH_ROUTE)
def check_health():
    if READY:
        return 200
    return 500


@app.post(AIP_PREDICT_ROUTE)
async def predict(request: Request):
    body = await request.json()
    prompts = body["prompt"]
    ret_val = run(prompts)
    logger.debug(ret_val)
    return FileResponse(ret_val, media_type="image/png")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
