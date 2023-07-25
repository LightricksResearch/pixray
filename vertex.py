import base64
import logging
import multiprocessing
import os
import time
from pathlib import Path
import tempfile, shutil

from fastapi import Request, FastAPI

from google.cloud import storage

import uvicorn

READY = False
TIME_OUT = 20 * 60
app = FastAPI()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

AIP_HEALTH_ROUTE = os.environ.get("AIP_HEALTH_ROUTE", "/health_check")
AIP_PREDICT_ROUTE = os.environ.get("AIP_PREDICT_ROUTE", "/predict")
MODEL_ARTIFACTS = os.environ.get("AIP_STORAGE_URI", "gs://ltx_text_us")
EXPORT_BUCKET = os.environ.get("EXPORT_BUCKET", "ltx_text_us")


def save_to_bucket(inputpath, output_path):
    if len(EXPORT_BUCKET) > 0:
        client = storage.Client()
        bucket = client.get_bucket(EXPORT_BUCKET)
        blob = bucket.blob(
            "outputs/" + output_path
        )  # This defines the path where the file will be stored in the bucket
        _ = blob.upload_from_filename(filename=inputpath)
        print("File {} uploaded to {}.".format(inputpath, output_path))


def create_temporary_copy(src_path):
    _, tf_suffix = os.path.splitext(src_path)
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"tempfile{tf_suffix}")
    shutil.copy2(src_path, temp_path)
    return temp_path


def run(kwargs):
    import pixray

    print(f"Model Called with args: `{kwargs}`")
    pixray.reset_settings()
    pixray.add_settings(**kwargs)
    settings = pixray.apply_settings()
    pixray.do_init(settings)
    run_complete = False
    i = 0
    prompts = kwargs.get("prompts", "")
    while not run_complete:
        run_complete = pixray.do_run(settings, return_display=True)
        output_file = os.path.join(settings.outdir, settings.output)
        time_string = time.strftime("%Y%m%d-%H%M%S")
        temp_copy = create_temporary_copy(output_file)
        logger.info(f"iterating result for `{prompts}` output num: {i}")
        i += 1
    ret_val = Path(os.path.realpath(temp_copy))
    save_to_bucket(inputpath=ret_val, output_path=prompts + "_¬" + time_string + ".png")
    return ret_val


logger.info("MODELS LOADED - RUNNING TEST RUN")
run_args = {"instances": {"prompts": "the eiffel tower", "iterations": 1}}
p = multiprocessing.Process(target=run, kwargs=run_args)
print("start - preload function")
p.start()

# Wait for X seconds or until process finishes
p.join(timeout=TIME_OUT)
# If thread is still active
if p.is_alive():
    print("Timeout - kill preload function")
    p.terminate()
    p.kill()
    p.join()

READY = True
print(f"TEST RUN FINISHED, READY = {READY}")


@app.get(AIP_HEALTH_ROUTE)
def check_health():
    if READY:
        return 200
    return 500


@app.post(AIP_PREDICT_ROUTE)
async def predict(request: Request):
    body = await request.json()
    instance = body["instances"]
    if isinstance(instance, list):
        instance = instance[0]
    logger.info(f"instances {instance}")

    output_path = run(instance)
    with open(output_path, "rb") as image_data:
        response = {
            "predictions": [{"image_bytes": {"b64": base64.b64encode(image_data.read()).decode()}}]
        }
    return response


if __name__ == "__main__":

    host = os.environ.get("VERTEX_HOST", "0.0.0.0")
    port = int(os.environ.get("VERTEX_PORT", 8080))
    workers = int(os.environ.get("WORKERS", 2))
    print(f"starting uvicorn with {host} and {port} workers={workers}")
    uvicorn.run("vertex:app", host=host, port=port, workers=workers)
