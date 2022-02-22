import base64
import logging
import os
from pathlib import Path
import tempfile, shutil

from fastapi import Request, FastAPI


import uvicorn

READY = False
app = FastAPI()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

AIP_HEALTH_ROUTE = os.environ.get("AIP_HEALTH_ROUTE", "/health_check")
AIP_PREDICT_ROUTE = os.environ.get("AIP_PREDICT_ROUTE", "/predict")
MODEL_ARTIFACTS = os.environ.get("AIP_STORAGE_URI", ".")


def create_temporary_copy(src_path):
    _, tf_suffix = os.path.splitext(src_path)
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"tempfile{tf_suffix}")
    shutil.copy2(src_path, temp_path)
    return temp_path


def run(prompt, **kwargs):
    import pixray

    pixray.reset_settings()
    pixray.add_settings(prompts=prompt, quality="draft", custom_loss="aesthetic", **kwargs)
    settings = pixray.apply_settings()
    pixray.do_init(settings)
    run_complete = False
    i = 0
    while not run_complete:
        run_complete = pixray.do_run(settings, return_display=True)
        output_file = os.path.join(settings.outdir, settings.output)
        temp_copy = create_temporary_copy(output_file)
        logger.info(f"iterating result for `{prompt}` output num: {i}")
        i += 1
    ret_val = Path(os.path.realpath(temp_copy))
    return ret_val


logger.info("MODELS LOADED - RUNNING TEST RUN")
# run("test", iterations=1)
READY = True
print(f"TEST RUN FINSHED, READY {READY}")


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
    print(f"Model Called with instance: `{instance}`")
    output_path = run(**instance)
    with open(output_path, "rb") as image_data:
        response = {
            "predictions": [{"image_bytes": {"b64": base64.b64encode(image_data.read()).decode()}}]
        }
    return response


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
