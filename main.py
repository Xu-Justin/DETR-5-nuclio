import json
import base64
from PIL import Image
import io

from model import get_model, get_result

def init_context(context):
    context.logger.info("Init context...  0%")

    context.user_data.model = get_model()
    context.logger.info(f"model device cuda: {next(context.user_data.model.parameters()).is_cuda}")

    context.logger.info("Init context...100%")

def handler(context, event):
    context.logger.info("Run custom model")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    image = Image.open(buf)
    
    result = get_result(context.user_data.model, image)

    return context.Response(body=json.dumps(result), headers={},
        content_type='application/json', status_code=200)