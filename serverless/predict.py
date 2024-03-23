import base64
from io import BytesIO
from PIL import Image

from google.cloud import aiplatform
from google.cloud.aiplatform import Endpoint
from google.cloud.aiplatform.gapic.schema import predict


def predict_endpoint(
    project: str,
    endpoint_id: str,
    img_bytes: str,
    project_number: str,
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
    endpoint = Endpoint(
        endpoint_name=f"projects/{project_number}/locations/us-central1/endpoints/{endpoint_id}",
        project=project,
        location=location
    )
    data = {"data": base64.b64encode(img_bytes).decode("utf-8")}
    response = endpoint.predict(instances=[data])
    prediction = response.predictions[0]
    prediction = dict(sorted(prediction.items(), key=lambda item: item[1], reverse=True))
    return prediction