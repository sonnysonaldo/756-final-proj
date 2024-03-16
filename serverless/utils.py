# Utils for preprocessing data etc 
import torch
import torchvision
from PIL import Image
from io import BytesIO
import torchvision.transforms as transforms
import googleapiclient.discovery
from google.api_core.client_options import ClientOptions


def predict_json(project, region, model, instances, version=None):
    """Send json data to a deployed model for prediction.

    Args:
        project (str): project where the Cloud ML Engine Model is deployed.
        model (str): model name.
        instances ([Mapping[str: Any]]): Keys should be the names of Tensors
            your deployed model expects as inputs. Values should be datatypes
            convertible to Tensors, or (potentially nested) lists of datatypes
            convertible to Tensors.
        version (str): version of the model to target.
    Returns:
        Mapping[str: any]: dictionary of prediction results defined by the 
            model.
    """
    # Create the ML Engine service object
    prefix = "{}-ml".format(region) if region else "ml"
    api_endpoint = "https://{}.googleapis.com".format(prefix)
    client_options = ClientOptions(api_endpoint=api_endpoint)

    # Setup model path
    model_path = "projects/{}/models/{}".format(project, model)
    if version is not None:
        model_path += "/versions/{}".format(version)

    # Create ML engine resource endpoint and input data
    ml_resource = googleapiclient.discovery.build(
        "ml", "v1", cache_discovery=False, client_options=client_options).projects()
    instances_list = instances.numpy().tolist() # turn input into list (ML Engine wants JSON)
    
    input_data_json = {"signature_name": "serving_default",
                       "instances": instances_list} 
    
    import json
    with open('request.json', 'w') as f:
        json.dump(input_data_json, f)
    
    

    request = ml_resource.predict(name=model_path, body=input_data_json)
    response = request.execute()
    
    # # ALT: Create model api
    # model_api = api_endpoint + model_path + ":predict"
    # headers = {"Authorization": "Bearer " + token}
    # response = requests.post(model_api, json=input_data_json, headers=headers)

    if "error" in response:
        raise RuntimeError(response["error"])
    return response["predictions"]

# Create a function to import an image and resize it to be able to be used with our model
def load_and_prep_image(filename, img_shape=224, rescale=False):
    """
    Reads in an image from filename, turns it into a tensor and reshapes into
    (224, 224, 3).
    """
    img = Image.open(BytesIO(filename))
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(size=img_shape)
            ])
    img = transform(img)[:3].unsqueeze(0)
    # img = img.flatten()
    
    if rescale:
        return img/255.
    else:
        return img

def update_logger(image, model_used, pred_class, pred_conf, correct=False, user_label=None):
    """
    Function for tracking feedback given in app, updates and reutrns 
    logger dictionary.
    """
    logger = {
        "image": image,
        "model_used": model_used,
        "pred_class": pred_class,
        "pred_conf": pred_conf,
        "correct": correct,
        "user_label": user_label
    }   
    return logger
