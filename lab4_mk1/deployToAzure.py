from azureml.core import Workspace
from azureml.core.model import Model

from lab4_mk1 import connectToWs

model_name = "poly-sorted-regression-2"
endpoint_name = "poly-sorted-regression-2-ep"

ws = connectToWs.getWorkspace()

# Locate the model in the workspace
model = Model(ws, name=model_name)

# Deploy the model as a real-time endpoint
service = Model.deploy(ws, endpoint_name, [model])

# Wait for the model deployment to complete
service.wait_for_deployment(show_output=True)