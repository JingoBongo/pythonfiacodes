import json

import azureml.core.run
import connectToWs
from azureml.core import Model

run = azureml.core.Run.register_model( model_name='polySortedRegression',
                                       tags={'ver': 'mk1'},
                                       model_path='../lab4_mk1/PolynomialSortedRegression.py')

json_file = open("wsConf.json")
variables = json.load(json_file)
json_file.close()


# model = Model(connectToWs.getWorkspace(), id=variables["subscription_id"])
model = run.model_framework
print(model.name, model.id, model.version, sep='\t')

# ws = Workspace.from_config()
#
# model = Model.register(workspace = ws,
#                        model_path ="breastcancerpredictor.joblib",
#                        model_name = "breast_cancer_predictor",
#                        model_framework=Model.Framework.SCIKITLEARN,
#                        model_framework_version='0.22.2',
#                        description = "Binary predictor for breast cancer on cell nuclei, \
#                                     trained on the Wisconsin Breast Cancer Database")