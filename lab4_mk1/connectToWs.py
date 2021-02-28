from azureml.core import Workspace

def getWorkspace():
    return Workspace.from_config(path="./wsConf.json")