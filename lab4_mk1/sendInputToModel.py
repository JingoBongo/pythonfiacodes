import json
import requests

input_array = [[ 52.0, 1467.0, 190.0, 496.0, 177.0 ]]
scoring_uri_first_not_working = "http://7f734165-cdae-42aa-b34a-e5b9f5f95e05.northeurope.azurecontainer.io/score"
scoring_uri = "http://2c568d7a-3ac6-4103-b90a-6469becc7f90.northeurope.azurecontainer.io/score"

# Add the 'data' field
data = { "data" : input_array,
        "method" : "predict"} # Write it in the required format for the REST API

input_data = json.dumps(data) # Convert to JSON string

# Set the content type to JSON
headers = {"Content-Type": "application/json"}

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)

# Return the model output
result = json.loads(resp.text)
print(result)
# 'result' will contain the dictionary: {'predict': 1}