from azureml.core import Workspace

ws = Workspace.create(name='lab4fia',
                      subscription_id='37266c87-2b24-4039-b276-8d3002959431',
                      resource_group='lab4fia',
                      create_resource_group=True,
                      location='northeurope'
                      )

# "name": "lab4fia",
# "id": "/subscriptions/37266c87-2b24-4039-b276-8d3002959431/resourcegroups/lab4fia/providers/Microsoft.DesktopVirtualization/workspaces/lab4fia",
# "type": "Microsoft.DesktopVirtualization/workspaces",
# "location": "northeurope",
# "tags": {},
# "kind": null,
# "properties": {
#     "description": "lab4fiaDescription",
#     "friendlyName": "lab4fia",
#     "applicationGroupReferences": [],
#     "cloudPcResource": false,
#     "objectId": "16bf4a31-829f-47f0-af49-14d293a402e8"