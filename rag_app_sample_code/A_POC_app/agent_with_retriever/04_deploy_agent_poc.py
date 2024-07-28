# Databricks notebook source
# MAGIC %pip install --upgrade -qqqq databricks-agents openai databricks-vectorsearch 
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import time

import mlflow
from databricks import agents
from mlflow.models.signature import ModelSignature
from mlflow.models.resources import DatabricksVectorSearchIndex, DatabricksServingEndpoint
from mlflow.models.rag_signatures import StringResponse, ChatCompletionRequest


from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointStateReady, EndpointStateConfigUpdate
from databricks.sdk.errors import NotFound, ResourceDoesNotExist

# COMMAND ----------

# MAGIC %run ./00_global_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save secret for using OpenAI client if it doesn't exist already
# MAGIC
# MAGIC TODO: Remove the need for this by auto-provisioning credentials for OpenAI SDK

# COMMAND ----------

# DBTITLE 1,Save Secret if it Doesn't exist
w = WorkspaceClient()

# Where to save the secret
SCOPE_NAME = "ep"
SECRET_NAME = "llm_chain_pat_token"

# PAT token
SECRET_TO_SAVE = ""

existing_scopes = [scope.name for scope in w.secrets.list_scopes()]
if SCOPE_NAME not in existing_scopes:
    print(f"Creating secret scope `{SCOPE_NAME}`")
    w.secrets.create_scope(scope=SCOPE_NAME)
else:
    print(f"Secret scope `{SCOPE_NAME}` exists")

existing_secrets = [secret.key for secret in w.secrets.list_secrets(scope=SCOPE_NAME)]
if SECRET_NAME not in existing_secrets:
    print(f"Saving secret to `{SCOPE_NAME}.{SECRET_NAME}`")
    w.secrets.put_secret(scope=SCOPE_NAME, key=SECRET_NAME, string_value=SECRET_TO_SAVE)
else:
    print(f"Secret named `{SCOPE_NAME}.{SECRET_NAME}` already exists")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure the agent's environment variables to make OpenAI SDK work

# COMMAND ----------

# TODO: Remove the need for this

os.environ["DATABRICKS_HOST"] = 'https://' + mlflow.utils.databricks_utils.get_workspace_url()
os.environ["DATABRICKS_TOKEN"] = dbutils.secrets.get(scope=SCOPE_NAME, key=SECRET_NAME)

print(os.environ["DATABRICKS_HOST"])

# COMMAND ----------

# Specify the full path to the chain notebook & config YAML
model_file = "03_agent_model"
model_path = os.path.join(os.getcwd(), model_file)

config_file = "agent_config.yaml"
config_path = os.path.join(os.getcwd(), config_file)

agent_resources = "agent_resources.yaml"
agent_resources_path = os.path.join(os.getcwd(), config_file)

print(f"Model path: {model_path}")
print(f"Config path: {config_path}")
print(f"Resources path: {agent_resources_path}")

# COMMAND ----------

import yaml

# Load YAML file into a Python dictionary
with open(config_path, 'r') as file:
    agent_config = yaml.safe_load(file)

display(agent_config)

# COMMAND ----------

import pkg_resources

def get_package_version(package_name):
    try:
        package_version = pkg_resources.get_distribution(package_name).version
        return package_version
    except pkg_resources.DistributionNotFound:
        return f"{package_name} is not installed"

# COMMAND ----------


# databricks_resources = [
#     DatabricksServingEndpoint(endpoint_name=agent_config.get("llm_endpoint")),
#     DatabricksServingEndpoint(endpoint_name=agent_config.get("llm_endpoint")),
#     DatabricksVectorSearchIndex(index_name=agent_config.get("search_note_tool").get("retriever_config").get("vector_search_index"))
# ]

with mlflow.start_run(run_name=POC_CHAIN_RUN_NAME):
    model_info = mlflow.pyfunc.log_model(
        python_model=model_path,
        model_config=agent_config,
        artifact_path="chain",
        input_example=agent_config["input_example"],
        resources="agent_resources.yaml",
        example_no_conversion=True,  # Required by MLflow to use the input_example as the chain's schema
        signature=ModelSignature(
            inputs=ChatCompletionRequest(),
            outputs=StringResponse(),
        ),
        # specify all python packages that are required by your Agent
        pip_requirements=[
            "openai==" + get_package_version("openai"),
            "databricks-agents==" + get_package_version("databricks-agents"),
            "databricks-vectorsearch=="+get_package_version("databricks-vectorsearch"),
            # "databricks-sql-connector=="+get_package_version("databricks-sql-connector"),
        ],
    )

# COMMAND ----------

agent_config['input_example']

# COMMAND ----------

### Test the logged model
model = mlflow.pyfunc.load_model(model_info.model_uri)
model.predict(agent_config['input_example'])



# COMMAND ----------

# Use Unity Catalog as the model registry
mlflow.set_registry_uri('databricks-uc')

# Register the model to the Unity Catalog
uc_registered_model_info = mlflow.register_model(model_uri=model_info.model_uri, 
                                                 name=UC_MODEL_NAME)

# COMMAND ----------



# Deploy to enable the Review APP and create an API endpoint
deployment_info = agents.deploy(UC_MODEL_NAME, 
                                uc_registered_model_info.version,
                                environment_vars={"DATABRICKS_HOST" : 'https://' + mlflow.utils.databricks_utils.get_workspace_url(), 
                                                  "DATABRICKS_TOKEN": "{{secrets/"+SCOPE_NAME+"/"+SECRET_NAME+"}}"}
                                )

# Wait for the Review App to be ready
print("\nWaiting for endpoint to deploy.  This can take 15 - 20 minutes.", end="")
while w.serving_endpoints.get(deployment_info.endpoint_name).state.ready == EndpointStateReady.NOT_READY or w.serving_endpoints.get(deployment_info.endpoint_name).state.config_update == EndpointStateConfigUpdate.IN_PROGRESS:
    print(".", end="")
    time.sleep(30)

print(f"\n\nReview App: {deployment_info.review_app_url}")

# COMMAND ----------

# MAGIC %environment
# MAGIC "client": "1"
# MAGIC "base_environment": ""
