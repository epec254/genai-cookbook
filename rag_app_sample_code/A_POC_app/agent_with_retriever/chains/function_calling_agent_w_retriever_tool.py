# Databricks notebook source
# MAGIC %md
# MAGIC # Function Calling Agent w/ Retriever 
# MAGIC
# MAGIC In this notebook, we construct the Agent with a Retriever tool and vibe check it locally.  Here, we suggest iterating on the prompts while testing a small number (2 - 5) queries.

# COMMAND ----------

# MAGIC %pip install --upgrade -qqqq databricks-agents openai databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import json
import os
from typing import Any, Callable, Dict, List, Optional, Union
import mlflow
from dataclasses import asdict, dataclass
import pandas as pd
from mlflow.models import set_model, ModelConfig
from mlflow.models.rag_signatures import StringResponse, ChatCompletionRequest, Message
from openai import OpenAI

import os

# COMMAND ----------

# MAGIC %md
# MAGIC ## Important
# MAGIC
# MAGIC Before logging & deploying this Agent with `04_deploy_agent_poc` you need to change `debug` to `False`.  Why?  MLflow code logging takes a copy of this entire notebook and loads it to the model serving environment.  Putting your debug / vibe check code behind the `debug` flag ensure this code does not run in the model serving environment.

# COMMAND ----------

# Set to False for deployment
debug = False

# COMMAND ----------

# MAGIC %md
# MAGIC Use the Notebook's PAT token for development.  When this Agent is deployed, Agent Framework will securely provision credentials for the Agent to use.

# COMMAND ----------


if debug:
    # Use OpenAI client with Model Serving
    API_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    API_ROOT = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
    os.environ["DATABRICKS_TOKEN"] = API_TOKEN
    os.environ["DATABRICKS_HOST"] = f"{API_ROOT}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Agent config
# MAGIC
# MAGIC Adjust the Agent's prompts and configuration settings while vibe checking.

# COMMAND ----------

# MAGIC %run ./00_global_config

# COMMAND ----------

import yaml

if debug:
    agent_config = {
        "databricks_resources": {
            # Databricks Model Serving endpoint name
            # This is the generator LLM where your LLM queries are sent.
            # "llm_endpoint_name": LLM_ENDPOINT,
            "llm_endpoint_name": "ep-gpt4o",
        },
        "llm_config": {
            # Define a template for the LLM prompt.  This is how the RAG chain combines the user's question and the retrieved context.
            # "llm_system_prompt_template": """You are a helpful assistant that answers questions by calling tools.  Please provide responses based on the information from these tools.  If you cannot answer a query or don't have a relevant tool, respond with 'Sorry, I'm not trained to answer that question'.""",
            "llm_system_prompt_template": """You are a helpful assistant that answers questions by calling tools.  Please provide responses based on the information from these tools.  If you cannot answer a query or don't have a relevant tool, respond with 'Sorry, I'm not trained to answer that question'.""",
    # "llm_system_prompt_template": """You are a helpful assistant that can access external functions. The responses from these function calls will be appended to this dialogue. Please provide responses based on the information from these function calls.""",
            
            # "few_shot_examples": few_shot_examples,
            # Parameters that control how the LLM responds.
            "llm_parameters": {
                "temperature": 0.001,
                "max_tokens": 500,
            },
        },
        "retriever_tool": {
            "prompt_template": """Use the following pieces of retrieved context to answer the question.\nSome pieces of context may be irrelevant, in which case you should not use them to form the answer.\n\nContext: {context}""",
            "retriever_config": {
                "vector_search_index": VECTOR_INDEX_NAME,

                # These values come from the schema defined by 02_data_pipeline in the chunk table.  They are required to display the Retriever's results inside the Agent Evaluation Review App.
                "vector_search_schema": {
                    # The column name in the retriever's response referred to the unique key
                    # If using Databricks vector search with delta sync, this should the column of the delta table that acts as the primary key
                    "primary_key": "chunk_id",
                    # The column name in the retriever's response that contains the returned chunk.
                    "chunk_text": "content_chunked",
                    # The template of the chunk returned by the retriever - used to format the chunk for presentation to the LLM.
                    "document_uri": "doc_uri",
                },
                # Prompt template used to format the retrieved information to present to the LLM to help in answering the user's question
                "vector_search_threshold": 0.1,
                # The column name in the retriever's response that refers to the original document.
                "vector_search_parameters": {
                    # Number of search results that the retriever returns
                    "num_results": 5,
                    # Type of search to run - use ann unless you have a specific reason to switch to hybrid e.g., your retrieval quality is low because the embedding fails to understand business specific terms that are present in both your documents and user's queries.
                    # Semantic search: `ann`
                    # Hybrid search (keyword + sementic search): `hybrid`
                    "query_type": "ann",
                    # defines the columns that will be returned by the Retriever and shown to the LLM.  This must include the 3 columns in `vector_search_schema` and any additional columns you wish to filter on.
                    "columns": [
                        "chunk_id",
                        "content_chunked",
                        "doc_uri",
                    ],
                },
            },
        },
        "tools": [
            # Define the Retriever tool
            {
                "type": "function",
                "function": {
                    "name": "retrieve_documents",
                    "description": "Search for documents that are relevant to a user's query.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The query to find documents about.",
                            },
                            "doc_name_filter": {
                                "type": "string",
                                "enum": ["/Volumes/ericpeter_catalog/agents/source_docs/2212.14024.pdf", "test_doc_2"],
                                "description": "A filter for the specific document name.",
                            }
                        },
                        "required": ["query"],
                    },
                },
            },
            
        ],
        "input_example": {
            "messages": [
                {
                    "role": "user",
                    "content": "What is RAG?",
                },
            ]
        },
    }

    # Write the configuration to a YAML so that the 04_deploy_agent_poc notebook can use it to deploy the Agent
    file_path = 'agent_config.yaml'
    with open(file_path, 'w') as file:
        yaml.dump(agent_config, file)
else:
    # when logging the Agent, load the config from the YAML
    # this is required since the %run command isn't executable when logging the model, and the parameters in the Agent's config come from this Notebook
    try:
        with open('agent_config.yaml', 'r') as file:
            agent_config = yaml.safe_load(file)
    except:
        # TODO: remove the need to catch this - it crashes model serving currently
        print("can't load agent config")



# COMMAND ----------

# MAGIC %md ## Retriever tool

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient


@dataclass
class Document:
    page_content: str
    metadata: Dict[str, str]
    type: str


class VectorSearchRetriever:
    """
    Class using Databricks Vector Search to retrieve relevant documents.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vector_search_client = VectorSearchClient(disable_notice=True)
        self.vector_search_index = self.vector_search_client.get_index(
            index_name=self.config.get("vector_search_index")
        )
        vector_search_schema = self.config.get("vector_search_schema")
        

    @mlflow.trace(span_type="RETRIEVER")
    def similarity_search(self, query: str, filters: str = None) -> List[Document]:
        """
        Performs vector search to retrieve relevant chunks.

        Args:
            query: Search query.
            filter_date: Date to filter the results by publication date. Format: "YYYY-MM-DD".

        Returns:
            List of retrieved Documents.
        """
   
        # filters = None
        # if filter_date:
        #     # Convert filter_date to unix timestamp
        #     filter_timestamp = eint(datetime.strptime(filter_date, "%Y-%m-%d").timestamp())
            
        #     # Filter results to retrieve only documents with a publication date greater than the specified date
        #     filters = {self.config.get("vector_search_schema").get("publication_date_unix_timestamp") + " >": filter_timestamp}
        print(filters)
        # filters={"doc_uri": [ "/Volumes/ericpeter_catalog/agents/source_docs/2212.14024.pdf"]}

        traced_search = mlflow.trace(
            self.vector_search_index.similarity_search,
            name="vector_search.similarity_search",
            # span_type="RETRIEVER",
        )

        results = traced_search(
            query_text=query,
            # filters=filters,
            **self.config.get("vector_search_parameters"),
        )

        vector_search_threshold = self.config.get("vector_search_threshold")
        documents = self.convert_vector_search_to_documents(results, vector_search_threshold)
      
        return [asdict(doc) for doc in documents]
    
    @mlflow.trace(span_type="PARSER")
    def convert_vector_search_to_documents(self, vs_results, vector_search_threshold) -> List[Document]:
        column_names = []
        for column in vs_results["manifest"]["columns"]:
            column_names.append(column)

        
        docs = []
        if vs_results["result"]["row_count"] >0:
            for item in vs_results["result"]["data_array"]:
                metadata = {}
                score = item[-1]
                if score >= vector_search_threshold:
                    metadata['similarity_score'] = score
                    # print(score)
                    i = 0
                    for field in item[0:-1]:
                        # print(field + "--")
                        metadata[column_names[i]["name"]] = field
                        i = i + 1
                    # put contents of the chunk into page_content
                    page_content = metadata[self.config.get("vector_search_schema").get("chunk_text")]
                    del metadata[self.config.get("vector_search_schema").get("chunk_text")]
                    
                    doc = Document(page_content=page_content, metadata=metadata, type="Document")  # , 9)
                    docs.append(doc)
        return docs


# COMMAND ----------

# MAGIC %md
# MAGIC ## Chain

# COMMAND ----------

class AgentWithRetriever(mlflow.pyfunc.PythonModel):
    """
    Class representing an Agent that includes a Retriever tool
    """

    def load_context(self, context):
        self.config = mlflow.models.ModelConfig(development_config=agent_config)
        self.databricks_resources = self.config.get("databricks_resources")

        # OpenAI client used to query Databricks Chat Completion endpoint
        self.model_serving_client = OpenAI(
            api_key=os.environ.get("DATABRICKS_TOKEN"),
            base_url=str(os.environ.get("DATABRICKS_HOST")) + "/serving-endpoints",
        )

        # Init the retriever for `search_customer_notes_for_topic` tool
        self.customer_notes_retriever = VectorSearchRetriever(
            self.config.get("retriever_tool").get("retriever_config")
        )

        # Configure the Review App
        vector_search_schema = (
            self.config.get("retriever_tool")
            .get("retriever_config")
            .get("vector_search_schema")
        )
        mlflow.models.set_retriever_schema(
            primary_key=vector_search_schema.get("primary_key"),
            text_column=vector_search_schema.get("chunk_text"),
            doc_uri=vector_search_schema.get("doc_uri"),
        )

        # # Init SQL connection
        # warehouse_id = (
        #     self.config.get("get_notes_for_customer")
        #     .get("databricks_resources")
        #     .get("sql_warehouse_id")
        # )

        # self.customer_notes_sql_warehouse = DatabrickSQL(warehouse_id=warehouse_id)

        self.tool_funcs = {
            "retrieve_documents": self.retrieve_documents,
            # "get_notes_for_customer": self.tool_get_notes_for_customer,
            # "search_customer_notes_for_topic": self.tool_search_customer_notes_for_topic,
        }
        # self.messages = None

        self.chat_history = None

    @mlflow.trace(name="chain", span_type="CHAIN")
    def predict(
        self,
        context: Any = None,
        model_input: Union[ChatCompletionRequest, Dict, pd.DataFrame] = None,
        params: Any = None,
    ) -> StringResponse:
        ##############################################################################
        # Extract `messages` key from the `model_input`
        messages = self.get_messages_array(model_input)

        ##############################################################################
        # Parse `messages` array into the user's query & the chat history
        with mlflow.start_span(name="parse_input", span_type="PARSER") as span:
            span.set_inputs({"messages": messages})
            user_query = self.extract_user_query_string(messages)
            # save in the object since we need this within the retriever tool for query re-writing
            self.chat_history = self.extract_chat_history(messages)
            span.set_outputs(
                {"user_query": user_query, "chat_history": self.chat_history}
            )

        ##############################################################################
        # Generate Answer
        system_prompt = self.config.get("llm_config").get("llm_system_prompt_template")

        # Add the previous history
        # TODO: Need a way to include the previous tool calls
        messages = (
            [{"role": "system", "content": system_prompt}]
            + self.chat_history  # append chat history for multi turn
            + [{"role": "user", "content": user_query}]
        )

        # Call the LLM to recursively calls tools and eventually deliver a generation to send back to the user
        (
            model_response,
            messages_log_with_tool_calls,
        ) = self.recursively_call_and_run_tools(messages=messages)

        # If your front end keeps of converastion history and automatically appends the bot's response to the messages history, remove this line.
        messages_log_with_tool_calls.append(model_response.choices[0].message.to_dict())

        # remove the system prompt - this should not be exposed to the Agent caller
        messages_log_with_tool_calls = messages_log_with_tool_calls[1:]

        return {
            "content": model_response.choices[0].message.content,
            # TODO: this should be returned back to the Review App (or any other front end app) and stored there so it can be passed back to this stateless agent with the next turns of converastion.
            "messages": messages_log_with_tool_calls,
        }

    @mlflow.trace(span_type="TOOL")
    def retrieve_documents(self, query, doc_name_filter) -> Dict:
        # Rewrite the query e.g., "what is it?" to "what is [topic from previous question]".  Test the chain with and without this - some function calling models automatically handle the query rewriting e.g., when they call the tool, they rewrite the query
        vs_query = query
        # if len(self.chat_history) > 0:
        #     vs_query = self.query_rewrite(query, self.chat_history)
        # else:
        #     vs_query = query

        print(doc_name_filter)
        results = self.customer_notes_retriever.similarity_search(vs_query)

        context = ""
        for result in results:
            context += "Document: " + json.dumps(result) + "\n"

        resulting_prompt = (
            self.config.get("retriever_tool")
            .get("prompt_template")
            .format(context=context)
        )

        return resulting_prompt  # json.dumps(results, default=str)

    @mlflow.trace(span_type="PARSER")
    def query_rewrite(self, query, chat_history) -> str:
        ############
        # Prompt Template for query rewriting to allow converastion history to work - this will translate a query such as "how does it work?" after a question such as "what is spark?" to "how does spark work?".
        ############
        query_rewrite_template = """Based on the chat history below, we want you to generate a query for an external data source to retrieve relevant documents so that we can better answer the question. The query should be in natural language. The external data source uses similarity search to search for relevant documents in a vector space. So the query should be similar to the relevant documents semantically. Answer with only the query. Do not add explanation.

        Chat history: {chat_history}

        Question: {question}"""

        chat_history_formatted = self.format_chat_history(chat_history)

        prompt = query_rewrite_template.format(
            question=query, chat_history=chat_history_formatted
        )

        model_response = self.chat_completion(
            messages=[{"role": "user", "content": prompt}]
        )
        if len(model_response.choices) > 0:
            return model_response.choices[0].message.content
        else:
            # if no generation, return the original query
            return query

    @mlflow.trace(span_type="AGENT")
    def recursively_call_and_run_tools(self, max_iter=10, **kwargs):
        tools = self.config.get("tools")
        messages = kwargs["messages"]
        del kwargs["messages"]
        i = 0
        while i < max_iter:
            response = self.chat_completion(messages=messages, tools=True)
            # response = client.chat.completions.create(tools=tools, messages=messages, **kwargs)
            assistant_message = response.choices[0].message
            tool_calls = assistant_message.tool_calls
            if tool_calls is None:
                # the tool execution finished, and we have a generation
                # print(response)
                return (response, messages)
            tool_messages = []
            for tool_call in tool_calls:  # TODO: should run in parallel
                function = tool_call.function
                # uc_func_name = decode_function_name(function.name)
                args = json.loads(function.arguments)
                # result = exec_uc_func(uc_func_name, **args)
                result = self.execute_function(function.name, args)
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
                tool_messages.append(tool_message)
            assistant_message_dict = assistant_message.dict()
            del assistant_message_dict["content"]
            del assistant_message_dict["function_call"]
            messages = (
                messages
                + [
                    assistant_message_dict,
                ]
                + tool_messages
            )
            # print("---current state of messages---")
            # print(messages)
        raise "ERROR: max iter reached"

    @mlflow.trace(span_type="FUNCTION")
    def execute_function(self, function_name, args):
        the_function = self.tool_funcs.get(function_name)
        # print(the_function)
        result = the_function(**args)
        return result

    def chat_completion(self, messages: List[Dict[str, str]], tools: bool = False):
        endpoint_name = self.databricks_resources["llm_endpoint_name"]
        llm_options = self.config.get("llm_config")["llm_parameters"]

        # Trace the call to Model Serving
        traced_create = mlflow.trace(
            self.model_serving_client.chat.completions.create,
            name="chat_completions_api",
            span_type="CHAT_MODEL",
        )

        if tools:
            return traced_create(
                model=endpoint_name,
                messages=messages,
                tools=self.config.get("tools"),
                **llm_options,
            )
        else:
            return traced_create(model=endpoint_name, messages=messages, **llm_options)

    @mlflow.trace(span_type="PARSER")
    def get_messages_array(
        self, model_input: Union[ChatCompletionRequest, Dict, pd.DataFrame]
    ) -> List[Dict[str, str]]:
        # TODO: This is required to handle both Dict + ChatCompletionRequest wrapped inputs.  If ChatCompletionRequest  supported .get(), this code wouldn't be required.

        if type(model_input) == ChatCompletionRequest:
            return model_input.messages
        elif type(model_input) == dict:
            return model_input.get("messages")
        ## required to test with the following code after logging
        ## model = mlflow.pyfunc.load_model(model_info.model_uri)
        ## model.predict(agent_config['input_example'])
        elif type(model_input) == pd.DataFrame:
            return model_input.iloc[0].to_dict().get("messages")

    @mlflow.trace(span_type="PARSER")
    def extract_user_query_string(
        self, chat_messages_array: List[Dict[str, str]]
    ) -> str:
        """
        Extracts user query string from the chat messages array.

        Args:
            chat_messages_array: Array of chat messages.

        Returns:
            User query string.
        """

        if isinstance(chat_messages_array, pd.Series):
            chat_messages_array = chat_messages_array.tolist()

        if isinstance(chat_messages_array[-1], dict):
            return chat_messages_array[-1]["content"]
        elif isinstance(chat_messages_array[-1], Message):
            return chat_messages_array[-1].content
        else:
            return chat_messages_array[-1]

    @mlflow.trace(span_type="PARSER")
    def extract_chat_history(
        self, chat_messages_array: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Extracts the chat history from the chat messages array.

        Args:
            chat_messages_array: Array of chat messages.

        Returns:
            The chat history.
        """

        if isinstance(chat_messages_array, pd.Series):
            chat_messages_array = chat_messages_array.tolist()

        if isinstance(chat_messages_array[0], dict):
            return chat_messages_array[:-1]  # return all messages except the last one
        elif isinstance(chat_messages_array[0], Message):
            new_array = []
            for message in chat_messages_array[:-1]:
                new_array.append(asdict(message))
            return new_array
        else:
            # if the messages are not in dict format, convert them to the expected format
            return [
                {"role": "user" if i % 2 == 0 else "assistant", "content": msg}
                for i, msg in enumerate(chat_messages_array[:-1])
            ]

    @mlflow.trace(span_type="PARSER")
    def format_chat_history(self, chat_history: List[Dict[str, str]]) -> str:
        """
        Formats the chat history into a string.

        Args:
            chat_history: List of chat messages.

        Returns:
            Formatted chat history string.
        """
        if not chat_history:
            return ""

        formatted_history = []
        for message in chat_history:
            if message["role"] == "user":
                formatted_history.append(f"User: {message['content']}")

            # this logic ignores assistant messages that are just about tool calling and have no user facing content
            elif message["role"] == "assistant" and message.get("content"):
                formatted_history.append(f"Assistant: {message['content']}")

        return "\n".join(formatted_history)


set_model(AgentWithRetriever())


if debug:
    chain = AgentWithRetriever()
    # 1st turn of converastion
    first_turn_input = {
        "messages": [
            {"role": "user", "content": f"what does the test_doc_a say?"},
        ]
    }
    chain.load_context(None)
    response = chain.predict(model_input=first_turn_input)
    print(response["content"])

    print()
    print("------")
    print()

    # 2nd turn of converastion
    new_messages = response["messages"]
    new_messages.append({"role": "user", "content": f"who invented it?"})
    # print(type(new_messages))
    second_turn_input = {"messages": new_messages}
    response = chain.predict(model_input=second_turn_input)
    print(response["content"])

# COMMAND ----------

# MAGIC %environment
# MAGIC "client": "1"
# MAGIC "base_environment": ""
