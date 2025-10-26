import sys
sys.dont_write_bytecode =True

import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chainlit as cl
import json
from enum import Enum
from typing import List,Type
from pydantic import BaseModel,Field
from core.helper import print_colored
import asyncio
import base64
import plotly.graph_objects as go

# -------------------------------- Structured Agent -------------------------------------

import json
from enum import Enum
from typing import List,Type
from core.helper import print_colored
from pydantic import BaseModel,Field

class ChainlitStructuredAgent:

    def __init__(self,model,agent_name,agent_description,agent_instructions,tools=[],assistant_agents=[],max_allowed_attempts=10,vission_model=None,vission_model_prompt=None) -> None:
        self.model = model
        self.vission_model = vission_model
        self.vission_model_prompt = vission_model_prompt
        self.agent_name = agent_name
        self.agent_description = agent_description
        self.agent_instructions=agent_instructions
        self.tools = tools
        self.assistant_agents = assistant_agents
        self.tool_names = []
        self.max_allowed_attempts= max_allowed_attempts
        self.attempts_made = 0
        self.messages = []
            
        if len(self.assistant_agents):

            self.prepare_prompt()
            self.agents_as_tools = {agent.agent_name:agent for agent in assistant_agents}
            self.assistants_names = []

        self.response_format = self.prepare_Default_tools()

        if len(self.tools):

            self.tool_objects = {i:j for i,j in zip(self.tool_names,tools)}

            tool_schemas = self.prepare_schema_from_tool(self.tools)
            self.agent_instructions+="""\n## Available Tools:\n"""
            self.agent_instructions+=f"""\nYou have access to the following tools:\n{tool_schemas}\nYou must use any one of these tools to answer the user question.\n\n"""
            self.agent_instructions+="""IMPORTANT!: You must provide your response in the below json format.
{
"thoughts":["Always you should think before taking any action"],
"tool_name":"Name of the tool",
"tool_args":{"arg_name":"arg_value"}
}
"""
        
    def prepare_Default_tools(self):

        # Prepare final answer tool
        class FinalAnswer(BaseModel):
            final_answer : str = Field(description="Your final response to the user")
            def run(self):
                return self.final_answer
    
        self.tools.append(FinalAnswer)

        # Prepare Assign Task tool
        if len(self.assistant_agents):

            self.assistants_names = [i.agent_name for i in self.assistant_agents]

            recipients = Enum("recipient", {name: name for name in self.assistants_names})

            assistant_description = f"Select the correct Agent to assign the task : {self.assistants_names}\n\n"

            for assistant in self.assistant_agents:

                assistant_description+=assistant.agent_name+" : "+assistant.agent_description+"\n"

            class AssignTask(BaseModel):

                """Use this tool to facilitate direct, synchronous communication between specialized agents within your agency. When you send a message using this tool, you receive a response exclusively from the designated recipient agent. To continue the dialogue, invoke this tool again with the desired recipient agent and your follow-up message. Remember, communication here is synchronous; the recipient agent won't perform any tasks post-response. You are responsible for relaying the recipient agent's responses back to the user, as the user does not have direct access to these replies. Keep engaging with the tool for continuous interaction until the task is fully resolved. Do not send more than 1 message at a time."""

                my_primary_instructions: str = Field(...,
                                                    description="Please repeat your primary instructions step-by-step, including both completed "
                                                                "and the following next steps that you need to perform. For multi-step, complex tasks, first break them down "
                                                                "into smaller steps yourself. Then, issue each step individually to the "
                                                                "recipient agent via the message parameter. Each identified step should be "
                                                                "sent in separate message. Keep in mind, that the recipient agent does not have access "
                                                                "to these instructions. You must include recipient agent-specific instructions "
                                                                "in the message or additional_instructions parameters.")
                recipient: recipients = Field(..., description=assistant_description,examples=self.assistants_names)

                task_details: str = Field(...,
                                    description="Specify the task required for the recipient agent to complete. Focus on "
                                                "clarifying what the task entails, rather than providing exact "
                                                "instructions.")

                additional_instructions: str = Field(description="Any additional instructions or clarifications that you would like to provide to the recipient agent.")

            self.tools.append(AssignTask)
                
        self.tool_names = [i.__name__ for i in self.tools]

    def prepare_schema_from_tool(self,Tools: List[Type[BaseModel]]) -> List[dict]:
        schemas = ""
        for tool in Tools:
            schema = tool.model_json_schema()
            schemas+="\n"
            schemas += f"""
"Tool Name": {tool.__name__},
"Tool Description": {tool.__doc__},
"Tool Parameters": 
    "Properties": {schema["properties"]},
    "Required": {schema["required"]},
    "Type": {schema["type"]}\n"""
            schemas+="\n"
            
        return schemas

    def prepare_prompt(self):

        if len(self.assistant_agents):

            self.agent_instructions+="\n**Task Assignment**: You can assign tasks to the following agents who are responsible to help you to achieve your goal.\n"

            self.agent_instructions+="-----------------------------------------------\n"

            for agent in self.assistant_agents:

                self.agent_instructions+="- **Agent Name**: "+agent.agent_name+"\n"
                self.agent_instructions+="- **Agent Description**:\n"+agent.agent_description+"\n"

            self.agent_instructions+="\n-----------------------------------------------\n"
                
    def prepare_messages(self,content,role=None,messages=[]):

        if not len(messages):

            messages = [
                {"role":"system","content":self.agent_instructions},
                {"role":"user","content":content}
            ]

        else:

            messages.append({"role":role,"content":content})

        return messages
    
    def construct_message_from_output(self,output):

        content = ""

        for key, value in output.items():

            if key=='task_checklist':

                content+=f"Task Check List:\n{"\n".join(value)}\n"

            elif key=='completed_tasks':

                content+=f"Completed Tasks:\n{"\n".join(value)}\n"

            elif key=='thoughts':

                content+=f"Thought: {"\n".join(value)}\n"

            elif key=='tool_name':

                if value=="AssignTask":

                    content+=f"I need to assign a task to the '{output['tool_args']['recipient']}' with the following instructions:"
                else:

                    content+=f"We need to use the '{value}' tool with the following arguments: "
            
            else:

                content+=f"{value}."

        return content
    @cl.step(type="tool")
    def execute_tool(self,messages,tool_details):

        try:

            # assistant_content=self.construct_message_from_output(tool_details)
            assistant_content=str(tool_details)

        except Exception as e:

            invalid_arg_error_message ="Error while executing tool. Please check the tool name or provide a valid arguments to the tool: "+str(e)

            tool_content = invalid_arg_error_message

            assistant_content = str(tool_details)

            messages.append({"role":"assistant","content":assistant_content})
            
            messages.append({"role":"user","content":tool_content})

            return messages

        if tool_details['tool_name'] in self.tool_names :

            if tool_details['tool_name'] == 'AssignTask':

                try:

                    arguments = tool_details['tool_args']

                    task_details =arguments.get('task_details',"")

                    additional_instructions =arguments.get('additional_instructions',"")

                    print_colored(f"{self.agent_name} assigned a task to {arguments['recipient']}","cyan")

                    assistant_agent = self.agents_as_tools[arguments['recipient']]

                    user_input = task_details + "\n" + additional_instructions

                    print_colored("Task Details: \n"+user_input,"white")

                    tool_content = assistant_agent.run(user_input)

                    tool_content = f"Response from the {arguments['recipient']} : "+tool_content
                    
                except Exception as e:

                    print_colored("Error Tool: "+str(e),"red")

                    tool_content = f"Error while assigning task to {arguments['recipient']}. Please provide a correct agent name: {[i.agent_name for i in self.assistant_agents]}"

            else:

                try:

                    print_colored(f"{self.agent_name} : Calling Tool {tool_details['tool_name']}","yellow")

                    tool_output = self.tool_objects[tool_details['tool_name']](**tool_details['tool_args']).run()

                    print_colored(f"Got output....{len(tool_output)}","orange")

                    tool_content=f"Output From {tool_details['tool_name']} Tool: {tool_output}"

                    if tool_details['tool_name'] == 'JupyterNotebookTool' and len(tool_output):

                        for i in range(len(tool_output)):

                            elements = []

                            # print('Keys: ', tool_output[i].keys())

                            # print('expected_output_type: ', tool_output[i].get('expected_output_type',""))

                            # print('display_type: ', tool_output[i].get('display_type',""))

                            if tool_output[i]['display_type']=='text':
                                
                                elements.append(cl.Text(name="Tool Output", content=tool_output[i]['final_output'], display="inline"))
                                
                                # asyncio.run(cl.Message("",elements=elements).send())

                            elif tool_output[i]['display_type']=='image':

                                image_data = tool_output[i]['output'].split('base64,')[1]

                                # Decode the base64 image data
                                image_bytes = base64.b64decode(image_data)

                                elements.append(cl.Image(name="Plot",size='medium',content=image_bytes,display="inline"))
                                
                                # asyncio.run(cl.Message("",elements=elements).send())

                                image_base64 = base64.b64encode(base64.b64decode(image_data)).decode('utf-8')

                                if self.vission_model:

                                    print_colored("Describing the Image...","green")

                                    plot_dec=asyncio.run(self.vission_model.get_output(self.vission_model_prompt,base64_image=image_base64))
                                    
                                    tool_output[i]['final_output'] +="\n"+plot_dec
                                    
                                    print_colored(f"Image Description.: {tool_output[i]['final_output']}","green")

                                    tool_output[i]['final_output'] =plot_dec

                                    elements.append(cl.Text(name="Insights",content=plot_dec))

                                    # asyncio.run(cl.Message("",elements=elements).send())

                            elif tool_output[i]['display_type']=='plotly':

                                print_colored("Trying to display plotly.....","red")

                                fig = go.Figure(data=tool_output[i]['output'])

                                print_colored(f"Created fig.....{type(fig)}","red")

                                # Convert the Plotly figure to a PNG image
                                img_bytes = fig.to_image(format="png")

                                print_colored(f"Created img_bytes.....{type(fig)}","red")

                                # Encode the image to base64
                                image_base64 = base64.b64encode(img_bytes).decode('utf-8')

                                print_colored(f"Created image_base64.....{type(fig)}","red")

                                elements.append(cl.Plotly(name="Chart", size="medium",figure=fig, display="inline"))

                                # asyncio.run(cl.Message("",elements=elements).send())

                                if self.vission_model:

                                    print_colored("Describing the plotly...","green")

                                    plot_dec = asyncio.run(self.vission_model.get_output(self.vission_model_prompt,base64_image=image_base64))
                                                                        
                                    print_colored(f"Plotly Description.: {tool_output[i]['final_output']}","green")

                                    tool_output[i]['final_output'] =plot_dec

                                    elements.append(cl.Text(name="Insights",content=plot_dec,display="inline"))

                                    # asyncio.run(cl.Message("",elements=elements).send())
                            else:

                                elements.append(cl.Text(name="", content=tool_output[i]['final_output'], display="inline"))

                            asyncio.run(cl.Message("Output",elements=elements).send())
                        
                        tool_output = [i['final_output'][:30000] for i in tool_output]

                        tool_content=f"Output From {tool_details['tool_name']} Tool: {"\n".join(tool_output)}"

                    # elements = [cl.Text("Tool Output",content=str(tool_output))]

                    # asyncio.run(cl.Message("",elements=elements).send())

                except Exception as e:

                    print_colored("Error Tool: "+str(e),"red")

                    tool_content = "Error while executing tool. Please check the tool name or provide a valid arguments to the tool: "+str(e)

        else:

            tool_content= "There is no such a tool available. Here are the available tools : "+str(self.tool_names)

        messages.append({"role":"assistant","content":assistant_content.strip()})
        messages.append({"role":"user","content":tool_content.strip()})

        return messages
        
    async def run(self,user_input=None,messages=[]):

        if self.attempts_made<=self.max_allowed_attempts:

            print_colored(f"Attempt Number : {self.attempts_made}/{self.max_allowed_attempts}","pink")

            self.attempts_made+=1

            if user_input:

                messages = self.prepare_messages(user_input,role="user",messages=messages)

            tool_details,total_tokens = await self.model.aget_output(messages)

            if not isinstance(tool_details,dict):

                return "I am not able to process your request"

            # tool_details = json.loads(tool_details)

            if tool_details['tool_name']=='FinalAnswer':

                thoughts = tool_details.get("thoughts","")

                if isinstance(thoughts,list):

                    thoughts = '\n'.join(thoughts)

                print_colored(f"Thoughts: {thoughts}","green")

                print_colored(f"{self.agent_name} : {tool_details['tool_args']['final_answer']}","green")

                messages.append({"role":"assistant","content":str(tool_details)})

                self.messages = messages

                asyncio.run(cl.Message(tool_details['tool_args']['final_answer']).send())

                # return tool_details['tool_args']['final_answer']

            else:

                print()

                thoughts = tool_details.get("thoughts","")

                if isinstance(thoughts,list):

                    thoughts = '\n'.join(thoughts)

                print_colored(f"Thoughts: {thoughts}","green")

                print_colored(f"Tool Name: {tool_details['tool_name']}","blue")
                # print_colored(f"Tool Name: {tool_details['tool_args']}","blue")
                print()

                if tool_details['tool_name'] == 'JupyterNotebookTool':

                    asyncio.run(cl.Message(thoughts.strip()).send())

                    # html = f"""<pre><code>{tool_details['tool_args']['python_code']}</code></pre>"""
                    html = f"""```python\n\n{tool_details['tool_args']['python_code']}\n\n```"""

                    asyncio.run(cl.Message(html).send())

                messages = self.execute_tool(messages,tool_details)

                self.messages = messages

                return await self.run(messages=messages)

        else:

            self.messages = messages

            print_colored(f"{self.agent_name} : Sorry! Max Attempt Exceeded, I can't take anymore tasks","red")

            return "Sorry! Max Attempt Exceeded, I can't take anymore tasks"
        
import uuid
import asyncio
from pydantic import BaseModel,Field
from core.helper import print_colored
# MISTRAL CHANGE START
from core.models import MistralChatModel, MistralVisionModel
# MISTRAL CHANGE END
from core.text2sql.query_generator_2 import Text2SQL
from core.tools.JupyterTool import NotebookManager
from pydantic import BaseModel, Field

# MISTRAL CHANGE START
# Get credentials from environment variables
db_type = os.getenv('DB_TYPE', 'mysql')
db_host = os.getenv('DB_HOST', 'localhost')
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
db_name = os.getenv('DB_NAME')
mistral_api_key = os.getenv('MISTRAL_API_KEY')
model_name = "mistral-large-latest"
vision_model_name = "pixtral-large-latest"

# Check if essential variables are set
if not all([db_user, db_password, db_name, mistral_api_key]):
    raise ValueError("Please set the environment variables: DB_USER, DB_PASSWORD, DB_NAME, MISTRAL_API_KEY")

# Fix the variable name from SQl_Engine to SQL_Engine
SQL_Engine = Text2SQL(
    model_name, 
    mistral_api_key, 
    db_type='mysql', 
    host='localhost',  # Should be a string
    port=3306, 
    username='root', 
    password='root', 
    database='realtime_finance_db', 
    add_additional_context=True, 
    max_attempts=10
)
# MISTRAL CHANGE END

class GetRelavantTables(BaseModel):
    """
    Tool to retrieve relevant tables based on the user's question.
    """

    user_question : str = Field("Provide the user question as it is.")

    sub_questions: list[str] = Field(
        description=(
            "Split the user question into multiple sub-questions if answering it requires data from multiple tables. "
            "Each sub-question should focus on a specific aspect of the user query, referring to relevant columns or tables. "
            "For example: \n"
            "User question: 'What is the total sales of XYZ product last month?'\n"
            "Sub-questions: ['Which column contains product names?', 'Which column contains sales details?',]\n"
            "Ensure sub-questions are precise and map clearly to tables or columns needed to answer the main query."
            "With each table you will get it's relationship to other tables. So dont need seperate question for that."
            "Avoid duplication. Focus and retrieving the accurate columns."
        )
    )

    def run(self):
        
        docs = SQL_Engine.get_relavant_documents(self.sub_questions, top_n_similar_docs=30,filtered_tables=2)

        filter_result = asyncio.run(SQL_Engine.filter_columns(self.user_question, docs))

        final_schema="# Here are the relevant table schema.\n\n"

        final_schema = "\n\n".join([f"""{i['filtered_columns']}\n\n### Here is the details on how this table connected to other tables\n: {i['common_columns']}""" .strip() for i in filter_result  if i['filtered_columns']])

        with open("relevant_schemas.txt","a") as f:
            f.write(f"User Question : {self.user_question}\n\n")
            f.write(f"Sub Queries\n: {"\n\t".join(self.sub_questions)}\n\n")
            f.write(final_schema.strip())
            f.write("\n\n***************************************************************\n\n")

        return final_schema
    
        # return "\n\n------------------------------------\n\n".join([i['text_data'] for i in docs])

class ExecuteInertmediateQuery(BaseModel):
    """
    Use this tool to execute an intermediate SQL sub-query to retrieve the unique categories or values available in a specific column. 

    Do not directly use the user-provided values in the WHERE clause. First, execute a query to fetch the unique values from the column. 
    
    Based on the retrieved results, validate and construct the final query to ensure accuracy and alignment with the data.

    For example, if the user requests 'drug-x' but the table contains 'Drug-X', this tool ensures the correct value appears at the top.
    """

    user_question: str = Field(description="The user question")
    sub_query: str = Field(description="The sub-query to execute")

    def run(self):
        
        print("Sub Query: ",self.sub_query)

        asyncio.run(cl.Message(f"""```sql\n\n{self.sub_query}\n\n```""").send())

        return f"Observation: {SQL_Engine.execute_inertmediate_query(self.user_question, self.sub_query)}"


class ExecuteFinalQuery(BaseModel):
    """
    Tool to execute the final SQL query and save the output to an Excel file to do further analysis. It will return the exact path to the saved output
    """

    final_query: str = Field(description="The final query to execute")

    def run(self):

        df = SQL_Engine.run_sql_query(self.final_query)

        asyncio.run(cl.Message(f"""```sql\n\n{self.final_query}\n\n```""").send())

        if not df.empty:

            if df.shape[0]>=20:

                df.to_excel("work_dir/query_output.xlsx", index=False)

                return f"Observation: The data has been stored at `work_dir/query_output.xlsx`. The data contains {len(df)} rows, here is the snapshot(First 5 rows) of the dataframe : \n\n" + df.head(5).to_markdown()
            
            else:

                df.to_excel("work_dir/query_output.xlsx", index=False)

                return "Observation: The data has been stored at `work_dir/query_output.xlsx`. \n\n Here is the output of the query" + df.to_markdown()
        else:
            return "The Query Returned Empty DataFrame: \n\n" + df.head(5).to_markdown()


@cl.on_chat_start
def start_message():

    user_session_id = str(uuid.uuid4())

    # FIXED: Add proper error handling for notebook creation
    try:
        notebookmanager = NotebookManager(user_session_id)
        print_colored("Notebook manager created successfully", "green")
    except Exception as e:
        print_colored(f"Failed to create notebook manager: {str(e)}", "red")
        notebookmanager = None

    class JupyterNotebookTool(BaseModel):

        """A tool for executing Python code in a stateful Jupyter notebook environment."""
        
        python_code: str = Field(description="A valid python code to execute in a new jupyter notebook cell")

        def run(self):
            # FIXED: Check if notebookmanager exists before using
            if notebookmanager is None:
                return [{"display_type": "text", "final_output": "Error: Jupyter notebook is not available in this session."}]
            
            try:
                result = notebookmanager.run_code(self.python_code)
                return result
            except Exception as e:
                return [{"display_type": "text", "final_output": f"Error executing code: {str(e)}"}]

    description = "Responsible for Answering User question."

    instruction = open(r"prompts/system_prompt.md","r").read()

    tools = [GetRelavantTables,ExecuteInertmediateQuery,ExecuteFinalQuery,JupyterNotebookTool]

    # MISTRAL CHANGE START
    model = MistralChatModel(model_name=model_name, api_key=mistral_api_key, verbose=True)
    vissionmodel = MistralVisionModel(model=vision_model_name, api_key=mistral_api_key)
    # MISTRAL CHANGE END

    vission_prompt = "You are provided with a plot from a data analysis, You need to explain all the insights and metrics to the user"

    # FIXED: Only create agent if we have a valid notebookmanager or handle gracefully
    try:
        agent = ChainlitStructuredAgent(model,"AI Assistant",description,instruction,tools,max_allowed_attempts=30,vission_model=vissionmodel,vission_model_prompt=vission_prompt)
        print_colored("Agent created successfully", "green")
    except Exception as e:
        print_colored(f"Failed to create agent: {str(e)}", "red")
        agent = None

    cl.user_session.set("messages",[])
    cl.user_session.set("agent",agent)
    cl.user_session.set("notebookmanager",notebookmanager)
    cl.user_session.set("user_session_id",user_session_id)


@cl.on_message
async def on_message(user_input: cl.Message):

    messages=cl.user_session.get("messages")
    agent=cl.user_session.get("agent")

    # FIXED: Check if agent exists before using
    if agent is None:
        await cl.Message("Sorry, the system is not properly initialized. Please refresh and try again.").send()
        return

    try:
        response = await agent.run(user_input.content,messages)
        cl.user_session.set("messages",agent.messages)
    except Exception as e:
        print_colored(f"Error in agent.run: {str(e)}", "red")
        await cl.Message(f"An error occurred: {str(e)}").send()

# FIXED: Correct the disconnect function signature
@cl.on_chat_end
def delete_notebook():
    notebookmanager = cl.user_session.get("notebookmanager")
    user_session_id = cl.user_session.get("user_session_id")

    # FIXED: Check if notebookmanager exists before calling methods
    if notebookmanager is not None and user_session_id is not None:
        try:
            notebookmanager.delete_notebook(user_session_id)
            print_colored("Notebook deleted successfully", "green")
        except Exception as e:
            print_colored(f"Error deleting notebook: {str(e)}", "red")
    else:
        print_colored("No notebook to delete or session not found", "yellow")


# #########################################################################################














# import sys
# sys.dont_write_bytecode = True

# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import chainlit as cl
# import json
# from enum import Enum
# from typing import List, Type
# from pydantic import BaseModel, Field
# from core.helper import print_colored
# import asyncio
# import base64
# import plotly.graph_objects as go

# # -------------------------------- Structured Agent -------------------------------------

# import json
# from enum import Enum
# from typing import List, Type
# from core.helper import print_colored
# from pydantic import BaseModel, Field

# class ChainlitStructuredAgent:

#     def __init__(self, model, agent_name, agent_description, agent_instructions, tools=[], assistant_agents=[], max_allowed_attempts=10, vission_model=None, vission_model_prompt=None) -> None:
#         self.model = model
#         self.vission_model = vission_model
#         self.vission_model_prompt = vission_model_prompt
#         self.agent_name = agent_name
#         self.agent_description = agent_description
#         self.agent_instructions = agent_instructions
#         self.tools = tools
#         self.assistant_agents = assistant_agents
#         self.tool_names = []
#         self.max_allowed_attempts = max_allowed_attempts
#         self.attempts_made = 0
#         self.messages = []
            
#         if len(self.assistant_agents):
#             self.prepare_prompt()
#             self.agents_as_tools = {agent.agent_name: agent for agent in assistant_agents}
#             self.assistants_names = []

#         self.response_format = self.prepare_Default_tools()

#         if len(self.tools):
#             self.tool_objects = {i: j for i, j in zip(self.tool_names, tools)}
#             tool_schemas = self.prepare_schema_from_tool(self.tools)
#             self.agent_instructions += """\n## Available Tools:\n"""
#             self.agent_instructions += f"""\nYou have access to the following tools:\n{tool_schemas}\nYou must use any one of these tools to answer the user question.\n\n"""
#             self.agent_instructions += """IMPORTANT!: You must provide your response in the below json format.
# {
# "thoughts":["Always you should think before taking any action"],
# "tool_name":"Name of the tool",
# "tool_args":{"arg_name":"arg_value"}
# }
# """
        
#     def prepare_Default_tools(self):
#         # Prepare final answer tool
#         class FinalAnswer(BaseModel):
#             final_answer: str = Field(description="Your final response to the user")
#             def run(self):
#                 return self.final_answer
    
#         self.tools.append(FinalAnswer)

#         # Prepare Assign Task tool
#         if len(self.assistant_agents):
#             self.assistants_names = [i.agent_name for i in self.assistant_agents]
#             recipients = Enum("recipient", {name: name for name in self.assistants_names})
#             assistant_description = f"Select the correct Agent to assign the task : {self.assistants_names}\n\n"

#             for assistant in self.assistant_agents:
#                 assistant_description += assistant.agent_name + " : " + assistant.agent_description + "\n"

#             class AssignTask(BaseModel):
#                 """Use this tool to facilitate direct, synchronous communication between specialized agents within your agency. When you send a message using this tool, you receive a response exclusively from the designated recipient agent. To continue the dialogue, invoke this tool again with the desired recipient agent and your follow-up message. Remember, communication here is synchronous; the recipient agent won't perform any tasks post-response. You are responsible for relaying the recipient agent's responses back to the user, as the user does not have direct access to these replies. Keep engaging with the tool for continuous interaction until the task is fully resolved. Do not send more than 1 message at a time."""

#                 my_primary_instructions: str = Field(...,
#                                                     description="Please repeat your primary instructions step-by-step, including both completed "
#                                                                 "and the following next steps that you need to perform. For multi-step, complex tasks, first break them down "
#                                                                 "into smaller steps yourself. Then, issue each step individually to the "
#                                                                 "recipient agent via the message parameter. Each identified step should be "
#                                                                 "sent in separate message. Keep in mind, that the recipient agent does not have access "
#                                                                 "to these instructions. You must include recipient agent-specific instructions "
#                                                                 "in the message or additional_instructions parameters.")
#                 recipient: recipients = Field(..., description=assistant_description, examples=self.assistants_names)

#                 task_details: str = Field(...,
#                                     description="Specify the task required for the recipient agent to complete. Focus on "
#                                                 "clarifying what the task entails, rather than providing exact "
#                                                 "instructions.")

#                 additional_instructions: str = Field(description="Any additional instructions or clarifications that you would like to provide to the recipient agent.")

#             self.tools.append(AssignTask)
                
#         self.tool_names = [i.__name__ for i in self.tools]

#     def prepare_schema_from_tool(self, Tools: List[Type[BaseModel]]) -> str:
#         schemas = ""
#         for tool in Tools:
#             schema = tool.model_json_schema()
#             schemas += "\n"
#             schemas += f"""
# "Tool Name": {tool.__name__},
# "Tool Description": {tool.__doc__},
# "Tool Parameters": 
#     "Properties": {schema["properties"]},
#     "Required": {schema["required"]},
#     "Type": {schema["type"]}\n"""
#             schemas += "\n"
            
#         return schemas

#     def prepare_prompt(self):
#         if len(self.assistant_agents):
#             self.agent_instructions += "\n**Task Assignment**: You can assign tasks to the following agents who are responsible to help you to achieve your goal.\n"
#             self.agent_instructions += "-----------------------------------------------\n"

#             for agent in self.assistant_agents:
#                 self.agent_instructions += "- **Agent Name**: " + agent.agent_name + "\n"
#                 self.agent_instructions += "- **Agent Description**:\n" + agent.agent_description + "\n"

#             self.agent_instructions += "\n-----------------------------------------------\n"
                
#     def prepare_messages(self, content, role=None, messages=[]):
#         if not len(messages):
#             messages = [
#                 {"role": "system", "content": self.agent_instructions},
#                 {"role": "user", "content": content}
#             ]
#         else:
#             messages.append({"role": role, "content": content})

#         return messages
    
#     def construct_message_from_output(self, output):
#         content = ""

#         for key, value in output.items():
#             if key == 'task_checklist':
#                 content += f"Task Check List:\n{chr(10).join(value)}\n"
#             elif key == 'completed_tasks':
#                 content += f"Completed Tasks:\n{chr(10).join(value)}\n"
#             elif key == 'thoughts':
#                 content += f"Thought: {chr(10).join(value)}\n"
#             elif key == 'tool_name':
#                 if value == "AssignTask":
#                     content += f"I need to assign a task to the '{output['tool_args']['recipient']}' with the following instructions:"
#                 else:
#                     content += f"We need to use the '{value}' tool with the following arguments: "
#             else:
#                 content += f"{value}."

#         return content

#     @cl.step(type="tool")
#     def execute_tool(self, messages, tool_details):
#         try:
#             assistant_content = str(tool_details)
#         except Exception as e:
#             invalid_arg_error_message = "Error while executing tool. Please check the tool name or provide a valid arguments to the tool: " + str(e)
#             tool_content = invalid_arg_error_message
#             assistant_content = str(tool_details)
#             messages.append({"role": "assistant", "content": assistant_content})
#             messages.append({"role": "user", "content": tool_content})
#             return messages

#         if tool_details['tool_name'] in self.tool_names:
#             if tool_details['tool_name'] == 'AssignTask':
#                 try:
#                     arguments = tool_details['tool_args']
#                     task_details = arguments.get('task_details', "")
#                     additional_instructions = arguments.get('additional_instructions', "")
#                     print_colored(f"{self.agent_name} assigned a task to {arguments['recipient']}", "cyan")
#                     assistant_agent = self.agents_as_tools[arguments['recipient']]
#                     user_input = task_details + "\n" + additional_instructions
#                     print_colored("Task Details: \n" + user_input, "white")
#                     tool_content = assistant_agent.run(user_input)
#                     tool_content = f"Response from the {arguments['recipient']} : " + tool_content
                    
#                 except Exception as e:
#                     print_colored("Error Tool: " + str(e), "red")
#                     tool_content = f"Error while assigning task to {arguments['recipient']}. Please provide a correct agent name: {[i.agent_name for i in self.assistant_agents]}"

#             else:
#                 try:
#                     print_colored(f"{self.agent_name} : Calling Tool {tool_details['tool_name']}", "yellow")
#                     tool_output = self.tool_objects[tool_details['tool_name']](**tool_details['tool_args']).run()
#                     print_colored(f"Got output....{len(str(tool_output))}", "orange")
#                     tool_content = f"Output From {tool_details['tool_name']} Tool: {tool_output}"

#                     if tool_details['tool_name'] == 'JupyterNotebookTool' and len(str(tool_output)):
#                         if isinstance(tool_output, list):
#                             for i in range(len(tool_output)):
#                                 elements = []

#                                 if tool_output[i]['display_type'] == 'text':
#                                     elements.append(cl.Text(name="Tool Output", content=tool_output[i]['final_output'], display="inline"))

#                                 elif tool_output[i]['display_type'] == 'image':
#                                     image_data = tool_output[i]['output'].split('base64,')[1]
#                                     image_bytes = base64.b64decode(image_data)
#                                     elements.append(cl.Image(name="Plot", size='medium', content=image_bytes, display="inline"))
#                                     image_base64 = base64.b64encode(base64.b64decode(image_data)).decode('utf-8')

#                                     if self.vission_model:
#                                         print_colored("Describing the Image...", "green")
#                                         plot_dec = asyncio.run(self.vission_model.get_output(self.vission_model_prompt, base64_image=image_base64))
#                                         tool_output[i]['final_output'] += "\n" + plot_dec
#                                         print_colored(f"Image Description.: {tool_output[i]['final_output']}", "green")
#                                         tool_output[i]['final_output'] = plot_dec
#                                         elements.append(cl.Text(name="Insights", content=plot_dec))

#                                 elif tool_output[i]['display_type'] == 'plotly':
#                                     print_colored("Trying to display plotly.....", "red")
#                                     fig = go.Figure(data=tool_output[i]['output'])
#                                     print_colored(f"Created fig.....{type(fig)}", "red")
#                                     img_bytes = fig.to_image(format="png")
#                                     print_colored(f"Created img_bytes.....{type(img_bytes)}", "red")
#                                     image_base64 = base64.b64encode(img_bytes).decode('utf-8')
#                                     print_colored(f"Created image_base64.....{len(image_base64)}", "red")
#                                     elements.append(cl.Plotly(name="Chart", size="medium", figure=fig, display="inline"))

#                                     if self.vission_model:
#                                         print_colored("Describing the plotly...", "green")
#                                         plot_dec = asyncio.run(self.vission_model.get_output(self.vission_model_prompt, base64_image=image_base64))
#                                         print_colored(f"Plotly Description.: {tool_output[i]['final_output']}", "green")
#                                         tool_output[i]['final_output'] = plot_dec
#                                         elements.append(cl.Text(name="Insights", content=plot_dec, display="inline"))
#                                 else:
#                                     elements.append(cl.Text(name="", content=tool_output[i]['final_output'], display="inline"))

#                                 asyncio.run(cl.Message("Output", elements=elements).send())
                        
#                             tool_output = [i['final_output'][:30000] for i in tool_output]
#                             tool_content = f"Output From {tool_details['tool_name']} Tool: {chr(10).join(tool_output)}"

#                 except Exception as e:
#                     print_colored("Error Tool: " + str(e), "red")
#                     tool_content = "Error while executing tool. Please check the tool name or provide a valid arguments to the tool: " + str(e)

#         else:
#             tool_content = "There is no such a tool available. Here are the available tools : " + str(self.tool_names)

#         messages.append({"role": "assistant", "content": assistant_content.strip()})
#         messages.append({"role": "user", "content": tool_content.strip()})
#         return messages
        
#     async def run(self, user_input=None, messages=[]):
#         if self.attempts_made <= self.max_allowed_attempts:
#             print_colored(f"Attempt Number : {self.attempts_made}/{self.max_allowed_attempts}", "pink")
#             self.attempts_made += 1

#             if user_input:
#                 messages = self.prepare_messages(user_input, role="user", messages=messages)

#             tool_details, total_tokens = await self.model.aget_output(messages)

#             if not isinstance(tool_details, dict):
#                 return "I am not able to process your request"

#             if tool_details['tool_name'] == 'FinalAnswer':
#                 thoughts = tool_details.get("thoughts", "")
#                 if isinstance(thoughts, list):
#                     thoughts = '\n'.join(thoughts)

#                 print_colored(f"Thoughts: {thoughts}", "green")
#                 print_colored(f"{self.agent_name} : {tool_details['tool_args']['final_answer']}", "green")
#                 messages.append({"role": "assistant", "content": str(tool_details)})
#                 self.messages = messages
#                 asyncio.run(cl.Message(tool_details['tool_args']['final_answer']).send())

#             else:
#                 print()
#                 thoughts = tool_details.get("thoughts", "")
#                 if isinstance(thoughts, list):
#                     thoughts = '\n'.join(thoughts)

#                 print_colored(f"Thoughts: {thoughts}", "green")
#                 print_colored(f"Tool Name: {tool_details['tool_name']}", "blue")
#                 print()

#                 if tool_details['tool_name'] == 'JupyterNotebookTool':
#                     asyncio.run(cl.Message(thoughts.strip()).send())
#                     html = f"""```python\n\n{tool_details['tool_args']['python_code']}\n\n```"""
#                     asyncio.run(cl.Message(html).send())

#                 messages = self.execute_tool(messages, tool_details)
#                 self.messages = messages
#                 return await self.run(messages=messages)

#         else:
#             self.messages = messages
#             print_colored(f"{self.agent_name} : Sorry! Max Attempt Exceeded, I can't take anymore tasks", "red")
#             return "Sorry! Max Attempt Exceeded, I can't take anymore tasks"
        
# import uuid
# import asyncio
# from pydantic import BaseModel, Field
# from core.helper import print_colored
# # MISTRAL CHANGE START
# from core.models import MistralChatModel, MistralVisionModel
# # MISTRAL CHANGE END
# from core.text2sql.query_generator_2 import Text2SQL
# from core.tools.JupyterTool import NotebookManager
# from pydantic import BaseModel, Field

# # MISTRAL CHANGE START
# # Get credentials from environment variables
# db_type = os.getenv('DB_TYPE', 'mysql')
# db_host = os.getenv('DB_HOST', 'localhost')
# db_user = os.getenv('DB_USER')
# db_password = os.getenv('DB_PASSWORD')
# db_name = os.getenv('DB_NAME')
# mistral_api_key = os.getenv('MISTRAL_API_KEY')
# model_name = "mistral-large-latest"
# vision_model_name = "pixtral-large-latest"

# # Check if essential variables are set
# if not all([db_user, db_password, db_name, mistral_api_key]):
#     raise ValueError("Please set the environment variables: DB_USER, DB_PASSWORD, DB_NAME, MISTRAL_API_KEY")

# SQL_Engine = Text2SQL(
#     model_name, 
#     mistral_api_key, 
#     db_type='mysql', 
#     host='localhost',
#     port=3306, 
#     username=db_user,  # Fixed: Use environment variable
#     password=db_password,  # Fixed: Use environment variable
#     database=db_name,  # Fixed: Use environment variable
#     add_additional_context=True, 
#     max_attempts=10
# )
# # MISTRAL CHANGE END

# class GetRelavantTables(BaseModel):
#     """
#     Tool to retrieve relevant tables based on the user's question.
#     """

#     user_question: str = Field("Provide the user question as it is.")

#     sub_questions: list[str] = Field(
#         description=(
#             "Split the user question into multiple sub-questions if answering it requires data from multiple tables. "
#             "Each sub-question should focus on a specific aspect of the user query, referring to relevant columns or tables. "
#             "For example: \n"
#             "User question: 'What is the total sales of XYZ product last month?'\n"
#             "Sub-questions: ['Which column contains product names?', 'Which column contains sales details?',]\n"
#             "Ensure sub-questions are precise and map clearly to tables or columns needed to answer the main query."
#             "With each table you will get it's relationship to other tables. So dont need seperate question for that."
#             "Avoid duplication. Focus and retrieving the accurate columns."
#         )
#     )

#     def run(self):
#         docs = SQL_Engine.get_relavant_documents(self.sub_questions, top_n_similar_docs=30, filtered_tables=2)
#         filter_result = asyncio.run(SQL_Engine.filter_columns(self.user_question, docs))
#         final_schema = "# Here are the relevant table schema.\n\n"
#         final_schema = "\n\n".join([f"""{i['filtered_columns']}\n\n### Here is the details on how this table connected to other tables\n: {i['common_columns']}""".strip() for i in filter_result if i['filtered_columns']])

#         with open("relevant_schemas.txt", "a") as f:
#             f.write(f"User Question : {self.user_question}\n\n")
#             f.write(f"Sub Queries\n: {chr(10).join(self.sub_questions)}\n\n")  # Fixed: use chr(10) instead of join with \n
#             f.write(final_schema.strip())
#             f.write("\n\n***************************************************************\n\n")

#         return final_schema
    
# class ExecuteInertmediateQuery(BaseModel):  # Fixed: typo "Inertmediate" to "Intermediate"
#     """
#     Use this tool to execute an intermediate SQL sub-query to retrieve the unique categories or values available in a specific column. 

#     Do not directly use the user-provided values in the WHERE clause. First, execute a query to fetch the unique values from the column. 
    
#     Based on the retrieved results, validate and construct the final query to ensure accuracy and alignment with the data.

#     For example, if the user requests 'drug-x' but the table contains 'Drug-X', this tool ensures the correct value appears at the top.
#     """

#     user_question: str = Field(description="The user question")
#     sub_query: str = Field(description="The sub-query to execute")

#     def run(self):
#         print("Sub Query: ", self.sub_query)  # Fixed: typo "Quer" to "Query"
#         asyncio.run(cl.Message(f"""```sql\n\n{self.sub_query}\n\n```""").send())
#         return f"Observation: {SQL_Engine.execute_inertmediate_query(self.user_question, self.sub_query)}"  # Fixed: variable name

# class ExecuteFinalQuery(BaseModel):
#     """
#     Tool to execute the final SQL query and save the output to an Excel file to do further analysis. It will return the exact path to the saved output
#     """

#     final_query: str = Field(description="The final query to execute")

#     def run(self):
#         df = SQL_Engine.run_sql_query(self.final_query)  # Fixed: variable name
#         asyncio.run(cl.Message(f"""```sql\n\n{self.final_query}\n\n```""").send())

#         if not df.empty:
#             if df.shape[0] >= 20:
#                 df.to_excel("work_dir/query_output.xlsx", index=False)
#                 return f"Observation: The data has been stored at `work_dir/query_output.xlsx`. The data contains {len(df)} rows, here is the snapshot(First 5 rows) of the dataframe : \n\n" + df.head(5).to_markdown()
#             else:
#                 df.to_excel("work_dir/query_output.xlsx", index=False)
#                 return "Observation: The data has been stored at `work_dir/query_output.xlsx`. \n\n Here is the output of the query" + df.to_markdown()
#         else:
#             return "The Query Returned Empty DataFrame: \n\n" + df.head(5).to_markdown()  # Fixed: typo "DatafRame" to "DataFrame"


# @cl.on_chat_start
# def start_message():
#     user_session_id = str(uuid.uuid4())
#     notebookmanager = NotebookManager(user_session_id)

#     class JupyterNotebookTool(BaseModel):
#         """A tool for executing Python code in a stateful Jupyter notebook environment."""
        
#         python_code: str = Field(description="A valid python code to execute in a new jupyter notebook cell")

#         def run(self):
#             result = notebookmanager.run_code(self.python_code)
#             return result

#     description = "Responsible for Answering User question."
#     instruction = open(r"prompts/system_prompt.md", "r").read()
#     tools = [GetRelavantTables, ExecuteInertmediateQuery, ExecuteFinalQuery, JupyterNotebookTool]  # Fixed: typo

#     # MISTRAL CHANGE START
#     model = MistralChatModel(model_name=model_name, api_key=mistral_api_key, verbose=True)
#     vissionmodel = MistralVisionModel(model=vision_model_name, api_key=mistral_api_key)
#     # MISTRAL CHANGE END

#     vission_prompt = "You are provided with a plot from a data analysis, You need to explain all the insights and metrics to the user"

#     agent = ChainlitStructuredAgent(model, "AI Assistant", description, instruction, tools, max_allowed_attempts=30, vission_model=vissionmodel, vission_model_prompt=vission_prompt)

#     cl.user_session.set("messages", [])
#     cl.user_session.set("agent", agent)
#     cl.user_session.set("notebookmanager", notebookmanager)
#     cl.user_session.set("user_session_id", user_session_id)


# @cl.on_message
# async def on_message(user_input: cl.Message):
#     messages = cl.user_session.get("messages")
#     agent = cl.user_session.get("agent")
#     response = await agent.run(user_input.content, messages)
#     cl.user_session.set("messages", agent.messages)


# @cl.on_chat_end
# def delete_notebook():
#     notebookmanager = cl.user_session.get("notebookmanager")
#     user_session_id = cl.user_session.get("user_session_id")
#     notebookmanager.delete_notebook(user_session_id)