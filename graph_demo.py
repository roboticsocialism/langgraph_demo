"""
Script Name: graph_demo.py
Author: Robotic Socialism
Date: January 21, 2024
Description: This examples builds off the base chat executor. 
It is highly recommended you learn about that executor before going through this script. 
You can find documentation for that example in official langgraph github.

"""

from langchain import hub
from langchain.agents import Tool, create_react_agent

from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAI

import os

os.environ["SERPER_API_KEY"] = "<Your Key here>"

from langchain_community.utilities import GoogleSerperAPIWrapper
search = GoogleSerperAPIWrapper()

tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
    ),

]

prompt = hub.pull("hwchase17/react")

llm = ChatGoogleGenerativeAI(model="gemini-pro",
                        google_api_key="<Your Key here>",
                        convert_system_message_to_human = True,
                        verbose = True,
                            )

agent_runnable = create_react_agent(llm, tools, prompt)


from typing import TypedDict, Annotated, List, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
import operator
from typing import TypedDict, Annotated, Sequence


class AgentState(TypedDict):
   input: str
   chat_history: list[BaseMessage]
   agent_outcome: Union[AgentAction, AgentFinish, None]
   return_direct: bool
   intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]


from langchain_core.agents import AgentFinish
from langgraph.prebuilt.tool_executor import ToolExecutor


tool_executor = ToolExecutor(tools)


from langchain_core.agents import AgentActionMessageLog



def run_agent(state):
    """
    #if you want to better manages intermediate steps
    inputs = state.copy()
    if len(inputs['intermediate_steps']) > 5:
        inputs['intermediate_steps'] = inputs['intermediate_steps'][-5:]
    """
    agent_outcome = agent_runnable.invoke(state)
    return {"agent_outcome": agent_outcome}



from langgraph.prebuilt import ToolInvocation

def execute_tools(state):

    messages = [state['agent_outcome'] ]
    last_message = messages[-1]
    ######### human in the loop ###########   
    # human input y/n 
    # Get the most recent agent_outcome - this is the key added in the `agent` above
    # state_action = state['agent_outcome']
    # human_key = input(f"[y/n] continue with: {state_action}?")
    # if human_key == "n":
    #     raise ValueError
    
    tool_name = last_message.tool
    arguments = last_message
    if tool_name == "Search":
        
        if "return_direct" in arguments:
            del arguments["return_direct"]
    action = ToolInvocation(
        tool=tool_name,
        tool_input= last_message.tool_input,
    )
    response = tool_executor.invoke(action)
    return {"intermediate_steps": [(state['agent_outcome'],response)]}

    
def should_continue(state):

    messages = [state['agent_outcome'] ] 
    last_message = messages[-1]
    if "Action" not in last_message.log:
        return "end"
    else:
        arguments = state["return_direct"]
        if arguments is True:
            return "final"
        else:
            return "continue"
        

def first_agent(inputs):
    action = AgentActionMessageLog(
      tool="Search",
      tool_input=inputs["input"],
      log="",
      message_log=[]
    )
    return {"agent_outcome": action}


from langgraph.graph import END, StateGraph


workflow = StateGraph(AgentState)

workflow.add_node("agent", run_agent)
workflow.add_node("action", execute_tools)
workflow.add_node("final", execute_tools)
# uncomment if you want to always calls a certain tool first
# workflow.add_node("first_agent", first_agent)


workflow.set_entry_point("agent")
# uncomment if you want to always calls a certain tool first
# workflow.set_entry_point("first_agent")

workflow.add_conditional_edges(

    "agent",
    should_continue,

    {
        "continue": "action",
        "final": "final",
        "end": END
    }
)


workflow.add_edge('action', 'agent')
workflow.add_edge('final', END)
# uncomment if you want to always calls a certain tool first
# workflow.add_edge('first_agent', 'action')
app = workflow.compile()

inputs = {"input": "what is the weather in Taipei", "chat_history": [],"return_direct": False}

for s in app.stream(inputs):
    print(list(s.values())[0])
    print("----")

