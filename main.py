from typing import Union
from autogen.agentchat.agent import Agent
from autogen import AssistantAgent, UserProxyAgent, GroupChatManager, GroupChat
from fastapi import FastAPI
from pydantic import BaseModel
import json
import os

class UserChatLine(BaseModel):
    chat_id: Union[str, None] = None
    message: str

app = FastAPI()

def initiate_agents(chat_history: list = []):
    llm_config = {
        "config_list": [
            {
                "model": "gpt-3.5-turbo",
                "api_key": "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"
            }
        ]
    }
    # create an AssistantAgent instance named "assistant"
    assistant = AssistantAgent(
        name="Assistant", llm_config=llm_config,
        system_message="Assistant. Provide support for any questions but code. You do not write code, instead, let the Engineer do it. You do not write code.",)
    
    code_assistant = AssistantAgent(
        name="Engineer", llm_config=llm_config,
        system_message="""Engineer. You write python/shell code to solve tasks, then tell the executor to execute such codes. Wrap the code in a code block that specifies the script type. The user can't modify your code. So do not suggest incomplete code which requires others to modify. Don't use a code block if it's not intended to be executed by the executor.
        Don't include multiple code blocks in one response. Do not ask others to copy and paste the result. Check the execution result returned by the executor.
        If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
        """,
        description = "Engineer. Only writes code."
    )

    user_proxy = UserProxyAgent(
        name="User", human_input_mode="NEVER", max_consecutive_auto_reply=0, code_execution_config={"use_docker": False}, system_message="User. A human admin user.",
        )
    
    executor = UserProxyAgent(
        name="Executor",
        system_message="Executor. Execute every code written and report the result.",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=5,
        code_execution_config={
            "last_n_messages": 3,
            "work_dir": "coding",
            "use_docker": False,
        }, 
    )
    
    speaker_transitions_dict = {user_proxy: [user_proxy, executor], code_assistant:[user_proxy, assistant, code_assistant]} #do not force user to repeat

    groupchat = GroupChat(
            agents=[user_proxy, assistant, code_assistant, executor], 
            messages=chat_history, 
            allowed_or_disallowed_speaker_transitions=speaker_transitions_dict,
            speaker_transitions_type="disallowed",
    )
    for agent in groupchat.agents:
        agent.reset()
    manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config, name="Manager", 
                                        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),)
    
    for message in chat_history:
        # broadcast the message to all agents except the speaker.  This idea is the same way GroupChat is implemented in AutoGen for new messages, this method simply allows us to replay old messages first.
        for agent in manager.groupchat.agents:
            if agent != manager:
                manager.send(message, agent, request_reply=False, silent=True)


    return user_proxy, manager

def retrieve_chat_history(chat_id: int) -> tuple[UserProxyAgent, GroupChatManager]:
    
    if os.path.exists(f'./chat_history/{chat_id}.json'):
        with open(f'./chat_history/{chat_id}.json', 'r') as f:
            chat_history = json.load(f)
            user_proxy, manager = initiate_agents(chat_history)
            print(f"retrieve chat history of chat {chat_id}")
    else:
        user_proxy, manager = initiate_agents()
    return user_proxy, manager

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.get("/chat/")
def initiate_chat_screen():
    pass

@app.post("/chat/")
def chatting(user_chat_line: UserChatLine):
    chat_id, message = user_chat_line.chat_id, user_chat_line.message
    user_proxy, manager = retrieve_chat_history(chat_id)

    chat_result = user_proxy.initiate_chat(manager, False, True, message=message)
    save_dir = f'./chat_history/{chat_id}.json'
    with open(save_dir, 'w') as f:
        json.dump(manager.groupchat.messages, f)
    return chat_result.summary

