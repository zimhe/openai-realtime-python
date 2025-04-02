from typing import Any

import replicate.prediction
from realtime_agent.tools import ToolContext

from agora_realtime_ai_api.rtc import Channel, ChatMessage, RtcEngine, RtcOptions
from .struct import to_json
from ..utils import PCMWriter
import asyncio
import replicate
import uuid
import json
import aiohttp
import os
from openai import OpenAI

# Function calling Example
# This is an example of how to add a new function to the agent tools.

t2i_api="black-forest-labs/flux-dev"
t2i_condition_api="black-forest-labs/flux-canny-dev"
tqa_api="black-forest-labs/flux-tqa-dev"

class AgentToolsMetaWorkplaces(ToolContext):
    def __init__(self) -> None:
        super().__init__()

        self.agent=None
        self.channel=None
        self.screen_keys=None
        self.openai_client=OpenAI()
        self.history_images={}
        self.t2i_condition_queue=asyncio.Queue()
        # create multiple functions here as per requirement
        self.register_function(
            name="set_mute_state",
            description="set the agent to be muted or unmuted, True for muted, False for unmuted", 
            parameters={
                "type": "object",
                "properties": {
                    "mute_state": {
                        "type": "boolean",
                        "description": "the mute state, True for muted, False for unmuted",
                    },
                },
            },
            fn=self._set_mute_state,
        )
        
        self.register_function(
            name="check_mute_state",
            description="check the current mute state of the agent, the agent can check if it's currently muted or not when user ask the agent for response.",
            parameters={
                "type": "object",
                "properties": {},
            },
            fn=self._check_mute_state,
        )

        self.register_function(
            name="text_to_image",
            description="generate image from text prompt, if user gives a screen number, the function should also specify a <screen_id> in the args, so that the image will be sent to the screen", 
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "the text prompt for generating image",
                    },
                    "screen_id": {
                        "type": "integer",
                        "description": "Optional, the screen index to cast the image to if user wants to put the image to a particular screen, this argument should be specified in the args",
                    }
                },
                "required":["prompt"]
            },
            fn=self._text2image,
        )
        
        self.register_function(
            name="text_to_image_condition",
            description="generate image from text prompt and given condition image, if user gives a screen number, the function should also specify a <screen_id> in the args, so that the image will be sent to the screen", 
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "the text prompt for generating image",
                    },
                    "condition_url":{
                        "type": "string",
                        "description": "the condition image url for generating image",
                    },
                    "screen_id": {
                        "type": "integer",
                        "description": "Optional, the screen index to cast the image to if user wants to put the image to a particular screen, this argument should be specified in the args",
                    }
                },
                "required":["prompt","condition_url"]
            },
            fn=self._text2image_condition,
        )
        
        self.register_function(
            name="get_web_search_results",
            description="perform an online search for relevant information based on given query keywords",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query string."},
                },
            },
            fn=self._get_web_search_results,
        )
        
        self.register_function(
            name="get_history_image_keys",
            description="get the history images info with json format that uses prompt as key and image url as value",
            parameters={
                "type": "object",
                "properties": {},
            },
            fn=self._get_history_images_info,
        )
    
    def process_tool_config(self,action:str,data):
        match action:
            case "set_screen_context":
                self.set_screen_context(data)
            case "set_t2i_condition":
                self.t2i_condition_queue.put_nowait(data)
                #print(f"Setting t2i_condition: {id(self)}")
            case _:
                pass
            
        
    def set_agent(self,agent):
        self.agent=agent
        self.channel=agent.channel
        
    def set_screen_context(self,screen_keys):
        self.screen_keys=screen_keys

    async def _set_mute_state(self,mute_state:bool)-> dict[str, Any]:
        
        try:
            self.agent.muted=mute_state
            return {
                "status": "success",
                "message": f"Set mute state to {mute_state} successfully.",
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to set mute state to {mute_state}",
            }
            
    async def _check_mute_state(self)-> dict[str, Any]:
        try:
            
            mute_state=""
            
            if self.agent.muted:
                mute_state="Agent is muted now, the users won't hear the agent's voice."
            else:
                mute_state="Agent is unmuted now, the users can hear the agent's voice."
            
            return {
                "status": "success",
                "message": mute_state,
                "result": self.agent.muted,
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to get mute state",
            }
            

    async def _text2image(self,prompt: str,screen_id:int=-1) -> dict[str, Any]:
        try:
            input = { "prompt": prompt,"guidance": 3.5,"output_format":"png","aspect_ratio":"16:9"}
            
            output=await replicate.async_run(t2i_api,input,use_file_output=False)
            
            screen_key=None
            proper_screen_id=screen_id-1
            if screen_id!=-1 and self.screen_keys is not None and proper_screen_id<len(self.screen_keys):
                screen_key=self.screen_keys[proper_screen_id]
            
            result={"text_to_image_output":output[0],"screen_key":screen_key}
            
            msg_id=uuid.uuid4().hex
            
            chat_message=ChatMessage(message=json.dumps(result),msg_id=msg_id)
            
            self.history_images[prompt]=output[0]
            
            await self.channel.chat.send_message(chat_message)
            
            return {
                "status": "success",
                "message": f"text to image success, the result is {output[0]}",
                "result": output[0],
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to get : {str(e)}",
            }
            
            
    
    async def _text2image_condition(self,prompt: str,condition_url:str,screen_id:int=-1) -> dict[str, Any]:
        try:
 
            input = { 
                     "prompt": prompt,
                     "guidance": 3.5,
                     "output_format":"png",
                     "aspect_ratio":"16:9",
                     "control_image":condition_url
                     }
            
            output=await replicate.async_run(t2i_condition_api,input,use_file_output=False)
            
            screen_key=None
            proper_screen_id=screen_id-1
            if screen_id!=-1 and self.screen_keys is not None and proper_screen_id<len(self.screen_keys):
                screen_key=self.screen_keys[proper_screen_id]
            
            result={"text_to_image_output":output[0],"screen_key":screen_key}
            
            msg_id=uuid.uuid4().hex
            
            chat_message=ChatMessage(message=json.dumps(result),msg_id=msg_id)
            
            self.history_images[prompt]=output[0]
            
            await self.channel.chat.send_message(chat_message)
            
            return {
                "status": "success",
                "message": f"text to image success, the result is {output[0]}",
                "result": output[0],
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to get : {str(e)}",
            }
            

            
    async def _get_history_images_info(self)-> dict[str, Any]:
        try:
            image_info=json.dumps(self.history_images)
            return {
                "status": "success",
                "message": f"Get history images info success",
                "result": image_info,
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to get history images {e}",
            }
            
            
    async def _cast_image_to_screen(self,image_url,screen_id=1)-> dict[str, Any]:
        try:
            screen_key=None
            proper_screen_id=screen_id-1
            if screen_id!=-1 and self.screen_keys is not None and proper_screen_id<len(self.screen_keys):
                screen_key=self.screen_keys[proper_screen_id]
            
            msg_id=uuid.uuid4().hex
            
            chat_message=ChatMessage(message=json.dumps({"text_to_image_output":image_url}),msg_id=msg_id)
            
            await self.channel.chat.send_message(chat_message)
            
            return {
                "status": "success",
                "message": f"cast image to screen success",
                "result": image_url,
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to cast image to screen, {e}",
            }
           
            
    async def _get_web_search_results(self,query: str) -> dict[str, Any]:
        """
        本地函数: 通过 OpenAI 的 gpt-4o-search-preview 执行网络搜索
        """
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-search-preview",
                # web_search_options={
                #     "search_context_size": "low",  # 设定搜索范围
                # },
                messages=[{
                    "role": "user",
                    "content": query,
                }],
            )

            # 解析返回结果
            result_text = response.choices[0].message.content if response.choices else "No results found"

            return {
                "status": "success",
                "message": f"Search results for '{query}': {result_text}",
                "result": result_text,
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to fetch search results: {str(e)}",
            }
    async def _interpert_image(self,image_url:str,query: str) -> dict[str, Any]:
         result="Success"
         return {
                "status": "success",
                "message": f"Search results for '{query}': {result}",
                "result": result,
            }
