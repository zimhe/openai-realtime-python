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

t2i_api="black-forest-labs/flux-kontext-pro"
t2i_condition_api="black-forest-labs/flux-depth-pro"
t2i_image_mask_api="black-forest-labs/flux-fill-pro"
aliyun_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
aliyun_model_id="qwen2.5-vl-7b-instruct"

class AgentToolsMetaWorkplaces(ToolContext):
    def __init__(self) -> None:
        super().__init__()

        self.agent=None
        self.channel=None
        self.screen_keys=None
        self.openai_client=OpenAI()
        self.history_images={}
        self.t2i_condition_queue=asyncio.Queue()
        self.aliyun_client=OpenAI(
            base_url=aliyun_url,
            api_key=os.environ.get("ALIYUN_API_KEY")
            )
        

        self.register_function(
            name="text_to_image",
            description="generate image from natural language prompt, if user gives a screen number, the function should also specify a <screen_id> in the args, so that the image will be sent to the screen", 
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
            name="refine_this_image",
            description="refine this image with natural language prompts.if user gives a screen number, the function should also specify a <screen_id> in the args, so that the image will be sent to the screen", 
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "the text prompt for generating image",
                    },
                    "input_image":{
                        "type": "string",
                        "description": "the condition image url for generating image",
                    },
                    "screen_id": {
                        "type": "integer",
                        "description": "Optional, the screen index to cast the image to if user wants to put the image to a particular screen, this argument should be specified in the args",
                    }
                },
                "required":["prompt","input_image"]
            },
            fn=self._refine_this_image,
        )
        
        self.register_function(
            name="refine_masked_area_of_image",
            description="re-generate the masked area in the given image from text prompt and given condition image, the prompt should be refined with delicate details for better quality.if user gives a screen number, the function should also specify a <screen_id> in the args, so that the result image will be sent to the screen", 
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "the text prompt for generating image",
                    },
                    "image_url":{
                        "type": "string",
                        "description": "the image url for the base image to modify",
                    },
                      "mask_url":{
                        "type": "string",
                        "description": "the mask image url for defining the area to be redrawn",
                    },
                    "screen_id": {
                        "type": "integer",
                        "description": "Optional, the screen index to cast the image to if user wants to put the image to a particular screen, this argument should be specified in the args",
                    }
                },
                "required":["prompt","image_url","mask_url"]
            },
            fn=self._refine_image_with_mask,
        )
        
        self.register_function(
            name="get_web_search_results",
            description="perform an online search for relevant information based on given query keywords, can be used when ever the user querys something that is not in the local knowledge base, the function will return a json object with the search results",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query string."},
                },
            },
            fn=self._get_web_search_results,
        )
        
        self.register_function(
            name="get_history_image_info",
            description="get the history images info with json format that uses prompt as key and image url as value",
            parameters={
                "type": "object",
                "properties": {},
            },
            fn=self._get_history_images_info,
        )
        
        self.register_function(
            name="interpert_image",
            description="interpert the image with given query, the image url should be a valid url",
            parameters={
                "type":"object",
                "properties":{
                    "image_url":{
                        "type":"string",
                        "description":"the image url that needs to be interpreted"
                    },
                    "query":{
                        "type":"string",
                        "description":"the query that needs to be interpreted"
                    }
                },
                "required":["image_url","query"]
            },
            fn=self._interpert_image,
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
        
    
    def format_t2i_result(self, output, screen_key,item_id):
        result={"type":"text2image.output","output":output,"screen_key":screen_key,"item_id":item_id}
        return result



    async def _text2image(self,prompt:str,screen_id:int=-1) -> dict[str, Any]:
        try:
            input = { "prompt": prompt,"output_format":"png","aspect_ratio":"16:9" }
            
            output=await replicate.async_run(t2i_api,input,use_file_output=False)
            
            screen_key=None
            proper_screen_id=screen_id-1
            if screen_id!=-1 and self.screen_keys is not None and proper_screen_id<len(self.screen_keys):
                screen_key=self.screen_keys[proper_screen_id]
            
            if isinstance(output, list):
                output=output[0]  # Ensure we get the first image if multiple are returned
            
            msg_id=uuid.uuid4().hex
            
            result=self.format_t2i_result(output,screen_key,msg_id)
            
            chat_message=ChatMessage(message=json.dumps(result),msg_id=msg_id)
            
            self.history_images[prompt]=output
            
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
            
            

    async def _refine_this_image(self,prompt: str,input_image:str,screen_id:int=-1) -> dict[str, Any]:
        try:
            input = { 
                     "prompt": prompt,
                     "aspect_ratio":"match_input_image",
                     "output_format":"png",
                     "input_image":input_image
                     }
            
            output=await replicate.async_run(t2i_api,input,use_file_output=False)
            
            if isinstance(output, list):
                output=output[0]  # Ensure we get the first image if multiple are returned
            

            screen_key=None
            proper_screen_id=screen_id-1
            if screen_id!=-1 and self.screen_keys is not None and proper_screen_id<len(self.screen_keys):
                screen_key=self.screen_keys[proper_screen_id]
            
            
            msg_id=uuid.uuid4().hex
            result=self.format_t2i_result(output,screen_key,msg_id)
            
            chat_message=ChatMessage(message=json.dumps(result),msg_id=msg_id)
            
            self.history_images[prompt]=output
            
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
            
            
    async def _refine_image_with_mask(self,prompt:str,image_url:str,mask_url:str,screen_id:int=-1)-> dict[str, Any]:
        try:

            input={
                    "mask": mask_url,
                    "image": image_url,
                    "steps": 30,
                    "prompt": prompt,
                    "guidance": 60,
                    "outpaint": "None",
                    "output_format": "png",
                    "safety_tolerance": 2,
                    "prompt_upsampling": False
                }
            
            output=await replicate.async_run(t2i_image_mask_api,input,use_file_output=False)
            
            screen_key=None
            proper_screen_id=screen_id-1
            if screen_id!=-1 and self.screen_keys is not None and proper_screen_id<len(self.screen_keys):
                screen_key=self.screen_keys[proper_screen_id]
            
            if isinstance(output, list):
                output=output[0]  # Ensure we get the first image if multiple are returned
            
            msg_id=uuid.uuid4().hex
            
            result=self.format_t2i_result(output,screen_key,msg_id)
            
            chat_message=ChatMessage(message=json.dumps(result),msg_id=msg_id)
            
            self.history_images[prompt]=output
            
            await self.channel.chat.send_message(chat_message)
            
            return {
                "status": "success",
                "message": f"refine image success, the result is {output}",
                "result": output,
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
            
            result=self.format_t2i_result(image_url,screen_key,msg_id)
            
            chat_message=ChatMessage(message=json.dumps(result),msg_id=msg_id)
            
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
        message={
            "role": "user",
            "content":[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                    }
                },
                {
                    "type": "text",
                    "text": query,
                }
            ]
        }
         
        try:
            completion = self.aliyun_client.chat.completions.create(
                model=aliyun_model_id,
                stream=False,
                max_tokens=512,
                temperature=0.7,
                messages=[message]
            )
            
            data = json.loads(completion.model_dump_json())
            result = data["choices"][0]["message"]["content"]
            
            return {
                "status": "success",
                "message": f"Search results for '{query}': {result}",
                "result": result,
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to interpret image: {str(e)}",
            }
         
