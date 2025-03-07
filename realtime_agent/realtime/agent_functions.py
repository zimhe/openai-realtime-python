
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

# Function calling Example
# This is an example of how to add a new function to the agent tools.

class AgetnToolsMetaWorkplaces(ToolContext):
    def __init__(self) -> None:
        super().__init__()

        self.agent=None
        self.channel=None
        self.image_api="black-forest-labs/flux-dev"
        self.screen_keys=None
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
            },
            fn=self._text2image,
        )
        
        
        
        
        
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
            return {
                "status": "success",
                "message": f"Agent is muted: {self.agent.muted}",
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
            
            output=await replicate.async_run(self.image_api,input,use_file_output=False)
            
            screen_key=None
            proper_screen_id=screen_id-1
            if screen_id!=-1 and self.screen_keys is not None and proper_screen_id<len(self.screen_keys):
                screen_key=self.screen_keys[proper_screen_id]
            
            result={"text_to_image_output":output[0],"screen_key":screen_key}
            
            msg_id=uuid.uuid4().hex
            
            chat_message=ChatMessage(message=json.dumps(result),msg_id=msg_id)
            
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
            
            
    async def _get_history_images(self)-> dict[str, Any]:
        try:
            return {
                "status": "success",
                "message": f"Get history images success",
                "result": self.agent.history_images,
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to get history images",
            }
            
    async def _cast_image_to_screen(self,image_key,screen_id):
        pass
        