
from typing import Any
from realtime_agent.tools import ToolContext
from ..utils import PCMWriter
import asyncio

# Function calling Example
# This is an example of how to add a new function to the agent tools.

class AgetnToolsMetaWorkplaces(ToolContext):
    def __init__(self) -> None:
        super().__init__()

        self.agent=None
        self.channel=None
        self.image_api=""
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
            description="Check the agent's state of being muted or unmuted, True for muted, False for unmuted", 
            parameters={
                "type": "object",
            },
            fn=self._set_mute_state,
        )
        
        
        
    def set_agent(self,agent):
        self.agent=agent
        self.channel=agent.channel

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
            
            
    async def _generate_image(self,country: str):
        try:
            result = "24 degree C" # Dummy data (Get the Required value here, like a DB call or API call)
            return {
                "status": "success",
                "message": f"Average temperature of {country} is {result}",
                "result": result,
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to get : {str(e)}",
            }