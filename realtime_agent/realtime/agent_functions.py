from typing import Any
from datetime import datetime, timezone
from urllib.parse import quote, urlparse
import hashlib
import hmac

from realtime_agent.tools import ToolContext

from agora_realtime_ai_api.rtc import Channel, ChatMessage, RtcEngine, RtcOptions
from .struct import to_json
from ..utils import PCMWriter
import asyncio
import uuid
import json
import aiohttp
import os
import logging
from contextvars import ContextVar
from openai import OpenAI

ark_image_url = os.environ.get("ARK_IMAGE_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
ark_image_model_id = os.environ.get("ARK_IMAGE_MODEL_ID", "doubao-seedream-5-0-lite-260128")
ark_image_size = os.environ.get("ARK_IMAGE_SIZE", "2K")
ark_api_key = os.environ.get("ARK_API_KEY")
visual_base_url = os.environ.get("VOLCENGINE_VISUAL_BASE_URL", "https://visual.volcengineapi.com")
visual_region = os.environ.get("VOLCENGINE_VISUAL_REGION", "cn-north-1")
visual_service = os.environ.get("VOLCENGINE_VISUAL_SERVICE", "cv")
visual_submit_action = "CVSync2AsyncSubmitTask"
visual_get_result_action = "CVSync2AsyncGetResult"
visual_api_version = "2022-08-31"

T2I_OUTPUT="text2image.output"
logger = logging.getLogger(__name__)

class AgentToolsMetaWorkplaces(ToolContext):
    def __init__(self) -> None:
        super().__init__()

        self.agent=None
        self.channel=None
        self.screen_keys=None
        self.selected_screen_by_uid={}
        self.request_screen_key: ContextVar[str | None] = ContextVar("request_screen_key", default=None)
        self.history_images={}
        self.pending_image_tasks: set[asyncio.Task] = set()
        self.t2i_condition_queue=asyncio.Queue()
        

        self.register_function(
            name="text_to_image",
            description='''generate image from natural language prompt, 
            the prompt should be refined with delicate details for better quality.
            if user specifies a screen number, the function should also specify a <screen_id> in the args,
            the screen_id should be specified as a number, use 1, 2, 3... for the screen index, or -2 for the selected screen''', 
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "the text prompt for generating image",
                    },
                    "screen_id": {
                        "type": "integer",
                        "description": "Optional, the screen index to cast the image to if user wants to put the image to a specified screen, use numeric index if user specified a numer, or use <-2> if user says put to the 'selected screen'",
                    }
                },
                "required":["prompt"]
            },
            fn=self._text2image,
        )
        
        self.register_function(
            name="refine_this_image",
            description='''refine this image with natural language prompts.
            the prompt should be refined with delicate details for better quality.
            if user specifies a screen number, the function should also specify a <screen_id> in the args,
            the screen_id should be specified as a number, use 1, 2, 3... for the screen index, or -2 for the selected screen
            ''', 
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
                        "description": "Optional, the screen index to cast the image to if user wants to put the image to a specified screen, use numeric index if user specified a numer, or use <-2> if user says put to the 'selected screen'",
                    }
                },
                "required":["prompt","input_image"]
            },
            fn=self._refine_this_image,
        )
        
        self.register_function(
            name="refine_masked_area_of_image",
            description='''re-generate the masked area in the given image from text prompt and given condition image, 
            the prompt should be refined with delicate details for better quality.
            if user specifies a screen number, the function should also specify a <screen_id> in the args,
            the screen_id should be specified as a number, use 1, 2, 3... for the screen index, or -2 for the selected screen
            ''', 
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
                        "description": "Optional, the screen index to cast the image to if user wants to put the image to a specified screen, use numeric index if user specified a numer, or use <-2> if user says put to the 'selected screen'",
                    }
                },
                "required":["prompt","image_url","mask_url"]
            },
            fn=self._refine_image_with_mask,
        )
        
        self.register_function(
            name="get_history_image_info",
            description="get the history images info with json format, can be used to when user queries the previous generation records, the function will return a json object with the history images info, including the message id, the prompt and image url",
            parameters={
                "type": "object",
                "properties": {},
            },
            fn=self._get_history_images_data,
        )
        
        self.register_function(
            name="cast_image_to_screen",
            description="cast the image to a specific screen, the image url should be a valid url, and the screen_id should be specified, numbers 1, 2, 3... for the screen index, or -2 for the selected screen",
            parameters={
                "type":"object",
                "properties":{
                    "image_url":{
                        "type":"string",
                        "description":"the image url that needs to be casted"
                    },
                    "screen_id": {
                        "type": "integer",
                        "description": "the screen index to cast the image to if user wants to put the image to a specified screen, use numeric index if user specified a numer, or use <-2> if user says put to the 'selected screen'",
                    }
                },
                "required":["image_url","screen_id"]
            },
            fn=self._cast_image_to_screen,
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
        self.screen_keys=sorted(set(screen_keys or []))

    def set_selected_screen(self, uid: int, key: str | None):
        if key and key in (self.screen_keys or []):
            self.selected_screen_by_uid[uid] = key
        else:
            self.selected_screen_by_uid.pop(uid, None)

    def resolve_screen_key(self, screen_id: int) -> str | None:
        if screen_id == -1:
            return None
        if screen_id == -2:
            key = self.request_screen_key.get()
            if not key and self.agent is not None:
                key = self.selected_screen_by_uid.get(getattr(self.agent, "duplex_active_uid", None))
            if not key or key not in (self.screen_keys or []):
                raise ValueError("No valid selected screen is registered for the requesting user")
            return key
        if screen_id < 1 or screen_id > len(self.screen_keys or []):
            raise ValueError(f"Screen {screen_id} is not registered")
        return self.screen_keys[screen_id - 1]
        
    
    def format_t2i_result(self, output, screen_key,item_id):
        result={"type":T2I_OUTPUT,"output":output,"screen_key":screen_key,"item_id":item_id}
        return result
    
    def append_history_image(self,msg_id:str,prompt:str,image_url:str):
        self.history_images[msg_id]={"prompt": prompt, "image_url": image_url}
        

    async def _text2image(self, prompt: str, screen_id: int = -1) -> dict[str, Any]:
        return self._schedule_image_job("generate", prompt, screen_id)

    async def _refine_this_image(self, prompt: str, input_image: str, screen_id: int = -1) -> dict[str, Any]:
        return self._schedule_image_job("refine", prompt, screen_id, input_image=input_image)

    async def _refine_image_with_mask(self, prompt: str, image_url: str, mask_url: str,
                                      screen_id: int = -1) -> dict[str, Any]:
        return self._schedule_image_job("inpaint", prompt, screen_id, input_image=image_url, mask_url=mask_url)

    def _schedule_image_job(self, kind: str, prompt: str, screen_id: int,
                            input_image: str | None = None, mask_url: str | None = None) -> dict[str, Any]:
        try:
            if not prompt or not prompt.strip():
                raise ValueError("Image prompt is empty")
            if self.channel is None:
                raise RuntimeError("Agent channel is not connected")
            if kind != "inpaint" and not ark_api_key:
                raise RuntimeError("ARK_API_KEY is required for Seedream image generation")
            # Resolve while the request's screen selection and speaker context are still available.
            screen_key = self.resolve_screen_key(screen_id)
            task_id = uuid.uuid4().hex
            task = asyncio.create_task(self._run_image_job(
                task_id, kind, prompt, screen_key, input_image, mask_url))
            self.pending_image_tasks.add(task)
            task.add_done_callback(self.pending_image_tasks.discard)
            logger.info("Image task queued: id=%s kind=%s screen=%s", task_id, kind, screen_key)
            return {
                "status": "queued",
                "task_id": task_id,
                "message": "Image generation has been scheduled. Tell the user it will appear in the conversation when ready; do not claim it is finished.",
            }
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    async def _run_image_job(self, task_id: str, kind: str, prompt: str,
                             screen_key: str | None, input_image: str | None,
                             mask_url: str | None) -> None:
        try:
            if kind == "generate":
                output = await self._generate_seedream_image(prompt=prompt)
            elif kind == "refine":
                output = await self._generate_seedream_image(prompt=prompt, image=input_image)
            else:
                output = await self._generate_visual_inpaint_image(
                    prompt=prompt, image_url=input_image, mask_url=mask_url)
            if not output:
                raise RuntimeError("Image service returned no image URL")
            result = self.format_t2i_result(output, screen_key, task_id)
            await self.channel.chat.send_message(
                ChatMessage(message=json.dumps(result), msg_id=task_id))
            self.append_history_image(task_id, prompt, output)
            logger.info("Image task completed: id=%s kind=%s", task_id, kind)
        except asyncio.CancelledError:
            logger.info("Image task cancelled: id=%s", task_id)
            raise
        except Exception as exc:
            logger.exception("Image task failed: id=%s kind=%s", task_id, kind)
            failure = {
                "type": "workspace.agent.text.response",
                "item_id": task_id,
                "text": "Image generation failed. Please try again.",
            }
            try:
                await self.channel.chat.send_message(
                    ChatMessage(message=json.dumps(failure), msg_id=task_id))
            except Exception:
                logger.exception("Could not notify clients that image task %s failed", task_id)

    async def cancel_image_tasks(self) -> None:
        tasks = tuple(self.pending_image_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _generate_seedream_image(self, *, prompt: str, image: str | None = None) -> str | None:
        if not ark_api_key:
            raise RuntimeError("ARK_API_KEY must be set for Seedream image generation")

        client = OpenAI(api_key=ark_api_key, base_url=ark_image_url)
        extra_body: dict[str, Any] = {
            "sequential_image_generation": "disabled",
            "watermark": False,
        }
        if image:
            extra_body["image"] = image

        response = await asyncio.to_thread(
            client.images.generate,
            model=ark_image_model_id,
            prompt=prompt,
            size=ark_image_size,
            response_format="url",
            extra_body=extra_body,
        )

        generated = getattr(response, "data", None) or []
        for item in generated:
            url = getattr(item, "url", None)
            if url:
                return url
        return None

    async def _generate_visual_inpaint_image(self, *, prompt: str, image_url: str, mask_url: str) -> str | None:
        task_payload = {
            "req_key": "i2i_inpainting_edit",
            "image_urls": [image_url, mask_url],
            "custom_prompt": prompt,
            "steps": int(os.environ.get("VOLCENGINE_VISUAL_INPAINT_STEPS", "25")),
            "scale": float(os.environ.get("VOLCENGINE_VISUAL_INPAINT_SCALE", "5")),
            "seed": int(os.environ.get("VOLCENGINE_VISUAL_INPAINT_SEED", "-1")),
        }
        submit_response = await self._visual_signed_post(visual_submit_action, task_payload)
        if not self._is_visual_success_code(submit_response.get("code")):
            raise RuntimeError(submit_response.get("message") or "Volcengine visual inpaint submit failed")

        task_id = str((submit_response.get("data") or {}).get("task_id") or "").strip()
        if not task_id:
            raise RuntimeError("Volcengine visual inpaint submit did not return task_id")

        max_attempts = int(os.environ.get("VOLCENGINE_VISUAL_INPAINT_MAX_ATTEMPTS", "30"))
        poll_interval_seconds = float(os.environ.get("VOLCENGINE_VISUAL_INPAINT_POLL_SECONDS", "2"))

        for attempt in range(max_attempts):
            result_response = await self._visual_signed_post(
                visual_get_result_action,
                {"task_id": task_id},
            )
            if not self._is_visual_success_code(result_response.get("code")):
                raise RuntimeError(result_response.get("message") or "Volcengine visual inpaint polling failed")

            data = result_response.get("data") or {}
            extracted_url = self._extract_first_image_url(data)
            if extracted_url:
                return extracted_url

            status = str(
                data.get("status")
                or data.get("task_status")
                or data.get("state")
                or data.get("phase")
                or ""
            ).strip().lower()

            if status in {"failed", "error", "timeout", "canceled", "cancelled"}:
                raise RuntimeError(data.get("message") or data.get("error") or f"Volcengine visual inpaint task failed: {status}")

            if attempt == max_attempts - 1:
                break

            await asyncio.sleep(poll_interval_seconds)

        raise RuntimeError("Volcengine visual inpaint task did not finish before timeout")

    async def _visual_signed_post(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        access_key_id, secret_access_key, security_token = self._get_visual_credentials()

        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        content_sha256 = hashlib.sha256(body).hexdigest()

        endpoint = urlparse(visual_base_url)
        host = endpoint.netloc
        path = endpoint.path or "/"
        timestamp = datetime.now(timezone.utc)
        x_date = timestamp.strftime("%Y%m%dT%H%M%SZ")
        short_date = x_date[:8]

        query = {
            "Action": action,
            "Version": visual_api_version,
        }
        canonical_query = "&".join(
            f"{quote(key, safe='-_.~')}={quote(value, safe='-_.~')}"
            for key, value in sorted(query.items())
        )
        signed_headers = "content-type;host;x-content-sha256;x-date"
        canonical_headers = (
            f"content-type:application/json\n"
            f"host:{host}\n"
            f"x-content-sha256:{content_sha256}\n"
            f"x-date:{x_date}\n"
        )
        canonical_request = (
            f"POST\n"
            f"{path}\n"
            f"{canonical_query}\n"
            f"{canonical_headers}\n"
            f"{signed_headers}\n"
            f"{content_sha256}"
        )
        credential_scope = f"{short_date}/{visual_region}/{visual_service}/request"
        string_to_sign = (
            f"HMAC-SHA256\n"
            f"{x_date}\n"
            f"{credential_scope}\n"
            f"{hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()}"
        )

        signing_key = self._sign_hmac(
            self._sign_hmac(
                self._sign_hmac(
                    self._sign_hmac(secret_access_key.encode("utf-8"), short_date),
                    visual_region,
                ),
                visual_service,
            ),
            "request",
        )
        signature = hmac.new(signing_key, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

        headers = {
            "Content-Type": "application/json",
            "Host": host,
            "X-Date": x_date,
            "X-Content-Sha256": content_sha256,
            "Authorization": (
                f"HMAC-SHA256 Credential={access_key_id}/{credential_scope}, "
                f"SignedHeaders={signed_headers}, Signature={signature}"
            ),
        }
        if security_token:
            headers["X-Security-Token"] = security_token

        request_url = f"{visual_base_url}?{canonical_query}"
        async with aiohttp.ClientSession() as session:
            async with session.post(
                request_url,
                data=body,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as response:
                response.raise_for_status()
                return await response.json()

    def _get_visual_credentials(self) -> tuple[str, str, str | None]:
        access_key_id = (
            os.environ.get("VOLCENGINE_VISUAL_ACCESS_KEY_ID")
            or os.environ.get("VOLCENGINE_ACCESS_KEY_ID")
            or ""
        ).strip()
        secret_access_key = (
            os.environ.get("VOLCENGINE_VISUAL_SECRET_ACCESS_KEY")
            or os.environ.get("VOLCENGINE_SECRET_ACCESS_KEY")
            or ""
        ).strip()
        security_token = (
            os.environ.get("VOLCENGINE_VISUAL_SESSION_TOKEN")
            or os.environ.get("VOLCENGINE_SESSION_TOKEN")
            or None
        )

        if not access_key_id or not secret_access_key:
            raise RuntimeError(
                "VOLCENGINE_VISUAL_ACCESS_KEY_ID and VOLCENGINE_VISUAL_SECRET_ACCESS_KEY must be set for visual inpaint"
            )

        return access_key_id, secret_access_key, security_token

    def _sign_hmac(self, key: bytes, message: str) -> bytes:
        return hmac.new(key, message.encode("utf-8"), hashlib.sha256).digest()

    def _is_visual_success_code(self, code: Any) -> bool:
        return str(code).strip() == "10000"

    def _extract_first_image_url(self, data: Any) -> str | None:
        if isinstance(data, dict):
            direct_url = data.get("url") or data.get("image_url")
            if isinstance(direct_url, str) and direct_url:
                return direct_url

            image_urls = data.get("image_urls") or data.get("urls") or data.get("result")
            if isinstance(image_urls, list):
                for item in image_urls:
                    extracted = self._extract_first_image_url(item)
                    if extracted:
                        return extracted
            elif isinstance(image_urls, str) and image_urls:
                return image_urls

            for value in data.values():
                extracted = self._extract_first_image_url(value)
                if extracted:
                    return extracted
            return None

        if isinstance(data, list):
            for item in data:
                extracted = self._extract_first_image_url(item)
                if extracted:
                    return extracted
            return None

        if isinstance(data, str) and data.startswith(("http://", "https://")):
            return data

        return None
    
            
    async def _get_history_images_data(self)-> dict[str, Any]:
        try:
            history_info=json.dumps(self.history_images)
            return {
                "status": "success",
                "message": f"Get history images info success",
                "result": history_info,
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to get history images {e}",
            }
            
            
    async def _cast_image_to_screen(self,image_url,screen_id=1)-> dict[str, Any]:
        try:
            screen_key=self.resolve_screen_key(screen_id)
            
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
