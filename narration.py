from generative_ai import groq_json_response
from pydantic import BaseModel, field_validator
from typing import List
import json
from models import ImageDataList, PromptResponseModel, PromptJsonResponseModel

NARRATION_PROMPT = f"""You are a Script Writer for YouTube shorts. You generate 30 seconds to 1 minute of narration. The shorts you create have a background that fades from image to image as the narration is going on.
                    Your Goal is to generate a narration script which explains the topic in a story-telling way, make it engaging, audience should learn about the topic.
                    Below are rules to keep in mind for generating narration:
                    1. Generate as detailed description of image as possible, use the tips given in describing images.
                    2. When describing image you can add use following tips if required
                        - TIP: Use camera settings such asmotion blur, soft focus, bokeh, portrait.
                        - TIP: Use lens types such as35mm, 50mm, fisheye, wide angle, macro.
                        - TIP: Use quality modifiers such as 4K, HDR, beautiful, by a professional.
                        - TIP: Use camera proximity such asclose up, zoomed out.
                        - TIP: Use lighting and shadow details.
                    3. Avoid using names of celebrities or people in image descriptions; it's illegal to generate images of celebrities.
                    4. Describe individuals without using their names; do not reference any real person or group.
                    5. Exclude any mention of the female figure or sexual content in image descriptions.
                    6. Allowed to use any content, including names, in the narration.
                    7. Narration will be fed into a text-to-speech engine, so avoid using special characters.
                    Create a YouTube short narration based on the following source material created by Content Strategist and only output the JSON. Don't forget to be creative. Take a deep breathe and be creative.
                    The JSON object must use the schema: {json.dumps(ImageDataList.model_json_schema(), indent=2)}
                    """

NARRATION_USER_PROMPT = '''SOURCE MATERIAL:
{source_material}'''

class Narration:
    def __init__(self, source_material: str) -> "Narration":
        self.source_material = source_material

    def prepare_prompt(self) -> tuple[str, str]:
        return (
            NARRATION_PROMPT,
            NARRATION_USER_PROMPT.format(source_material=self.source_material)
        )

    def generate_content(self) -> PromptJsonResponseModel:
        sys_prompt, user_prompt = self.prepare_prompt()
        generate_content_response = groq_json_response(sys_prompt, user_prompt)
        # print(generate_content_response)
        return PromptJsonResponseModel(
            prompt_type=self.__class__.__name__,
            func_input_kwargs={"source_material": self.source_material},
            generate_content_response=generate_content_response,
            prompt_used=NARRATION_PROMPT,
        )
