from generative_ai import groq_json_response
from pydantic import BaseModel, field_validator
from typing import List
import json
from models import ImageDataList, PromptResponseModel

NARRATION_PROMPT = ""

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

    def generate_content(self) -> PromptResponseModel:
        sys_prompt, user_prompt = self.prepare_prompt()
        generate_content_response = groq_json_response(sys_prompt, user_prompt)
        print(generate_content_response)
        return PromptResponseModel(
            prompt_type=self.__class__.__name__,
            func_input_kwargs={"source_material": self.source_material},
            generate_content_response=generate_content_response,
            prompt_used=NARRATION_PROMPT,
        )
