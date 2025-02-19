import asyncio
from generative_ai import groq_response
from pydantic import BaseModel, field_validator
from models import PromptResponseModel

IDEATION_PROMPT = """
You are a Content Strategist. Here's what you do,
- You generate creative and engaging ideas for YouTube Shorts content.
- You brainstorm and expand content ideas. 
- You analyze content quality and whether your content reflects what's most important to your audience. Then, use that information to inform which direction you take next.
- Create Creative and Newsworthy content.
- Come up with the idea of; what should be the Audience Hook?

Utilizing your ideas, you produce text that provides essential information for a scriptwriter to grasp topics, concepts, and ideas. This aids the scriptwriter in creating a compelling narration script.

Now, you are provided with initial idea and with target audience persona. Take a deep breath and show your magic as a Content Strategist.
"""
IDEATION_USER_PROMPT = '''INITITAL IDEA: {initial_idea}
TARGET AUDIENCE PERSONA:{target_audience_persona}'''

class Ideation:
    def __init__(
        self, initial_idea: str, target_audience_persona: list[str]
    ) -> "Ideation":
        self.initial_idea = initial_idea
        self.target_audience_persona = target_audience_persona

    def prepare_prompt(self) -> tuple[str, str]:
        return (
            IDEATION_PROMPT,
            IDEATION_USER_PROMPT.format(
                initial_idea=self.initial_idea,
                target_audience_persona=",".join(self.target_audience_persona),
            ),
        )

    def generate_content(self) -> PromptResponseModel:
        sys_prompt, user_prompt = self.prepare_prompt()
        generate_content_response = groq_response(sys_prompt, user_prompt)
        return PromptResponseModel(
            prompt_type=self.__class__.__name__,
            func_input_kwargs={
                "initial_idea": self.initial_idea,
                "target_audience_persona": ",".join(self.target_audience_persona),
            },
            generate_content_response=generate_content_response,
            prompt_used=IDEATION_PROMPT,
        )
