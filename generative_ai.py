from functools import cache
import replicate
from groq import AsyncGroq, Groq
import logging
import replicate
import json
import nltk  # we'll use this to split into sentences
import numpy as np

from bark.generation import (
    generate_text_semantic,
    preload_models,
)
from bark.api import semantic_to_waveform
from bark import generate_audio, SAMPLE_RATE
from transformers import BarkModel
from transformers import AutoProcessor
import torch
import scipy
import os, uuid, tempfile, requests
from models import ImageDataList, ImagePromptExtractionResult
from dotenv import load_dotenv

load_dotenv(".env")
client = Groq(api_key = )

@cache
def groq_response(sys_prompt: str, user_prompt: str) -> str:
    logging.info("Initialising groq....")

    chat_completion = client.chat.completions.create(
        #
        # Required parameters
        #
        messages=[
            # Set an optional system message. This sets the behavior of the
            # assistant and can be used to provide specific instructions for
            # how it should behave throughout the conversation.
            {
                "role": "system",
                "content": sys_prompt
            },
            # Set a user message for the assistant to respond to.
            {
                "role": "user",
                "content": user_prompt,
            }
        ],

        # The language model which will generate the completion.
        model="llama-3.1-70b-versatile",
        temperature=0.5,
        max_tokens=4096,
        # Controls diversity via nucleus sampling: 0.5 means half of all
        # likelihood-weighted options are considered.
        top_p=1,
    )

    # Print the completion returned by the LLM.
    # print(chat_completion.choices[0].message.content)
    
    return chat_completion.choices[0].message.content


def groq_json_response(content: str) -> ImagePromptExtractionResult:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": """You are a Script Writer for YouTube shorts. You generate 30 seconds to 1 minute of narration. The shorts you create have a background that fades from image to image as the narration is going on.
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
                    """
                    f"The JSON object must use the schema: {json.dumps(ImagePromptExtractionResult.model_json_schema(), indent=2)}",
                },
                {
                    "role": "user",
                    "content": f"SOURCE MATERIAL:{content}.",
                },
            ],
            model="llama3-8b-8192",
            temperature=0.7,
            stream=False,
            response_format={"type": "json_object"},
        )
        print(chat_completion.choices[0].message.content)
        return ImagePromptExtractionResult.model_validate_json(chat_completion.choices[0].message.content)

res = groq_json_response("What a fantastic initial idea! As a Content Strategist, I'll expand on this concept, brainstorm new ideas, and create a compelling narrative for a comical video on Iron Man's origin story.\n\n**Content Title:** \"Tony Stark's Epic Fail: The Iron Man Origin Story (Sort Of)\"\n\n**Genre:** Comedic, Superhero, Origin Story\n\n**Target Audience:** Marvel fans of all ages\n\n**Key Objectives:**\n\n1. Entertain the audience with a humorous take on Iron Man's origin story.\n2. Introduce the character of Tony Stark and his transformation into Iron Man.\n3. Highlight the themes of innovation, perseverance, and self-discovery.\n\n**Storyline:**\n\nThe video opens with Tony Stark, a billionaire inventor, showcasing his latest creation: a high-tech toaster. However, things quickly go awry as the toaster explodes, covering Tony in a mess of bread and jam.\n\nAs Tony tries to clean up the mess, he receives a call from his business partner, Obadiah Stane, informing him that their company, Stark Industries, is in trouble. Tony's father, Howard Stark, had been working on a top-secret project before his death, and the company is struggling to complete it.\n\nTony decides to take matters into his own hands and embark on a journey to find the missing pieces of his father's project. Along the way, he gets kidnapped by a group of terrorists who demand that he build a powerful missile system for them.\n\nUsing his wit and resourcefulness, Tony pretends to work on the missile system while secretly building the first Iron Man suit. However, things don't go as planned, and the suit's maiden flight ends with Tony crashing into a wall.\n\nDespite the setbacks, Tony perseveres and eventually completes the Iron Man suit. In a comedic twist, he uses the suit to escape from the terrorists, but not before accidentally setting off the fire alarm and getting stuck in a ventilation shaft.\n\n**Audience Hook:** \"What if Tony Stark's origin story was more like a series of epic fails? Join us as we reimagine the birth of Iron Man in a hilarious and action-packed adventure!\"\n\n**Key Comedic Elements:**\n\n1. Tony's bumbling attempts to create innovative technology, resulting in explosions and chaos.\n2. The terrorists' misadventures, including their inability to understand Tony's sarcasm and their own ineptitude.\n3. The Iron Man suit's malfunctioning, leading to Tony getting stuck in awkward situations.\n\n**Tone:** Light-hearted, comedic, and entertaining, with a touch of satire and self-aware humor.\n\n**Visuals:**\n\n1. Fast-paced editing to emphasize the comedic moments.\n2. Exaggerated special effects to highlight the absurdity of the situations.\n3. A mix of close-up shots and wide angles to create a sense of chaos and confusion.\n\n**Scriptwriter's Notes:**\n\n* Emphasize Tony's wit and sarcasm throughout the script.\n* Use comedic timing to deliver punchlines and humorous moments.\n* Balance action and comedy to keep the audience engaged.\n\nThis comical take on Iron Man's origin story is sure to entertain Marvel fans of all ages. With its lighthearted tone, humorous twists, and comedic elements, this video is poised to become a hilarious and memorable addition to the Marvel universe.")
print(res)

@cache
def generate_image(prompt: str) -> str:
    response = replicate.run(
                "stability-ai/stable-diffusion-3",
            input={
            "cfg": 6.5,
            "steps": 28,
            "prompt": prompt,
            "aspect_ratio": "9:16",
            "output_format": "jpg",
            "output_quality": 90,
            "negative_prompt": "",
            "prompt_strength": 0.85
        },
    )
    return response.data[0]

@cache
def generate_audio(narration:str, audio_path):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = BarkModel.from_pretrained("suno/bark-small")
    model = model.to(device)
    processor = AutoProcessor.from_pretrained("suno/bark")
    sampling_rate = model.generation_config.sample_rate
    inputs = processor(narration)
    speech_output = model.generate(**inputs.to(device))
    scipy.io.wavfile.write(audio_path, rate=sampling_rate, data=speech_output[0].cpu().numpy())
