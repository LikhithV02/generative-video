import replicate
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
from IPython.display import Audio
import os, uuid, tempfile, requests
from groq import Groq
from dotenv import load_dotenv

load_dotenv(".env")
OPENAI_API_KEY = os.environ.get("api_key")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
client = Groq(api_key=OPENAI_API_KEY)
model = BarkModel.from_pretrained("suno/bark-small")
model = model.to(device)

processor = AutoProcessor.from_pretrained("suno/bark")
sampling_rate = model.generation_config.sample_rate


IDEATION_PROMPT = """
You are a Content Strategist. Here's what you do,
- You generate creative and engaging ideas for YouTube Shorts content.
- You brainstorm and expand content ideas. 
- You analyze content quality and whether your content reflects what's most important to your audience. Then, use that information to inform which direction you take next.
- Create Creative and Newsworthy content.
- Come up with the idea of; what should be the Audience Hook?

Utilizing your ideas, you produce text that provides essential information for a scriptwriter to grasp topics, concepts, and ideas. This aids the scriptwriter in creating a compelling narration script.

Now, you are provided with initial idea and with target audience persona. Take a deep breath and show your magic as a Content Strategist.
INITITAL IDEA: {initial_idea}
TARGET AUDIENCE PERSONA:{target_audience_persona}
"""

NARRATION_PROMPT = """
You are a Script Writer for YouTube shorts. You generate 30 seconds to 1 minute of narration. The shorts you create have a background that fades from image to image as the narration is going on.
Your Goal is to generate a narration script which explains the topic in a story-telling way, make it engaging, audience should learn about the topic.

Below are rules to keep in mind for generating narration:
1. Generate as detailed description of image as possible, use the tips given in describing images.
2. When describing image you can add use following tips if required
    - TIP: Use camera settings such asmotion blur, soft focus, bokeh, portrait.
    - TIP: Use lens types such as35mm, 50mm, fisheye, wide angle, macro.
    - TIP: Use quality modifiers such as4K, HDR, beautiful, by a professional.
    - TIP: Use camera proximity such asclose up, zoomed out.
    - TIP: Use lighting and shadow details.
3. Avoid using names of celebrities or people in image descriptions; it's illegal to generate images of celebrities.
4. Describe individuals without using their names; do not reference any real person or group.
5. Exclude any mention of the female figure or sexual content in image descriptions.
6. Allowed to use any content, including names, in the narration.
7. Narration will be fed into a text-to-speech engine, so avoid using special characters.

Respond in JSON list with a pair of a detailed image description and a narration. Maximum 6 pairs should be generated. Both of them should be on their own lines, as follows:
Example:
[
    {{"image_description":"A Detailed Description of a background image", "narration":"One sentence of narration"}},
    {{"image_description":"A Detailed Description of a background image", "narration":"One sentence of narration"}}
]

Create a YouTube short narration based on the following source material created by Content Strategist and only output the JSON list. Don't forget to be creative. Take a deep breathe and be creative.

SOURCE MATERIAL:
{source_material}
"""

# config_list = autogen.config_list_from_json(env_or_file="OAI_CONFIG_LIST.json")
# llm_config = {"config_list": config_list, "cache_seed": 42}

# chatbot = autogen.AssistantAgent(
#     name="chatbot",
#     system_message="For currency exchange tasks, only use the functions you have been provided with. Reply TERMINATE when the task is done.",
#     llm_config=llm_config,
# )

# # create a UserProxyAgent instance named "user_proxy"
# user_proxy = autogen.UserProxyAgent(
#     name="user_proxy",
#     is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
#     human_input_mode="NEVER",
#     max_consecutive_auto_reply=10,
# )



# @user_proxy.register_for_execution()
# @chatbot.register_for_llm(description="Generate image and returns image file path")

