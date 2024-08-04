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
from gradio_client import Client
import torch
import scipy
import shutil
import os, uuid, tempfile, requests
from models import ImageDataList, ImagePromptExtractionResult
from dotenv import load_dotenv

load_dotenv(".env")
client = Groq(api_key = "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")

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


def groq_json_response(sys_prompt: str, user_prompt:str) -> ImageDataList:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": sys_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            model="llama-3.1-70b-versatile",
            temperature=0,
            stream=False,
            response_format={"type": "json_object"},
        )
        print(chat_completion.choices[0].message.content)
        return ImageDataList.model_validate_json(chat_completion.choices[0].message.content)

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
    return response[0]

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


# def generate_audio(narration:str):
#     client = Client("mrfakename/MeloTTS")
#     result = client.predict(
#             text="The field of text-to-speech has seen rapid development recently.",
#             speaker="EN-US",
#             speed=1,
#             language="EN",
#             api_name="/synthesize"
#     )
#     print(type(result))
#     # if os.path.exists(result):
#     #     shutil.copy(result, audiopath)
#     #     print(f"Audio successfully copied to {audiopath}")
#     # else:
#     #     print(f"Failed to copy audio. Source file does not exist: {result}")

# res = generate_audio("Hi this is Likhith. how are you doing ?")
# print(res)
