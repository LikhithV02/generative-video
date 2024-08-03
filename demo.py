# from dotenv import load_dotenv
# import os
# load_dotenv(".env")
# from groq import Groq
# # GROQ_API_KEY = os.environ.get("api_key")
# # print(GROQ_API_KEY)
# client = Groq(api_key = "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")

# chat_completion = client.chat.completions.create(
#     #
#     # Required parameters
#     #
#     messages=[
#         # Set an optional system message. This sets the behavior of the
#         # assistant and can be used to provide specific instructions for
#         # how it should behave throughout the conversation.
#         {
#             "role": "system",
#             "content": "you are a helpful assistant."
#         },
#         # Set a user message for the assistant to respond to.
#         {
#             "role": "user",
#             "content": "Explain the importance of fast language models",
#         }
#     ],

#     # The language model which will generate the completion.
#     model="llama-3.1-8b-instant",

#     #
#     # Optional parameters
#     #

#     # Controls randomness: lowering results in less random completions.
#     # As the temperature approaches zero, the model will become deterministic
#     # and repetitive.
#     temperature=0.5,

#     # The maximum number of tokens to generate. Requests can use up to
#     # 32,768 tokens shared between prompt and completion.
#     max_tokens=1024,

#     # Controls diversity via nucleus sampling: 0.5 means half of all
#     # likelihood-weighted options are considered.
#     top_p=1,

#     # A stop sequence is a predefined or user-specified text string that
#     # signals an AI to stop generating content, ensuring its responses
#     # remain focused and concise. Examples include punctuation marks and
#     # markers like "[end]".
#     stop=None,

#     # If set, partial message deltas will be sent.
#     stream=False,
# )

# # Print the completion returned by the LLM.
# print(chat_completion.choices[0].message.content)

# from gradio_client import Client

# client = Client("PixArt-alpha/PixArt-Sigma")
# result = client.predict(
# 		prompt="goat flying in sky",
# 		negative_prompt="",
# 		style="(No style)",
# 		use_negative_prompt=False,
# 		num_imgs=1,
# 		seed=0,
# 		width=1024,
# 		height=1024,
# 		schedule="DPM-Solver",
# 		dpms_guidance_scale=4.5,
# 		sas_guidance_scale=3,
# 		dpms_inference_steps=14,
# 		sas_inference_steps=25,
# 		randomize_seed=True,
# 		api_name="/run"
# )
# print(result)

from typing import List, Optional
import json

from pydantic import BaseModel
from groq import Groq

groq = Groq(api_key = "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")


# Data model for LLM to generate
class Ingredient(BaseModel):
    name: str
    quantity: str
    quantity_unit: Optional[str]


class Recipe(BaseModel):
    recipe_name: str
    ingredients: List[Ingredient]
    directions: List[str]


def get_recipe(recipe_name: str) -> Recipe:
    chat_completion = groq.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a recipe database that outputs recipes in JSON.\n"
                # Pass the json schema to the model. Pretty printing improves results.
                f" The JSON object must use the schema: {json.dumps(Recipe.model_json_schema(), indent=2)}",
            },
            {
                "role": "user",
                "content": f"Fetch a recipe for {recipe_name}",
            },
        ],
        model="llama3-8b-8192",
        temperature=0,
        # Streaming is not supported in JSON mode
        stream=False,
        # Enable JSON mode by setting the response format
        response_format={"type": "json_object"},
    )
    return Recipe.model_validate_json(chat_completion.choices[0].message.content)


def print_recipe(recipe: Recipe):
    print("Recipe:", recipe.recipe_name)

    print("\nIngredients:")
    for ingredient in recipe.ingredients:
        print(
            f"- {ingredient.name}: {ingredient.quantity} {ingredient.quantity_unit or ''}"
        )
    print("\nDirections:")
    for step, direction in enumerate(recipe.directions, start=1):
        print(f"{step}. {direction}")


recipe = get_recipe("apple pie")
print_recipe(recipe)
