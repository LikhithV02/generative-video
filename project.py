import pathlib
import json
import ast
import requests
import logging
from models import ProjectRequest, PromptResponseModel, ImageDataList, PromptJsonResponseModel
from ideation import Ideation
from narration import Narration
from generative_ai import generate_image, generate_audio
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
from pydantic import BaseModel, field_validator
from typing import List

class ImageData(BaseModel):
    image_description: str
    narration: str

class ImageDataList(BaseModel):
    items: List[ImageData]
class Project:
    def __init__(self, project_request: ProjectRequest) -> None:
        self.project_request = project_request
        self._setup()

    def _setup(self):
        (pathlib.Path("projects") / self.project_request.project_name).mkdir(
            exist_ok=True, parents=True
        )

    def generate_idea(self):
        ideation_response = Ideation(
            initial_idea=self.project_request.initial_idea,
            target_audience_persona=self.project_request.target_audience_persona,
        ).generate_content()
        pathlib.Path(
            f"projects/{self.project_request.project_name}/ideation.json"
        ).write_text(ideation_response.model_dump_json())
        logging.info("Ideation phase complete")
        return self

    def generate_script(self):
        ideation_response = PromptResponseModel.model_validate_json(
            pathlib.Path(
                f"projects/{self.project_request.project_name}/ideation.json"
            ).read_text()
        )
        naration_response = Narration(
            source_material=ideation_response.generate_content_response
        ).generate_content()
        pathlib.Path(
            f"projects/{self.project_request.project_name}/narration.json"
        ).write_text(naration_response.model_dump_json())
        logging.info("Narration phase complete")
        return self

    def generate_image_files(self):
        narration_response = PromptJsonResponseModel.model_validate_json(
            pathlib.Path(
                f"projects/{self.project_request.project_name}/narration.json"
            ).read_text()
        )
        scenes = narration_response.generate_content_response
        print("Type: ", type(scenes))
        print("Actual scenes", scenes)
        # Extract the list part from the string
        # list_string = narration_response.generate_content_response.split('=', 1)[1]
        # print(type(list_string))
        # print(list_string)
        # if type(list_string) == str:
        #     data = ast.literal_eval(list_string)
        # else:
        # data = json.loads(list_string)
            
        # # Use ast.literal_eval to safely evaluate the string as a Python expression
        # data = ast.literal_eval(list_string)

        # # Convert to the desired format
        # scenes = [{"image_description": item.image_description, "narration": item.narration} for item in data]
        image_folder = pathlib.Path(
            f"projects/{self.project_request.project_name}/images"
        )
        image_folder.mkdir(exist_ok=True, parents=True)
        for i, scene in enumerate(scenes.items, start=1):
            image_url = generate_image(scene.image_description)
            # Send a GET request to the URL
            response = requests.get(image_url)
            image_path = image_folder / f"{i}.jpg"
            # Check if the request was successful
            if response.status_code == 200:
                # Open a file in write-binary mode
                with open(image_path, "wb") as file:
                    # Write the contents of the response (the image) to the file
                    file.write(response.content)
                logging.info(f'Image content written to file "{image_path}"')
                # print("Image successfully downloaded: downloaded_image.jpg")
            else:
                logging.info("Failed to download image. Status code:", response.status_code)
        return self

    def generate_audio_files(self):
        narration_response = PromptJsonResponseModel.model_validate_json(
            pathlib.Path(
                f"projects/{self.project_request.project_name}/narration.json"
            ).read_text()
        )
        scenes = narration_response.generate_content_response
        audio_folder = pathlib.Path(
            f"projects/{self.project_request.project_name}/audio"
        )
        audio_folder.mkdir(exist_ok=True, parents=True)
        for i, scene in enumerate(scenes.items, start=1):
            audio_path = audio_folder / f"{i}.wav"
            generate_audio(scene.narration, audio_path)
            logging.info(f'Audio content written to file "{audio_path}"')

    def generate_video(self):
        clips = []
        for i in range(1, 7):
            audio_clip = AudioFileClip(
                f"projects/{self.project_request.project_name}/audio/{i}.wav"
            )
            image_clip = ImageClip(
                f"projects/{self.project_request.project_name}/images/{i}.jpg"
            ).set_duration(audio_clip.duration + 2)
            image_clip = image_clip.set_audio(audio_clip)
            clips.append(image_clip.crossfadein(2))
        concat_clip = concatenate_videoclips(clips, method="compose", padding=-2)
        concat_clip.write_videofile(
            f"projects/{self.project_request.project_name}/{self.project_request.project_name}.mp4",
            fps=30,
        )
        return f"projects/{self.project_request.project_name}/{self.project_request.project_name}.mp4"
