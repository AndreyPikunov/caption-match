import json
import requests

import numpy as np
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

from caption_match.config import settings


class GPTResponse(BaseModel):
    match: str
    description: str


class RerankRow(BaseModel):
    match: str
    description: str
    score: float


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Reranker:
    def __init__(self):
        self.url = settings.reranker.openai_url
        self.__headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.openai_api_key}",
        }
        self.reranker = CrossEncoder(settings.reranker.model_name)

    def _create_prompt(self, caption: str) -> str:
        prompt = (
            "Assess if the caption aligns with the images. "
            "For every image collect matching entities and attributes from the caption. "
            "Respond with empty string if no matches at all. "
            "Add short descriptions for each image (just one sentence). "
            "For instance, if the caption is 'A cat on a red sofa with pillows' "
            "and three images are: "
            "1. a dog on a red sofa with pillows; "
            "2. a cat on a sofa without pillows; "
            "3. a fish in a bowl; "
            'reply {"response": [\n'
            '    {"match": "red sofa with pillows", "description": "<YOUR DESCRIPTION OF IMAGE1>"}, \n'
            '    {"match": "cat on a sofa", "description": "<YOUR DESCRIPTION OF IMAGE2>"}, \n'
            '    {"match": "cat on a sofa", "description": "<YOUR DESCRIPTION OF IMAGE3>"}, \n'
            "]}.\n\n"
            "So, the output should be a JSON array with a string for each image.\n\n"
            f"Here's the caption: '{caption}'."
        )
        return prompt

    def _parse_response(self, response: dict) -> list[GPTResponse]:
        content = response["choices"][0]["message"]["content"]
        rows = [GPTResponse(**x) for x in json.loads(content)["response"]]
        return rows

    def _create_payload_content(
        self, caption: str, images_base64: list[str]
    ) -> list[dict]:

        prompt = self._create_prompt(caption)
        content = [{"type": "text", "text": prompt}]

        for img in images_base64:
            item = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img}",
                    "detail": "low",
                },
            }
            content.append(item)

        return content

    def rerank(self, caption: str, images_base64: list[str]) -> list[RerankRow]:

        content = self._create_payload_content(caption, images_base64)

        payload = {
            "model": "gpt-4-turbo",
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant designed to output JSON. "
                        "Format of your output is described in the prompt."
                    ),
                },
                {
                    "role": "user",
                    "content": content,
                },
            ],
            "max_tokens": 300,
            "temperature": 0,
            "seed": 42,
        }

        response = requests.post(self.url, json=payload, headers=self.__headers)
        response.raise_for_status()
        response_rows = self._parse_response(response.json())

        pairs = [[caption, x.description] for x in response_rows]
        scores = self.reranker.predict(pairs)
        scores = sigmoid(scores)

        result = []

        for response_row, score in zip(response_rows, scores):
            rerank_row = RerankRow(score=score, **response_row.model_dump())
            result.append(rerank_row)

        return result
