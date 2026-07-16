from pathlib import Path


def load_vehicle_classes() -> list[str]:
    classes_path = Path(__file__).resolve().parents[1] / "vehicle_classes.txt"
    if not classes_path.exists():
        return [
            "car",
            "truck",
            "bus",
            "motorcycle",
            "bicycle",
            "pedestrian",
            "traffic sign",
            "traffic light",
            "road",
            "sidewalk",
            "building",
            "tree",
            "fence",
            "pole",
        ]
    return [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]


class GPTPrompt:
    def __init__(self):
        valid_tag_list = load_vehicle_classes()
        valid_tags = ", ".join(valid_tag_list)
        valid_tags_json = ", ".join(f'"{tag}"' for tag in valid_tag_list)

        self.system_prompt = """You identify outdoor nuScenes objects from multiple captions of the same crop.
Return only valid JSON with keys:
1. summary: a short description grounded in the captions.
2. possible_tags: exactly the full valid tag list given below.
3. object_tag: the best single final tag.

Important rules:
- This is an outdoor driving dataset, not an indoor scene.
- Valid object_tag values are: {valid_tags}, invalid.
- possible_tags must always be exactly: [{valid_tags}].
- If captions repeatedly mention a specific subtype such as motorcycle, bicycle, car, truck, or bus, prefer that subtype over a generic label.
- Keep fine-grained attributes when supported by captions, such as color, brand, or body type, in the summary.
- If captions are inconsistent or clearly describe different objects, set object_tag to invalid.
- Do not mention anything outside JSON.""".format(valid_tags=valid_tags)
        
        self.example_1 = """{
        "id": 1,
        "captions": [
          "a motorcycle.",
          "a motorcycle.",
          "a motorcycle."
        ]
        }"""

        self.response_1 = """{
        "summary": "motorcycle observed in nuScenes crops",
        "possible_tags": [""" + valid_tags_json + """],
        "object_tag": "motorcycle"
        }"""

        self.example_2 = """{
        "id": 2,
        "captions": [
          "a blue car.",
          "a blue car.",
          "a blue car."
        ]
        }"""

        self.response_2 = """{
        "summary": "blue car observed in nuScenes crops",
        "possible_tags": [""" + valid_tags_json + """],
        "object_tag": "car"
        }"""

        self.example_3 = """{
        "id": 3,
        "captions": [
          "a silver Honda Fit.",
          "a white car.",
          "a car."
        ]
        }"""

        self.response_3 = """{
        "summary": "silver Honda Fit car",
        "possible_tags": [""" + valid_tags_json + """],
        "object_tag": "car"
        }"""

        self.example_4 = """{
        "id": 4,
        "captions": [
          "a road sign.",
          "a traffic sign.",
          "a bus stop sign."
        ]
        }"""

        self.response_4 = """{
        "summary": "traffic sign or bus stop sign",
        "possible_tags": [""" + valid_tags_json + """],
        "object_tag": "traffic sign"
        }"""

        self.example_5 = """{
        "id": 5,
        "captions": [
          "a pedestrian.",
          "a person walking on the sidewalk."
        ]
        }"""

        self.response_5 = """{
        "summary": "pedestrian on sidewalk",
        "possible_tags": [""" + valid_tags_json + """],
        "object_tag": "pedestrian"
        }"""


    def get_json(self):
        prompt_json = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": self.example_1
            },
            {
                "role": "assistant",
                "content": self.response_1
            },
            {
                "role": "user",
                "content": self.example_2
            },
            {
                "role": "assistant",
                "content": self.response_2
            },
            {
                "role": "user",
                "content": self.example_3
            },
            {
                "role": "assistant",
                "content": self.response_3
            },
            {
                "role": "user",
                "content": self.example_4
            },
            {
                "role": "assistant",
                "content": self.response_4
            },
            {
                "role": "user",
                "content": self.example_5
            },
            {
                "role": "assistant",
                "content": self.response_5
            }
        ]
        return prompt_json

# Usage example
if __name__ == "__main__":
    prompt_obj = GPTPrompt()
    json_data = prompt_obj.get_json()
    print(json_data)
