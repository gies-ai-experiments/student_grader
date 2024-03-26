import requests
from bs4 import BeautifulSoup
import os
import json
import re
from langchain_openai import AzureChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import asyncio

rubric = None
message = None
rubric_file = "docs/rubric_data.json"


def extract_and_save_instruction(input_url, access_token):
    global base_url, rubric, message
    match = re.match(
        r"https://canvas.illinois.edu/courses/(\d+)/discussion_topics/(\d+)", input_url
    )
    if match:
        course_id, discussion_topic_id = match.groups()
    else:
        raise ValueError("Invalid URL")
    base_url = "https://canvas.illinois.edu"
    headers = {"Authorization": f"Bearer {access_token}"}

    instruction_url = (
        f"{base_url}/api/v1/courses/{course_id}/discussion_topics/{discussion_topic_id}"
    )
    instruction_response = requests.get(instruction_url, headers=headers)

    if instruction_response.ok:
        instruction_data = instruction_response.json()
        rubric = []

        # Extract title
        if "title" in instruction_data:
            title = instruction_data["title"]
            rubric = [{"title": title}]

        # Try extracting from 'assignment' description first
        description_html = instruction_data.get("assignment", {}).get("description")
        if description_html:
            soup = BeautifulSoup(description_html, "html.parser")
            description = soup.get_text()
            rubric.append({"instruction": description})
        else:
            # If no description, try extracting from 'message'
            message_html = instruction_data.get("message")
            if message_html:
                soup = BeautifulSoup(message_html, "html.parser")

                # Handle HTML content for message
                for br in soup.find_all("br"):
                    br.replace_with("\n")
                for li in soup.find_all("li"):
                    li.insert_after(soup.new_string("\n"))
                for p in soup.find_all("p"):
                    p.insert_before(soup.new_string("\n"))
                    p.insert_after(soup.new_string("\n"))

                message = soup.get_text()
                rubric.append({"instruction": message.strip()})

        # Extract rubric and points possible, if available
        if "rubric" in instruction_data.get("assignment", {}):
            rubric.extend(instruction_data["assignment"]["rubric"])
            points_possible = instruction_data["assignment"].get("points_possible")
            if points_possible is not None:
                rubric.append({"points_possible": points_possible})

        # Handling the 'docs' folder
        print("Creating docs folder")
        if not os.path.exists("docs"):
            os.makedirs("docs")

        # Save to JSON file
        with open(rubric_file, "w") as f:
            json.dump(rubric, f)

        print("Extracted instructions and rubric")
    else:
        print(f"Error: {instruction_response.text}")


class Summarizer:
    def __init__(self, model, rubric_file="docs/rubric_data.json"):
        self.model = model
        self.rubric_file = rubric_file
        self.rubric_text = self.create_rubric_text()
        self.llm = AzureChatOpenAI(
            azure_deployment="gpt-4-32k",
            openai_api_version="2023-09-01-preview",
            temperature=0.3,
        )

    def create_rubric_text(self):
        with open(self.rubric_file, "r") as file:
            rubric = json.load(file)
        rubric_text = []
        for r in rubric:
            if "description" in r and "ratings" in r:
                rubric_text.append(
                    f"RUBRIC CATEGORY: {r['description']}\n"
                    + "\n".join(
                        [
                            f"POINTS: {rating['points']} CRITERIA: {rating['description']}"
                            for rating in r["ratings"]
                        ]
                    )
                )
            elif "points_possible" in r:
                rubric_text.append(f"MAX POINTS POSSIBLE: {r['points_possible']}")
            elif "title" in r:
                rubric_text.append(f"TITLE: {r['title']}")
            elif "instruction" in r:
                rubric_text.append(f"DISCUSSION INSTRUCTIONS: {r['instruction']}")
        return "\n".join(rubric_text)

    async def summarize(self):
        prompt = f"""
        Summarize the following instructions and rubric, focusing on the main assignment tasks and evaluation criteria. Exclude any instructions about critiquing peers or giving feedback. 
        Structure the summary to highlight the assignment's objectives, tasks, and rubric categories with their points. Ensure the summary is concise.
        Do not add points or ratings to the rubric categories.

        Instructions and Rubric:
        {self.rubric_text}

        Format the summary as follows:
        Assignment Objective:
        Main Tasks:
        Evaluation Criteria:
        """

        response = await asyncio.get_event_loop().run_in_executor(
            None, self.llm.invoke, prompt
        )

        if hasattr(response, "message"):
            return response.message
        else:
            return str(response)


async def main():
    summ = Summarizer(model="gpt-4-32k")
    summary = await summ.summarize()
    print(summary)


if __name__ == "__main__":
    asyncio.run(main())
