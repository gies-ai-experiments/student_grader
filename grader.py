import json
import asyncio
from langchain_openai import AzureChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import re


class Grader:
    def __init__(self, model, rubric_file="docs/rubric_data.json"):
        self.model = model
        self.rubric_file = rubric_file
        self.rubric_text = self.create_rubric_text()
        self.llm = AzureChatOpenAI(
            azure_deployment="gpt-4-32k",
            openai_api_version="2024-02-01",
            temperature=0.3,
        )
        self.parser = PydanticOutputParser(pydantic_object=self.ToolArgsSchema)

    class ToolArgsSchema(BaseModel):
        feedback: dict = Field(
            description="Feedback for each rubric criterion without scores"
        )
        summary: str = Field(
            description="Summary of the student's answer based on the rubric"
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

    @staticmethod
    def parse_ai_response_to_json(text_response):
        # Split the response into sections based on the pattern of titles followed by their descriptions
        sections = re.split(r"(?=\b[A-Za-z\s]+?:)", text_response)

        extracted_data = {"feedback": {}, "summary": ""}

        # Process each section to fill the feedback and summary appropriately
        for section in sections:
            # Use regex to capture the title and the content of each section
            match = re.match(r"([A-Za-z\s]+?):\s*(.*)", section, re.DOTALL)
            if match:
                title, content = match.groups()
                # Normalize line breaks and extra spaces for content
                content = re.sub(r"\s+", " ", content).strip()
                if "Summary" in title:
                    extracted_data["summary"] = content
                else:
                    extracted_data["feedback"][title.strip()] = content

        return json.dumps(extracted_data, indent=2)

    async def grade_text(self, student_text):
        prompt = f"""Based on the following rubric, provide detailed feedback for the student's response:
        {self.rubric_text}
        
        Exclude any scores and do not provide feedback on critique-related criteria.

        Student's response:
        {student_text}

        Provide the feedback in the following format:
        [Rubric Criterion]: [Feedback]
        ...
        Summary: [Summary of the student's answer based on the rubric]
        """
        response = await asyncio.get_event_loop().run_in_executor(
            None, self.llm.invoke, prompt
        )

        if hasattr(response, "message"):
            text_response = response.message
        else:
            text_response = str(response)
        print(text_response)
        response_json = self.parse_ai_response_to_json(text_response)

        try:
            grading_result = self.parser.parse(response_json)
            print(grading_result)
            return grading_result
        except Exception as e:
            print("Failed to parse the response. Response text was:", response_json)
            raise e


async def main():
    grader = Grader(model="gpt-4")
    student_text = "Here is the student's response to be graded..."
    grading_result = await grader.grade_text(student_text)
    print(grading_result)


if __name__ == "__main__":
    asyncio.run(main())
