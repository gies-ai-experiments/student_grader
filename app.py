import gradio as gr
import os
from langchain_openai import AzureChatOpenAI
import dotenv
from instructions import extract_and_save_instruction, Summarizer
from utils import reset_folder
from grader import Grader
import asyncio
import glob
import shutil
import time
import traceback
import json
import re
import pandas as pd
from fastapi import FastAPI
from concurrent.futures import ThreadPoolExecutor, as_completed


dotenv.load_dotenv()
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
os.environ["AZURE_OPENAI_API_KEY"] = AZURE_OPENAI_API_KEY
os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_OPENAI_ENDPOINT
app = FastAPI()
grader = None
grader_qa = None
disabled = gr.update(interactive=False)
enabled = gr.update(interactive=True)
grading_model = "gpt-4-32k"
qa_model = "gpt-4-32k"

submit_visible = True
view_rubric_visible = False

llm = AzureChatOpenAI(
    azure_deployment=grading_model,
    openai_api_version="2024-02-01",
    temperature=0.1,
    streaming=True,
)


def add_text(history, text):
    print("Question asked: " + text)
    response = run_model(text)
    history = history + [(text, response)]
    print(history)
    return history, ""


def run_model(text):
    global grader, grader_qa
    start_time = time.time()
    print("start time:" + str(start_time))
    try:
        response = grader_qa.agent.run(text)
    except Exception as e:
        response = "I need a break. Please ask me again in a few minutes"
        print(traceback.format_exc())

    sources = []
    source = ",".join(set(sources))
    end_time = time.time()

    response = response + "\n\n" + "Time taken: " + str(end_time - start_time)
    print(response)
    print(sources)
    print("Time taken: " + str(end_time - start_time))
    return response


def ingest(url, canvas_api_key):
    global grader, llm, submit_visible, view_rubric_visible
    try:
        text = f"Downloaded discussion data from {url} to start grading"
        extract_and_save_instruction(url, canvas_api_key)
        grader = Grader(grading_model)

        submit_visible = False
        view_rubric_visible = True
        
        return (
            gr.update(value="Instructions processed", interactive=False),
            gr.update(value="Instructions processed", interactive=False),
            gr.update(visible=submit_visible),
            gr.update(visible=view_rubric_visible),
        )
    except Exception as e:
        print(f"Error during ingestion: {str(e)}")
        
        return (
            "Failed to ingest data. Please check the URL and API Key.",
            gr.update(interactive=True),
            gr.update(visible=True),
            gr.update(visible=False),
        )


async def summarize_rubric():
    try:
        # Initialize the Summarizer with the model and rubric file path
        summarizer = Summarizer(model="gpt-4", rubric_file="docs/rubric_data.json")

        # Generate the summary
        summary = await summarizer.summarize()
        content = summary.replace('\r\n', '\n').replace('\r', '\n')

        # Patterns to match each section based on the headings.
        assignment_objective_pattern = r"Assignment Objective:\s*(.*?)(?=Main Tasks:|$)"
        main_tasks_pattern = r"Main Tasks:\s*(.*?)(?=Evaluation Criteria:|$)"
        evaluation_criteria_pattern = r"Evaluation Criteria:\s*(.*)"

        # Using regex to find each section in the content.
        assignment_objective_match = re.search(assignment_objective_pattern, content, re.DOTALL)
        main_tasks_match = re.search(main_tasks_pattern, content, re.DOTALL)
        evaluation_criteria_match = re.search(evaluation_criteria_pattern, content, re.DOTALL)

        # Extracting text if matches are found, otherwise return an empty string.
        assignment_objective = assignment_objective_match.group(1).strip() if assignment_objective_match else ""
        main_tasks = main_tasks_match.group(1).strip() if main_tasks_match else ""
        evaluation_criteria = evaluation_criteria_match.group(1).strip() if evaluation_criteria_match else ""

        # Optionally replace newline characters with HTML breaks for display.
        assignment_objective = assignment_objective.replace('\\', '')[1:-2]
        main_tasks = main_tasks.replace('\\', '')[1:-2]
        evaluation_criteria = evaluation_criteria.replace('\\', '')[1:-1]
        
        print(f"Assignment Objective: {assignment_objective}")
        print(f"Main Tasks: {main_tasks}")
        print(f"Evaluation Criteria: {evaluation_criteria}")
        
        rubric_html = f"""
        <div style='font-family: sans-serif; max-width: 800px; margin: auto; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); background-color: #f9f9f9;'>
            <h2 style='color: #333;'>Rubric Summary</h2>
            <p><strong>Assignment Objective:</strong><br>{assignment_objective}</p>
            <p><strong>Main Tasks:</strong><br>{main_tasks}</p>
            <p><strong>Evaluation Criteria:</strong><br>{evaluation_criteria}</p>
        </div>
        """

        return rubric_html
    except Exception as e:
        print(f"Error summarizing rubric: {str(e)}")
        return "Failed to summarize rubric data."


def view_rubric_summary():
    return asyncio.run(summarize_rubric())


def clean_feedback_text(text):
    # Initially replace escaped newlines and potential escaped quotes
    cleaned_text = text.replace("\\n", "<br>").replace('\\"', "")
    # Trim leading and trailing <br> tags that might result from initial or final newlines
    cleaned_text = cleaned_text.strip("<br>").strip()
    # Remove any trailing periods that may be left after other replacements
    cleaned_text = cleaned_text.rstrip(".").rstrip()
    return cleaned_text


def get_feedback(student_input):
    global grader
    message = (
        "Please submit your response to get AI-powered feedback."
    )

    if True:  # Assuming submission has occurred
        if grader is None:
            grader = Grader(model="gpt-4")  # Ensure Grader is initialized

        def run_async_grader(student_text):
            async def grade_text_async():
                return await grader.grade_text(student_text)

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(grade_text_async())
            loop.close()
            return result

        with ThreadPoolExecutor() as executor:
            future = executor.submit(run_async_grader, student_input)
            grading_result = future.result()  # This blocks until the future is complete

        feedback_html = """
        <div style='font-family: sans-serif; max-width: 800px; margin: auto; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); background-color: #f9f9f9;'>
            <h2 style='color: #333;'>Feedback Summary</h2>
        """

        # Use dot notation to access feedback attribute
        if hasattr(grading_result, "feedback"):
            for criterion, feedback in grading_result.feedback.items():
                formatted_feedback = feedback.replace("\n", "<br>")
                feedback_html += f"""
                <p><strong>{criterion}:</strong> {formatted_feedback}</p>
                """
        else:
            feedback_html += "<p>No feedback details available.</p>"

        # Access summary using dot notation
        summary = getattr(grading_result, "summary", "No summary provided").replace(
            "\n", "<br>"
        )
        feedback_html += f"""
            <h3 style='color: #333;'>Overall Summary</h3>
            <p>{summary}</p>
        </div>
        """
        message = "This feedback is powered by GPT4 and given based on description and rubric. Review the feedback, revise your submission accordingly, and consider resubmitting for further evaluation."
        return feedback_html, message
    else:
        return "", message


def reset():
    global submit_visible, view_rubric_visible

    rubric_file_path = "docs/rubric_data.json" 
    try:
        os.remove(rubric_file_path) 
        print("Rubric data deleted successfully.")
    except FileNotFoundError:
        print("Rubric data file not found.")
    except Exception as e:
        print(f"An error occurred while deleting rubric data: {e}")
        
    submit_visible = True
    view_rubric_visible = False
    return (
        gr.update(value="", interactive=True),  # Update for url Textbox
        gr.update(value="", interactive=True),  # Update for canvas_api_key Textbox
        gr.update(value="", interactive=True),  # Update for student_input TextArea
        gr.update(visible=submit_visible),      # Update for submit_button visibility
        gr.update(visible=view_rubric_visible)  # Update for view_button visibility
    )  


def add_text_to_chatbot(text):
    # Function to handle chatbot interaction
    # The chat should work only if the state 'submitted' is True
    if True:  # Assuming submission has occurred
        return text
    else:
        return "Please submit your response first."


def bot(history):
    return history


def get_output_dir(orig_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output", orig_name)
    return output_dir


# def check_rubric_file_exists():
#     rubric_file_path = "docs/rubric_data.json"  # Update this path as necessary
#     return os.path.exists(rubric_file_path)


with gr.Blocks() as demo:
    gr.Markdown("<h2><center>Canvas Discussion Feedback Interface</center></h2>")


    with gr.Row():
        with gr.Column(scale=2):
            with gr.Group():
                # Set the interactive state based on the initial_submitted_state
                url = gr.Textbox(
                    label="Canvas Discussion URL",
                    placeholder="Enter your Canvas Discussion URL",
                )
                canvas_api_key = gr.Textbox(
                    label="Canvas API Key",
                    placeholder="Enter your Canvas API Key",
                    type="password",
                )
                with gr.Column(scale=2):
                    submit_button = gr.Button("Submit", visible=True)  # Initial visibility based on the default assumption
                    view_button = gr.Button("View Rubric", visible=False)

                reset_button = gr.Button("Reset")

            with gr.Group():
                student_input = gr.TextArea(
                    label="Your Response",
                    placeholder="Enter your discussion response here",
                )
                feedback_button = gr.Button("Get Feedback")

        with gr.Column(scale=3):
            ai_message = gr.Textbox(
                label="AI Processing Message",
                value="Submit your response to receive AI-powered feedback.",
                interactive=False,
                visible=True,
            )
            feedback_html_display = gr.HTML()

    # Link functions to buttons without state included in inputs and outputs
    submit_button.click(
        ingest,
        inputs=[url, canvas_api_key],
        outputs=[url, canvas_api_key, submit_button, view_button],
    )
    view_button.click(
        view_rubric_summary, inputs=[], outputs=[feedback_html_display]
    )
    feedback_button.click(
        get_feedback,
        inputs=[student_input],
        outputs=[feedback_html_display, ai_message],
    )
    reset_button.click(
        reset, inputs=[], outputs=[url, canvas_api_key, student_input, submit_button, view_button]
    )

app = gr.mount_gradio_app(app, demo, path="/")
#demo.launch(server_name="0.0.0.0", server_port=7000)
