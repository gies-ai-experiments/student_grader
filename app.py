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
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed


dotenv.load_dotenv()
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
os.environ["AZURE_OPENAI_API_KEY"] = AZURE_OPENAI_API_KEY
os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_OPENAI_ENDPOINT

grader = None
grader_qa = None
disabled = gr.update(interactive=False)
enabled = gr.update(interactive=True)
grading_model = "gpt-4-32k"
qa_model = "gpt-4-32k"

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


def ingest(url, canvas_api_key, state):
    global grader, llm, embeddings
    try:
        text = f"Downloaded discussion data from {url} to start grading"
        extract_and_save_instruction(url, canvas_api_key)
        grader = Grader(grading_model)
        response = "Ingested canvas data successfully"
        history = state.get("history", []) + [(text, response)]
        state.update(
            {
                "submitted": True,
                "history": history,
            }
        )
        # Indicate that the instructions have been processed and disable the submit button
        submit_button_text = "Instructions processed"
        submit_button_disabled = True
        # Return updated state and update UI elements as needed
        return (
            state,
            "",
            gr.update(value=submit_button_text),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    except Exception as e:
        print(f"Error during ingestion: {str(e)}")
        state.update(
            {
                "submitted": False,
            }
        )
        # Keep the submit button enabled in case of failure so the user can try again
        return (
            state,
            "Failed to ingest data. Please check the URL and API Key.",
            "",
            "",
            "",
        )


async def summarize_rubric(state):
    try:
        # Initialize the Summarizer with the model and rubric file path
        summarizer = Summarizer(model="gpt-4", rubric_file="docs/rubric_data.json")

        # Generate the summary
        summary = await summarizer.summarize()

        # Format the summary as HTML
        rubric_html = f"""
        <div style='font-family: sans-serif; max-width: 800px; margin: auto; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); background-color: #f9f9f9;'>
            <h2 style='color: #333;'>Rubric Summary</h2>
            <p><strong>Assignment Objective:</strong> The objective of the Module 4 Assignment: New Business Idea is to apply the concepts learned in the course to develop a new business idea leveraging Large Language Models or other generative AI tools. The idea should have potential for monetization and should leverage the principles of the 'Social Web' or Web 3.0.</p>
            <p><strong>Main Tasks:</strong> The assignment is divided into two parts. In Part 1, students are required to come up with a new business idea, considering their interests, passions, and potential relevance to the CU community. They should interact with a Discord bot to generate and refine five business ideas, ultimately selecting one for submission. The idea should be presented in a 600-word essay, detailing the need it fulfills, the goals it serves, how it will leverage course concepts, its competitive advantage, and its monetization strategy. All sources, including the chatbot, must be cited.</p>
            <p><strong>Evaluation Criteria:</strong> The assignment will be evaluated on the following rubric categories: Business Idea Creativity (5 points), Business Need and Goals (5 points), Competitive Advantage (5 points), Leveraging Course Concepts (5 points), Monetization Strategy (5 points), and Organization and Structure (5 points). The maximum possible points for the assignment is 15.</p>
        </div>
        """

        return rubric_html, state
    except Exception as e:
        print(f"Error summarizing rubric: {str(e)}")
        return "Failed to summarize rubric data.", state


def view_rubric_summary(state):
    return asyncio.run(summarize_rubric(state))


def clean_feedback_text(text):
    # Initially replace escaped newlines and potential escaped quotes
    cleaned_text = text.replace("\\n", "<br>").replace('\\"', "")
    # Trim leading and trailing <br> tags that might result from initial or final newlines
    cleaned_text = cleaned_text.strip("<br>").strip()
    # Remove any trailing periods that may be left after other replacements
    cleaned_text = cleaned_text.rstrip(".").rstrip()
    return cleaned_text


def get_feedback(student_input, state):
    global grader
    message = (
        "Please submit your response to get AI-powered feedback."
        if not state.get("submitted", False)
        else ""
    )

    if state.get("submitted", False):
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
        message = "This feedback is generated by AI. Review the feedback, revise your submission accordingly, and consider resubmitting for further evaluation."
        return feedback_html, state, message
    else:
        return "", state, message


def reset(state):
    # Function to reset the UI
    state["submitted"] = False
    return "", "", "", state


def add_text_to_chatbot(text, state):
    # Function to handle chatbot interaction
    # The chat should work only if the state 'submitted' is True
    if state.get("submitted"):
        return text, state
    else:
        return "Please submit your response first.", state


def bot(history):
    return history


def get_output_dir(orig_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output", orig_name)
    return output_dir


def check_rubric_file_exists():
    rubric_file_path = "docs/rubric_data.json"  # Update this path as necessary
    return os.path.exists(rubric_file_path)


with gr.Blocks() as demo:
    gr.Markdown("<h2><center>Canvas Discussion Feedback Interface</center></h2>")

    initial_submitted_state = check_rubric_file_exists()

    # Initialize state with the correct 'submitted' value
    state = gr.State({"submitted": initial_submitted_state})

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Group():
                # Set the interactive state based on the initial_submitted_state
                url = gr.Textbox(
                    label="Canvas Discussion URL",
                    placeholder="Enter your Canvas Discussion URL",
                    interactive=not initial_submitted_state,
                )
                canvas_api_key = gr.Textbox(
                    label="Canvas API Key",
                    placeholder="Enter your Canvas API Key",
                    type="password",
                    interactive=not initial_submitted_state,
                )
                with gr.Column(scale=2):
                    submit_button = gr.Button(
                        "Submit", visible=not initial_submitted_state
                    )
                    view_button = gr.Button(
                        "View Rubric", visible=initial_submitted_state
                    )

                # If the rubric file exists, update the UI elements to reflect the processed state
                if initial_submitted_state:
                    url.value = "Instructions processed"
                    canvas_api_key.value = "Instructions processed"
                    submit_button.visible = False

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

    # Link functions to buttons with state included in inputs and outputs
    submit_button.click(
        ingest,
        inputs=[url, canvas_api_key, state],
        outputs=[state, url, canvas_api_key, submit_button],
    )
    view_button.click(
        view_rubric_summary, inputs=[state], outputs=[feedback_html_display, state]
    )
    feedback_button.click(
        get_feedback,
        inputs=[student_input, state],
        outputs=[feedback_html_display, state, ai_message],
    )
    reset_button.click(
        reset, inputs=[state], outputs=[url, canvas_api_key, student_input, state]
    )


demo.launch()
