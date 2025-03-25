import os
import asyncio
import json
from datetime import datetime
from typing import List
from pydantic import BaseModel, SecretStr
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent, Controller, Browser, BrowserConfig

# Load environment variables
load_dotenv()


# Define the output format as a Pydantic model
class Output(BaseModel):
    short_file_title: str
    title: str
    content: str
    urls: List[str]
    creation_date: str
    summary: str
    keywords: List[str]


# Initialize the model
def initialize_llm() -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set in the environment variables.")
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp", api_key=SecretStr(api_key)
    )


# Initialize the browser
def initialize_browser() -> Browser:
    return Browser(
        config=BrowserConfig(
            chrome_instance_path="C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
        )
    )


def initialize_system_message() -> str:
    extend_system_message = """
    REMEMBER the these important RULES:
    ALWAYS open first a new tab
    IF you are stuck in a Cloudflare Securety Check (or similar), try to ask the User or find a way to bypass it
    """
    return extend_system_message


def save_out(parsed: Output):
    with open(
        f'./result-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}-{parsed.short_file_title}',
        "w",
    ) as f:

        content = f"""
        Title: {parsed.title}
        ---------------------------------------------------
        Content: {parsed.content}
        ---------------------------------------------------
        URLs: {parsed.urls}
        Keywords: {parsed.keywords}
        Summary: {parsed.summary}
        ---------------------------------------------------
        Creation Date: {parsed.creation_date}
        """

        f.write(content)

    print(
        f"File saved as result-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{parsed.short_file_title}"
    )
    print(10 * "\n")
    print(content)


# Main function
async def main(task: str):
    llm = initialize_llm()
    browser = initialize_browser()
    controller = Controller(output_model=Output)

    agent = Agent(
        task=task,
        llm=llm,
        browser=browser,
        controller=controller,

    )

    out = await agent.run()
    result = out.final_result()

    if result:
        if isinstance(result, str):
            result = json.loads(result)

        parsed: Output = Output.model_validate(result)
        save_out(parsed)

    input("Press Enter to close the browser...")
    await browser.close()


if __name__ == "__main__":
    #task = ""   # initialize_system_message()
    #task += f"\n\nTASK:\n{input('Enter the Task >> ')}"
    task = input('Enter the Task >> ')
    asyncio.run(main(task))


# Which API Key is better (Free Plans do not include): ChatGPT, Gemini, DeepSeek or Claude. Also rate the API Key from 1 to 5 and also rate the Price-performance ratio on a skala from 1 to 10 for each API Key.
