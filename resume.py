from langchain_community.llms.openai import OpenAI
from config import OPENAI_API_KEY
import os
from loguru import logger
from pprint import pprint
ai = OpenAI()

resumes_db = []

# @ai.cache
def extract_resume(resume_text):
    # Generate field extraction prompts
    name_prompt = [f"Extract the candidate name from this resume:\n{resume_text}\nName:"]
    email_prompt = [f"Extract the candidate email from this resume:\n{resume_text}\nEmail:"]
    # Call LLM to extract fields
    name = ai.generate(name_prompt)
    email = ai.generate(email_prompt)
    logger.info(f"email is {email.generations[0][0].text}")
    logger.info(f"name is {name.generations[0][0].text}")
    return {
        "name": name.llm_output,
        "email": email.llm_output
    }


# Process corpus
for path in os.listdir("resumes"):
    path = os.path.join("resumes", path)
    text = open(path).read()
    logger.info(f"text from resume is {text}")

    data = extract_resume(text)

    resumes_db.append(data)
pprint(resumes_db)