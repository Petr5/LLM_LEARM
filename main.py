import os

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from loguru import logger
from pprint import pprint
# will be re-used by all my chains
LLM = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1, request_timeout=60)

# example chain8
answer_question_chain = LLMChain.from_string(
    LLM,
    """
    You are folder organizer. 
    I will load my files in specific folder and you will suggest which files could
    be placed together logically. Also suggest name that generalize this files purposes or
    meanings
    Answer in the following format:
    Folder1 - file1, file2, file3
    Folder2 - file1, file, ...
    ...
    """,
)


def call_llm(filenames):

    llm_answer = answer_question_chain({
        "user_question": f"{filenames}",
    })
    return llm_answer


def get_files_in_directory(dir_path):
    filenames = os.listdir(dir_path)
    logger.info(f"filenames are {filenames}")
    pprint(filenames)
    return filenames

if __name__ == "__main__":
    dir_path = r"C:\Users\User\Downloads"
    call_llm(dir_path)
    filenames = get_files_in_directory(dir_path)
    llm_answer = call_llm(filenames)
    logger.info(f"llm_answer is {llm_answer}")
    pprint(llm_answer)
