from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from config import OPENAI_API_KEY
from get_filenames import get_files_in_directory
from loguru import logger
from pprint import pprint
model = ChatOpenAI(temperature=0)


class Category(BaseModel):
    categories: str = Field(description="Provide several names for"
                                        "folders that fits each filename")


class MapFilesToCategory(BaseModel):
    map_files: str = Field(description="Each file should be in his own directory."
                                       "Map files to suggested directory names")


def construct_chain_invoke(filenames):  # NOQA
    parser = JsonOutputParser()
    suggest_dirs = ("Suggest category for list of my files.\n"
                    f"Files for organize:\n{filenames}"
                    )
    map_files = ("Map each file to his owd directory.\n"
                 "Each file should relate to only one category\n"
                 "Answer in the following format:\n"
                 "NameDirectory: files to include. Etc for each dir"
                 )
    query = suggest_dirs + map_files
    dirs_prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["suggest_dirs"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    map_prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["map_files"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = dirs_prompt | map_prompt | model | parser
    result = chain.invoke({"query": query})
    return result


if __name__ == "__main__":
    filenames = get_files_in_directory()
    llm_answer = construct_chain_invoke(filenames)
    logger.info(f"llm_answer is {llm_answer}")
    pprint(llm_answer)
