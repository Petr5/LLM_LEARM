from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from config import OPENAI_API_KEY
from get_filenames import get_files_in_directory
from loguru import logger
model = ChatOpenAI(temperature=0)


class Category(BaseModel):
    categories: str = Field(description="Provide names for folder that fits filenames")
    map_files: str = Field(description="Each file should be in his own directory. Map file to suggested directory names")


def construct_chain_invoke(filenames):
    parser = JsonOutputParser()
    filenames_query = ("Suggest category for list of my files.\n"
                       f"Files for organize:\n{filenames}"
                       )
    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | model | parser
    result = chain.invoke({"query": filenames_query})
    return result


if __name__ == "__main__":
    filenames = get_files_in_directory()
    llm_answer = construct_chain_invoke(filenames)
    logger.info(f"llm_answer is {llm_answer}")
