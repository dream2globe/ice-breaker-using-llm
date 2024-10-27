from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from third_parties.linkedin import scrape_linkedin_profile

if __name__ == "__main__":
    load_dotenv()
    print("Hello LangChain")

    summary_template = """
    given the information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """
    
    linkedin_data = scrape_linkedin_profile(
        linkedin_profile_url="https://www.linkedin.com/in/eden-marco/",
        mock=True
    )
        
    prompt = PromptTemplate.from_template(summary_template)
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=2048)
    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser
    res = chain.invoke(input={"information": linkedin_data})

    print(res)
