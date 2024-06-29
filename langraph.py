import os
import pandas as pd
import json
from langgraph.graph import END, StateGraph, START
from typing_extensions import TypedDict
from openai import OpenAI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from scripts.chains import structured_chunk, structured_completeness_check, kg_creation, kg_completeness_check
from scripts.io import load_data, write_to_json

load_dotenv()

output_file_path = "D:/LLM/Knowledge_Graph_Groq/KG/data_input/check1.json"
pdf_file = "D:/LLM/input/check.pdf"
chunks, pages = load_data(path=pdf_file)


# -------------------LLM prompts and chains---------------------------------

schema = '''
```[
    {
        "node_1": str(key_entity_1 extracted from structured data),
        "node_2": str(key_entity_2 extracted from sturctured data),
        "edge" : str(relationship between node_1 and node_2)
    },
    {...}
]```
'''

chat = ChatOpenAI(
    temperature=0,
    model = "llama3-70b-8192",
    base_url="https://api.groq.com/openai/v1",
    api_key=" ",
)


structured_chunk_chain = structured_chunk(chat)
structured_completeness_check_chain = structured_completeness_check(chat)
kg_creation_chain = kg_creation(chat)
kg_completeness_check_chain = kg_completeness_check(chat)
# --------------------------Langraph Implementstion------------------------------

class GraphState(TypedDict):
        """
        Represents the state of our graph.

        Attributes:
            question: question
            generation: LLM generation
            documents: list of documents 
        """
        text_chunk : str
        struct_out: str
        struct_check: str
        kg_out : list
        kg_check : list
        


def struct_out(state):
        
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---JSON_Creation---")
        text = state["text_chunk"]
        # print(text)

        struct_output = structured_chunk_chain.invoke({"text_chunk": text})
        # print(json_output)
        
        return {"struct_out": struct_output.content}





def kg_out(state):
        
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---Knowledge_Graph_Creation---")
        struct_check = state["struct_check"]
        # print(json_output)
        
        kg_output = kg_creation_chain.invoke({"struct_out": struct_check, "schema": schema})
        # print(kg_output)
        
        return {"kg_out": kg_output}


def struct_check(state):
    
    
    print("------Completeness Check----------")
    text = state["text_chunk"]
    struct_out = state["struct_out"]
    struct_updated = structured_completeness_check_chain.invoke({"text_chunk": text, "struct_out": struct_out})
    struct_updated = struct_updated.content
    # output
    start_index = struct_updated.find("Updated Complete Structured Data:") + 33
    end_index = len(struct_updated)
    struct_updated=struct_updated[start_index:end_index]
    return {"struct_check": struct_updated}


def kg_check(state):
     
     print("---------KG completeness check---------")
     struct_out = state["struct_check"]
     kg_out = state["kg_out"]

     kg_updated = kg_completeness_check_chain.invoke({"struct_out": struct_out, "kg_out": kg_out, "schema": schema})
     kg_updated = kg_updated.content
     start_index = kg_updated.find("New entries to Knowledge Graph:") + 31
     end_index = len(kg_updated)

     kg_updated=kg_updated[start_index:end_index]
     kg_updated_list = json.loads(kg_updated)
     kg_updated = kg_out.append(kg_updated_list)
     
     return {"kg_check": kg_updated}




workflow = StateGraph(GraphState)

    # Define the nodes
workflow.add_node("structured_out", struct_out) # web search
workflow.add_node("kg_creation", kg_out)
workflow.add_node("structured_validation", struct_check)
workflow.add_node("kg_validation", kg_check)
workflow.set_entry_point("structured_out")
workflow.add_edge("structured_out", "structured_validation")
workflow.add_edge("structured_validation", "kg_creation")
workflow.add_edge("kg_creation", "kg_validation")
workflow.set_finish_point("kg_validation")

app = workflow.compile()

for chunk in range(chunks):

    ans = app.invoke({"text_chunk": pages[chunk].page_content})
    try:
        kg_append = pd.DataFrame(ans["kg_check"])
        kg_df = pd.concat([kg_df, kg_append], ignore_index=True)

    except:
        kg_df = pd.DataFrame(ans["kg_check"])

    print("relationships generated: ", len(kg_df))
    

# --------convert and save to JSON file----------

write_to_json(output_file_path=output_file_path, kg_df=kg_df)