from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate




def structured_chunk(chat):
    system = '''Please convert the following text chunk into structured knowledge. Also take some liberties with the formatting and categorization of the information to make it more readable and structured. \n
    Don't include any explanation and don't leave any information \n
    Also make sure that don't hallucinate any information \n
    '''
    human = "{text_chunk}"
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    chain = prompt | chat
    return chain


def structured_completeness_check(chat):
    system = '''Imagine you are a Completeness Checker tasked with ensuring that each and every information is captured in structured data based on a given text excerpt. \n
    Your objective is to analyze the text chunk provided and identify any missing details that is missing in the structured data. \n
    Your recommendations should be clear, concise, and reflect an understanding of the context to enhance the accuracy and completeness of the data.\n
    To guide you through this task, use the following format to assess the situation and propose any necessary updates: \n

    ## Missing Information:
    $missing_information

    ## Updated Complete Structured Data:
    $updated_structured_data

    Your challenge is to analyze the given text chunk and the structured data to identify any missing information and update the given knowledge graph to ensure completeness.
    '''


    human = '''Here is the given text chunk:\n
        {text_chunk}
        \n ------- \n
        Here is the structured data:\n
        {struct_out}
    '''
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    chain = prompt | chat
    return chain




def kg_creation(chat):
    system = '''You are given a structured knowledge data. Extrapolate as many entities (nodes) and relationships (edges) as you can from the structured data and generate a structured knowledge graph. \n
    Make sure to include each and every minute detail in knowledge graph. \n
    Make sure to return a JSON blob with keys 'node_1, 'node_2' and 'edge'. \n
    The output JSON format: \n
    {schema}
    '''
    human = '''Here is the structured knowledge data: \n
    {struct_out}
    '''
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    chain = prompt | chat | JsonOutputParser()
    return chain


def kg_completeness_check(chat):
    system = '''Based on given structured data find out if there is any information not captured in given knowledge graph and provide new entries to the given knowledge graph accordingly. \n
    To guide you through this task, use the following format to assess the situation and propose any necessary updates: \n

    ## Missing Information:
    $missing_information

    ## New entries to Knowledge Graph:
    - Give output in the follwoing format: 
        {schema}

    Your challenge is to analyze the given structured data and the knoledge graph to identify any missing information and update knowledge graph accordingly. \n
    Only give output which is not included in given knowledge graph keeping output schema in mind
    Don't include any explanation and description in the output'''


    human = '''Here is the given structured data:\n
        {struct_out}
        \n ------- \n
        Here is the incomplete knowledge graph:\n
        {kg_out}
    '''
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    chain = prompt | chat
    return chain



