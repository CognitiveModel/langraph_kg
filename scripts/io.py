from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
import os

def load_data(path):

    
    loader = PyMuPDFLoader(path)
    # loader = DirectoryLoader(inputdirectory, show_progress=True)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )

    pages = splitter.split_documents(documents)
    print("Number of chunks = ", len(pages))
    # print("First chunk content: \n", pages[0].page_content)

    # pages[1].page_content
    return len(pages), pages


def write_to_json(output_file_path, kg_df):
    if os.path.exists(output_file_path):
        json_string = kg_df.to_dict(orient='records')
            # print(json_string2)


            # Read the existing data from the JSON file
        with open(output_file_path, 'r') as file:
            data = json.load(file)

            data.extend(json_string)    


            # Write the updated data back to the JSON file
        with open(output_file_path, 'w') as file:
            json.dump(data, file)

        print("New entry added successfully.")





    else:
        json_string = kg_df.to_json(orient='records')  # 'records' format for a list of dictionaries


                    # Save JSON to a file

        with open(output_file_path, 'w') as json_file:
            json_file.write(json_string)
        print("Entry added successfully.")