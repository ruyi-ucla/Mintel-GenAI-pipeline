import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import json
import os   
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import DataFrameLoader
from dotenv import load_dotenv
import os
from langchain_core.documents.base import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
import textwrap



df = pd.read_csv('data/flavor_prev.csv')
months = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 
          7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
df['month'] = df['month'].map(months)
df['text'] = df.apply(
    lambda x: f"""This flavor {x['flavor_category']} appears in {x['num_product']} product category {x['product_category']} with {x['prevalence_flavor']:.2%} prevalence in {x['country']} in {x['month']} of {x['year']}.""".strip(), 
    axis=1
)
df_cut = df[df.year.isin([2023])&(df.product_super_category == 'Drink')]
df_cut = df_cut.drop_duplicates()



# Initialize LLM (GPT-4)
os.environ["OPENAI_API_KEY"] = "Enter Your Key Here or Retrieve from Environment"
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
def extract_query_details_llm(user_query):
    """
    Uses LLM to extract structured data from user input via prompt engineering.
    """
    system_prompt = """Extract the following details from the user's query:
    - Year (e.g., 2023)
    - Month (1-12)
    - Region (e.g., Asia Pacific, North America)
    - Country (e.g., AU, US, UK, China, France, Germany)
    - Product Category (e.g., Alcoholic Beverages, Hot Beverages)
    - Flavor Category (e.g., Berry Fruit, Citrus Fruit, Sugar)
    - Function (trend_analysis, compare_flavors, stat_summary)
    
    If any information is missing, return `null` for that field.
    Provide the response in strict JSON format.
    """
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"User Query: {user_query}")
    ]
    # Generate LLM response
    response = llm(messages)
    try:
        extracted_data = json.loads(response.content)
        return extracted_data
    except json.JSONDecodeError:
        return {"error": "Failed to parse JSON response from LLM"}


# Use local embeddings instead of OpenAI
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# Convert DataFrame to LangChain Documents
def process_dataframe_for_faiss(df):
    documents = []
    for _, row in df.iterrows():
        text = row["text"]
        # Create a LangChain Document object
        doc = Document(page_content=text, metadata={"year": row["year"], 
                                                    "month": row["month"],
                                                    "country": row["country"],
                                                    "product_category": row["product_category"],
                                                    "flavor_category": row["flavor_category"]}
                                                    )
        documents.append(doc)  
    return documents

# Prepare the FAISS index
vectorstore_path = "data/flavor_vector_db"
if os.path.exists(vectorstore_path):
    vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
else:
    docs = process_dataframe_for_faiss(df_cut)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(vectorstore_path)  # Persist the store

def retrieve_similar_queries(user_query):
    """
    Searches the vector database for past similar user queries to use as context.
    """
    similar_results = vectorstore.similarity_search(user_query, k=10)
    if similar_results:
        return [result.page_content for result in similar_results]
    return []



# Modify the function to generate more dynamic and conversational follow-up questions

def generate_conversational_follow_up(structured_query, similar_queries):
    """
    Uses LLM to generate a dynamic and conversational follow-up question based on missing fields
    and retrieved RAG context, adapting the style based on the query context.
    """
    missing_fields = [key for key, value in structured_query.items() if value is None]

    if not missing_fields:
        return None  # No missing fields, query is complete

    # Construct a conversational LLM prompt
    system_prompt = f"""
    You are an AI assistant helping a user analyze flavor trends. Your role is to make the interaction natural, engaging, 
    and context-aware by generating a well-formed, conversational follow-up question based on the missing information in their query.
    
    **User's Current Query Context:**
    {json.dumps(structured_query, indent=2)}
    
    **Past Relevant Queries (For Context & Suggestions):**
    {json.dumps(similar_queries, indent=2)}
    
    **Your Task:**
    - Identify **one missing detail** from the user's query.
    - Generate a **natural-sounding, engaging** follow-up question.
    - Adapt the tone dynamically based on the available information.
    - If possible, reference past trends or suggest relevant insights.

    **Examples of Good Responses:**
    - "I see you're exploring berry fruit trends in Asia Pacific! Would you like to focus on a specific country, such as Japan or China?"
    - "I noticed citrus flavors were trending in alcoholic beverages in 2022. Are you interested in comparing this with berry fruit trends in a particular country?"
    - "You're analyzing sugar flavors in North America. Can you specify the product category, like hot beverages or snacks, for a more detailed comparison?"
    - "Given the prevalence of berry fruit flavors in the UK, would you like to explore the seasonal trends in different months?"

    **Now, generate a single well-formed, personalized follow-up question for the user.**
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="What is one dynamic follow-up question I can ask?")
    ]

    response = llm(messages)
    return response.content


conversation_memory = []
def update_memory(user_query, structured_query, follow_up_questions, user_response=None):
    """
    Stores past interactions in memory.
    """
    memory_entry = {
        "user_query": user_query,
        "structured_query": structured_query,
        "follow_up_questions": follow_up_questions,
        "user_response": user_response
    }
    conversation_memory.append(memory_entry)


def agent1_process(user_query):
    """
    Agent1 workflow:
    1. Extracts structured data from the query.
    2. Retrieves relevant past queries from FAISS RAG.
    3. Generates conversational follow-up questions.
    4. Iteratively refines the query until complete.
    """
    
    # Step 1: Extract initial structured query
    structured_query = extract_query_details_llm(user_query)

    while True:
        # Step 2: Retrieve similar past queries
        similar_queries = retrieve_similar_queries(user_query)

        # Step 3: Generate a conversational follow-up question
        follow_up_question = generate_conversational_follow_up(structured_query, similar_queries)

        # If no more follow-ups are needed, break
        if not follow_up_question:
            break  

        print("\nAgent1:", textwrap.fill(follow_up_question, width=100))

        # Step 4: Capture user response
        user_response = input("User Response: ")

        # Step 5: Extract new details from user response
        updated_data = extract_query_details_llm(user_response)

        # Step 6: Fill in missing fields with user-provided info
        for key, value in updated_data.items():
            if structured_query[key] is None and value is not None:
                structured_query[key] = value

        # Step 7: Update memory with new details
        update_memory(user_query, structured_query, follow_up_question, user_response)

    # Final Structured Query (Ready for Agent2)
    print("\nFinal Structured Query JSON:")
    print(json.dumps(structured_query, indent=2))
    return structured_query


from functions import compare_flavors

def map_query_to_function(structured_query, df=None, available_years=None):
    """
    Maps the structured query from the agent to the correct function and its parameters.
    Handles missing values, ensures correct types, and calls the appropriate function.

    Parameters:
    structured_query (dict): Extracted query details from the agent.
    df (pd.DataFrame): The DataFrame containing flavor data.
    available_years (list): List of available years in the dataset.

    Returns:
    None: Calls the correct function.
    """
    
    # Define available years if not provided
    if available_years is None:
        available_years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

    # Ensure df is provided
    if df is None:
        return "Error: DataFrame (df) is required but not provided."

    # Normalize keys to lowercase for consistency
    structured_query = {k.lower(): v for k, v in structured_query.items()}  

    # Extract details from query
    selected_function = structured_query.get("function")
    year = structured_query.get("year")
    country = structured_query.get("country")
    product_categories = structured_query.get("product category")
    flavors = structured_query.get("flavor category")

    # Debugging: Print structured query
    print("\nDebugging: Structured Query Received:")
    print(structured_query)

    # Check for missing values
    missing_keys = [key for key, value in structured_query.items() if value is None]
    if missing_keys:
        print(f"Warning: Missing values detected in {missing_keys}")

    # Ensure product categories are always a list
    if isinstance(product_categories, str):
        product_categories = [product_categories]  # Convert single value to list
    elif product_categories is None:
        product_categories = []  # Default to an empty list

    # Ensure "Flavor Category" is always a list
    if isinstance(flavors, str):
        flavors = [flavor.strip() for flavor in flavors.replace("&", " and ").replace(",", " and ").split(" and ")]
    elif flavors is None:
        flavors = []  # Default to an empty list

    # Handle year range dynamically
    if year:
        # Expand year range (2 years before, 1 year after, within dataset limits)
        year_range = tuple(y for y in range(year - 2, year + 2) if y in available_years)
    else:
        year_range = None  # No filtering on year if not specified

    # Prepare function arguments
    function_args = {
        "df": df,
        "predictor1": "product_category",
        "predictor2": "year",  # Keeping "year" as predictor2
        "target": "prevalence_flavor",
        "color_by": "flavor_category",
        "category_list": product_categories if product_categories else None,
        "year_range": year_range if year_range else None,  
        "country_list": [country] if country else None,
        "flavor_list": flavors if flavors else None, 
    }

    # Debugging: Print final function arguments
    # print("\nDebugging: Function Arguments Prepared:")
    # print(function_args)

    # Run the correct function
    if selected_function in ["compare_flavors"]:
        return compare_flavors(**function_args)
    else:
        return f"Error: No function mapped for {selected_function}"



user_query = "Show me flavors in Australia in Jan 2023"
final_query = agent1_process(user_query)
map_query_to_function(final_query, df)




