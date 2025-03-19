# **Mintel GenAI: GenAI Chatbot for Flavor Analysis**  


## **Project Overview**  
This project focuses on developing a **GenAI-powered pipeline** for **exploratory analysis of product attributes** using **Mintel’s GNPD data**. The goal is to provide an intuitive and interactive way for users to analyze trends, discover replacement flavors, and identify key clusters in the dataset.  

Through **automated query processing, structured analysis functions, and machine learning models**, this pipeline enables more effective decision-making for flavor innovation and market analysis.  

---

## **Pipeline Overview**  

The system is designed to handle **user input dynamically**, process it using **Agent 1 and Agent 2**, and generate insights via **various analytical tools**. The pipeline consists of the following components:  

1. **Agent 1** – Processes user input, extracts structured data, and stores memory for iterative refinement.  
2. **Agent 2** – Interprets the structured query, maps it to the appropriate function, retrieves relevant data, and triggers analytical processing.  
3. **Analysis Functions** – Includes trend analysis, replacement flavor analysis, and clustering-based segmentation.  
4. **Reports and Outputs** – Generates **interactive visualizations** and **statistical summaries** to support strategic decision-making.  

---
## **Agent Introduction**  

Our pipeline is built around two core agents: **Agent 1** and **Agent 2**, which work together to process user queries, extract structured insights, and generate relevant analytical outputs. These agents facilitate the interaction between users and our analytical tools, ensuring efficient data retrieval and processing.

### **Agent 1: Query Processing & Memory Handling**  
- **Function:** Extracts structured data from user queries and refines inputs iteratively.  
- **Key Responsibilities:**  
  - Receives **natural language input** from the user.  
  - Uses **LLM-powered extraction** to convert the query into a structured JSON format.  
  - Stores **user context** for multi-turn conversations, allowing follow-up queries to be processed with context awareness.  
  - Passes the structured query to **Agent 2** for execution.  

**Example Workflow:**  
1. User inputs:  
   > "Show me the trend of plant flavors in food from 2016 to 2020."  
2. Agent 1 processes and converts it to:  
   ```json
   {
     "Year": [2016, 2017, 2018, 2019, 2020],
     "Product Category": ["Food"],
     "Flavor Category": ["Plant"],
     "Function": ["analyse_flavor_trend"]
   }
   ```

### **Agent 2: Table Extracting, Analytical Toolkit Calling, and Report Analysis Generating**  
This part of the project includes two main classes:
- Table_Extraction: A data processing class that loads, merges, cleans, and computes flavor prevalence from product data.
- Agent2: A function-execution framework that processes user requests, executes analytical functions, and generates automated reports using AI.

#### **1. Table Extraction**
The Table_Extraction class performs the following key operations:
- Loads CSV files (countries, products, flavors)
- Merges datasets based on foreign keys (c_id, p_id)
- Prepares features (extracts year, month, year-month from dates)
- Computes flavor prevalence per region, time period, and category
- Handles text standardization (lowercase conversion)
- Filters data dynamically based on JSON queries
##### Initialization
``
table_extractor = Table_Extraction("countries.csv", "flavors.csv", "products.csv")
``
##### Access Processed Data
``
df = table_extractor.get_dataframe()
print(df.head())
``
##### Process a Query
``
filtered_data = table_extractor.process_request(query)
print(filtered_data.head())
``
#### **2. Agent 2**
The Agent2 class automates function execution and report generation:
- Executes predefined analytical functions (stat_summary, flavor_state, generate_plot)
- Loads and filters data dynamically from extracted_data.csv
- Finds best matches when exact filtering values are unavailable
- Stores function outputs for later report generation
- Generates AI-powered analysis reports using GPT-3.5-turbo
  
##### Initialization
``
agent = Agent2()
``
##### Execute the specified Function in Query
``
output = agent.process_request(request_json)
print(output)
``
##### Process a Query
``
analysis_report = agent.generate_analysis_report(report_request)
print(analysis_report)
``



## **Function Introduction**  

### **1. Trend Analysis**  
- **Function Name:** `analyse_flavor_trend`  
- **Key Questions Answered:**  
  - Which flavors have shown the highest growth in popularity over time?  
  - Are there seasonal patterns in the popularity of specific flavors?  
  - Which flavor categories are declining in popularity?  
  - How do trend scores change for different flavors over multiple years?  

**Methodology:**  
- Uses **trend scores** to measure the rise or fall of specific flavors over time.  
- Identifies **seasonality** and **market trends** using **time series analysis**.  
- Detects **flavor categories with sustained growth or decline** to inform product innovation.  

---

### **2. Replacement Flavor Analysis**  
- **Function Name:** `find_top_replacements`  
- **Key Questions Answered:**  
  - What are the best alternative flavors for a given base flavor?  
  - Which flavors are most frequently interchangeable in product formulations?  
  - Do certain flavor categories have a wider range of replacement options?  

**Methodology:**  
- Uses **similarity scores** to identify top flavor replacements.  
- Highlights **interchangeable flavor pairs** commonly found in formulations.  
- Determines **versatility of flavors** across different product categories.  

---

### **3. Clustering and Segmentation Analysis**  
- **Function Name:** `state_flavor_analysis`  
- **Key Questions Answered:**  
  - How are flavors naturally grouped based on their characteristics?  
  - Do specific regions have distinct flavor clusters?  
  - Are certain flavor segments associated with specific product supercategories?  

**Methodology:**  
- Uses **unsupervised learning (e.g., K-Means, hierarchical clustering)** to find natural groupings.  
- Analyzes **regional variations in flavor preferences**.  
- Identifies **key flavor categories** that dominate in different product types.  

---

## **Future Improvements & Next Steps**  

### **1. Enhancing Readability & Documentation**  
- Add **descriptive comments** for each code block to clarify logic and functionality.  
- Provide **clear function descriptions** and **examples of expected outputs**.  

### **2. Improving Conciseness & Code Reusability**  
- Implement **modular functions** for common tasks such as **data loading, preprocessing, and visualization**.  
- Develop **reusable templates** for repetitive operations.  
- Include **summary statistics** at the end of major steps, such as:  
  > “After preprocessing, the dataset contains 2,281,370 product entries across 6 categories.”  

### **3. Boosting Efficiency & Visualization Quality**  
- Use **optimized data structures** for faster computation.  
- Standardize **plot aesthetics** (e.g., size, font, style) for consistency across reports.  
- Implement **interactive visualizations** using **Plotly, Seaborn, and Matplotlib**.  



## **How to Use This Project**  

### **1. Install Dependencies**  
```bash
pip install pandas numpy plotly statsmodels langchain faiss-cpu
```

### **2. Running the Demo**
To quickly see the project in action, navigate to the `demo` folder and run the `demo2.py` script.  

Steps to Run the Demo: 
1. **Open a terminal or command prompt.**  
2. **Navigate to the `demo` directory:**  
   ```bash
   cd demo
   python demo2.py
   ```


The script will process a sample query, interact with Agent 1 and Agent 2, and display the output, including:
- Query Processing: Converting a natural language query into structured JSON.
- Function Execution: Mapping the query to the appropriate analytical function.
- Data Retrieval & Visualization: Generating relevant insights and plots.



### **Project Members:**  
- **Runxuan Li**  
- **Ruyi Lu**  
- **Truong Vo**  
- **Bo Zhao**  
