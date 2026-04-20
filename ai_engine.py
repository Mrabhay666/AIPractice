import os
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain

class AIEngine:
    def __init__(self, api_key):
        self.llm = ChatGroq(        # Change this
            model="llama-3.3-70b-versatile", 
            temperature=0.2, 
            groq_api_key=api_key    # Change this
        )

    def generate_narrative_report(self, stats, anomalies):
        template = """
        You are a Senior Data Scientist. Analyze the following dataset summary and write a professional EDA report.
        
        Dataset Metadata: {stats}
        Anomalies/Outliers Detected: {anomalies}
        
        Your report should include:
        1. Executive Summary (What is this data about?)
        2. Data Quality Observation (Missing values, outliers, etc.)
        3. Key Statistical Insights (Notable correlations or distributions).
        4. Strategic Recommendations for Business Stakeholders.
        
        Keep the tone professional and concise. Use Markdown for formatting.
        """
        prompt = PromptTemplate(input_variables=["stats", "anomalies"], template=template)
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        return chain.run(stats=stats, anomalies=anomalies)