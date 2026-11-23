from string import Template

class PromptManager:
    def __init__(self):
        self.system_prompt = """You are an expert Legal advisor with experience in framing agreements for organizations.  
Keep your response in natural language only. Perform the task provided.

Context: {context}"""
        
        self.prompt_templates = ["""
        Task: Using the scenario for data sharing activity and the context provided, generate content for the title '$title' in a data sharing agreement between two parties mentioned.
        
        To understand the title $title, use the following entity and properties
        
        $pname: $hint
        
        $triples

        Strictly follow the rules mentioned below:
        
        1. Answer concisely without closing statements or repetitive offers of help.
        2. Provide the output as simple text under the given title. 
        3. Do not leave any sentences incomplete.
        4. Do not generate any note in your response.
        5. Respond concisely with just the result, avoiding phrases like 'Here is the final output:', 'The estimated time required to generate the output', 'Please let me know'.
        6. Limit the text length as appropriate to the title provided. 
        7. Do not generate explanations.
        8. Do not include any additional information that are irrelevant to the title given. 
        9. Do not generate follow-up questions.
        10. The data sharing is governed by Data Oversight Committee.
        
        $ctext
        
        Generated Output:
        """,
        """
        Task: Using the scenario for data sharing activity and the context provided, generate content for the title '$title' in a data sharing agreement between two parties mentioned.
        
        To understand the title $title, use the following entity and properties
        
        ## Entity: 
        - Type/Class Description: $classes

        ## Properties:
        Properties: $pname : $hint
        
        $triples
        
        Strictly follow the rules mentioned below:
        
        1. Answer concisely without closing statements or repetitive offers of help.
        2. Provide the output as simple text under the given title. 
        3. Do not leave any sentences incomplete.
        4. Do not generate any note in your response.
        5. Respond concisely with just the result, avoiding phrases like 'Here is the final output:', 'The estimated time required to generate the output', 'Please let me know'.
        6. Limit the text length as appropriate to the title provided. 
        7. Do not generate explanations.
        8. Do not include any additional information that are irrelevant to the title given. 
        9. Do not generate follow-up questions.
        10. The data sharing is governed by Data Oversight Committee.
        
        $ctext
        
        Generated Output:
        """,
        """
        Task: Using the scenario for data sharing activity and the context provided, generate content for the title '$title' in a data sharing agreement between two parties mentioned.
        
        To understand the title $title, use the following entity and properties
        
        ## Entity: 
        - Type/Class Description: $classes

        ## Properties:
        Properties: $pname : $hint
        
        $triples

        Strictly follow the rules mentioned below:
        
        1. Answer concisely without closing statements or repetitive offers of help.
        2. Provide the output as simple text under the given title. 
        3. Do not leave any sentences incomplete.
        4. Do not generate any note in your response.
        5. Respond concisely with just the result, avoiding phrases like 'Here is the final output:', 'The estimated time required to generate the output', 'Please let me know'.
        6. Limit the text length as appropriate to the title provided. 
        7. Do not generate explanations.
        8. Do not include any additional information that are irrelevant to the title given. 
        9. Do not generate follow-up questions.
        10. The data sharing is governed by Data Oversight Committee.
        
        $ctext
        
        Generated Output:
        """
        ]

        self.prompt_template1 = """

        Scenario: $ctext
        
        Task: Using the scenario about data sharing activity, identify and list 25 section headings of all relevant sections to be included in the Data Sharing Agreement between two parties mentioned in the scenario. Strictly provide the response as list of identified section headings under the heading 'Section Headings:'.
    Generated Output:
    """

    def get_prompt(self, index, **kwargs):
        return Template(self.prompt_templates[index]).substitute(**kwargs)
    def get_title_prompt(self, **kwargs):
        return Template(self.prompt_template1).substitute(**kwargs)