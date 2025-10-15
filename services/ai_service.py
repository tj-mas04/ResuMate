"""
AI service for recommendations and chatbot.
"""
from groq import Groq
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_RECOMMENDATION_MODEL,
    GROQ_GPT5_MODEL,
    ENABLE_GPT5_FOR_ALL_CLIENTS,
)


class AIService:
    """Service for AI-powered features."""
    
    def __init__(self):
        """Initialize AI service with Groq client."""
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        self.client = Groq(api_key=GROQ_API_KEY)
        self.api_key = GROQ_API_KEY
        # Choose models based on feature flag
        self.chat_model = GROQ_GPT5_MODEL if ENABLE_GPT5_FOR_ALL_CLIENTS else GROQ_MODEL
        self.reco_model = GROQ_GPT5_MODEL if ENABLE_GPT5_FOR_ALL_CLIENTS else GROQ_RECOMMENDATION_MODEL
    
    def generate_recommendation(self, evaluation_data):
        """
        Generate tailored recommendations using LLM.
        
        Args:
            evaluation_data (dict): Resume evaluation results
            
        Returns:
            str: AI-generated recommendation
        """
        prompt = f"""
        You are an expert resume reviewer for ATS systems.

        Analyze the following resume evaluation details and give a short, actionable paragraph
        (3-5 sentences) suggesting how to improve the resume to better match the job description.

        Resume Details:
        - Similarity Score: {evaluation_data.get('similarity', 0)}%
        - ATS Score: {evaluation_data.get('ats_score', 0)}
        - Missing Keywords: {', '.join(evaluation_data.get('missing_keywords', []))}
        - Missing Skills: {', '.join(evaluation_data.get('missing_skills', []))}
        - Grammar Errors: {evaluation_data.get('grammar_errors', 0)}
        - Action Verbs Used: {evaluation_data.get('action_verbs_count', 0)}

        Give clear and specific feedback, e.g.:
        "Add more mentions of cloud frameworks and deployment tools."
        Avoid bullet points, give one cohesive paragraph.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.reco_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=180,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"(AI Error) Could not generate feedback: {str(e)}"
    
    def create_chatbot(self, context_info=""):
        """
        Create a conversational chatbot with context.
        
        Args:
            context_info (str): Context about uploaded documents
            
        Returns:
            tuple: (conversation_chain, memory)
        """
        # Initialize LLM
        try:
            llm = ChatGroq(
                temperature=0.7,
                model_name=self.chat_model,
                groq_api_key=self.api_key,
            )
        except Exception:
            # Fallback to stable default model if flagged model is unavailable
            llm = ChatGroq(
                temperature=0.7,
                model_name=GROQ_MODEL,
                groq_api_key=self.api_key,
            )
        
        # Conversation memory
        memory = ConversationBufferMemory()
        
        # Prompt template with context
        prompt = PromptTemplate(
            input_variables=["history", "input"],
            template=(
                "You are Resumate, an intelligent resume mentor bot. "
                "You analyze resumes and job descriptions and provide improvement feedback.\n"
                f"{context_info}\n"
                "Use the above job description and resume evaluation data to provide specific, "
                "contextual advice. Reference actual data points from the evaluations.\n\n"
                "Conversation history:\n{history}\n\n"
                "Human: {input}\nAI:"
            ),
        )
        
        # Create conversation chain
        conversation = ConversationChain(
            llm=llm,
            memory=memory,
            prompt=prompt,
            verbose=False,
        )
        
        return conversation, memory
