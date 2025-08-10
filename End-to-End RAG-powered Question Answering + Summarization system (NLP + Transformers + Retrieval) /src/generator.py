from __future__ import annotations

"""
Text generation module for RAG system.
Handles question answering and summarization using Hugging Face models.
"""

import logging
from typing import List, Dict, Any, Optional
import re

# Import will be available after installation completes
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    from langchain.llms import HuggingFacePipeline
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain.schema import Document
except ImportError:
    print("Warning: transformers and langchain not yet installed")
    pipeline = None
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None
    HuggingFacePipeline = None
    PromptTemplate = None
    LLMChain = None
    Document = None  # Fallback so annotations and runtime checks don't crash

logger = logging.getLogger(__name__)


class RAGGenerator:
    """Text generator for RAG question answering and summarization."""
    
    def __init__(self, 
                 model_name: str = "google/flan-t5-small",
                 max_length: int = 512,
                 temperature: float = 0.1):
        """
        Initialize the RAG generator.
        
        Args:
            model_name: Hugging Face model name for generation
            max_length: Maximum length of generated text
            temperature: Temperature for generation (lower = more deterministic)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.temperature = temperature
        self.pipeline = None
        self.llm = None
        self.qa_chain = None
        self.summarization_chain = None
        
        # Initialize model if available
        if pipeline is not None:
            self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the generation model and chains."""
        try:
            logger.info(f"Loading generation model: {self.model_name}")
            
            # Create Hugging Face pipeline (seq2seq model expected)
            self.pipeline = pipeline(
                "text2text-generation",
                model=self.model_name,
                max_length=self.max_length,
                temperature=self.temperature,
                do_sample=True if self.temperature > 0 else False
            )
            
            # Create LangChain LLM wrapper
            self.llm = HuggingFacePipeline(pipeline=self.pipeline)
            
            # Create QA chain
            self._create_qa_chain()
            
            # Create summarization chain
            self._create_summarization_chain()
            
            logger.info("Generation model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    def _create_qa_chain(self):
        """Create a question-answering chain with prompt template."""
        qa_template = """You are a helpful assistant that answers questions based on the provided context. 
Use only the information from the context to answer the question. If the answer cannot be found in the context, say "I don't know based on the provided context."

Context:
{context}

Question: {question}

Answer:"""
        
        qa_prompt = PromptTemplate(
            template=qa_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = LLMChain(
            llm=self.llm,
            prompt=qa_prompt,
            verbose=False
        )
    
    def _create_summarization_chain(self):
        """Create a summarization chain with prompt template."""
        summarization_template = """Summarize the following text in a clear and concise manner. 
Focus on the main points and key information.

Text to summarize:
{text}

Summary:"""
        
        summarization_prompt = PromptTemplate(
            template=summarization_template,
            input_variables=["text"]
        )
        
        self.summarization_chain = LLMChain(
            llm=self.llm,
            prompt=summarization_prompt,
            verbose=False
        )
    
    def answer_question(self, 
                       question: str, 
                       retrieved_docs: List[Document],
                       max_context_length: int = 2000) -> Dict[str, Any]:
        """
        Generate an answer to a question using retrieved documents.
        
        Args:
            question: The question to answer
            retrieved_docs: List of retrieved documents
            max_context_length: Maximum length of context to use
            
        Returns:
            Dictionary containing answer and metadata
        """
        if self.qa_chain is None:
            raise RuntimeError("QA chain not initialized")
        
        if not retrieved_docs:
            return {
                'answer': "I don't have any relevant information to answer this question.",
                'sources': [],
                'confidence': 0.0
            }
        
        # Prepare context from retrieved documents
        context = self._prepare_context(retrieved_docs, max_context_length)
        
        try:
            # Generate answer
            response = self.qa_chain.run(
                context=context,
                question=question
            )
            
            # Clean up the response
            answer = self._clean_response(response)
            
            # Extract source information
            sources = [
                {
                    'content': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    'metadata': doc.metadata
                }
                for doc in retrieved_docs
            ]
            
            # Simple confidence estimation based on answer length and content
            confidence = self._estimate_confidence(answer, context)
            
            return {
                'answer': answer,
                'sources': sources,
                'confidence': confidence,
                'context_used': context[:500] + "..." if len(context) > 500 else context
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                'answer': "I encountered an error while generating the answer.",
                'sources': [],
                'confidence': 0.0
            }
    
    def summarize_documents(self, 
                          documents: List[Document],
                          max_input_length: int = 3000) -> Dict[str, Any]:
        """
        Generate a summary of the provided documents.
        
        Args:
            documents: List of documents to summarize
            max_input_length: Maximum length of input text
            
        Returns:
            Dictionary containing summary and metadata
        """
        if self.summarization_chain is None:
            raise RuntimeError("Summarization chain not initialized")
        
        if not documents:
            return {
                'summary': "No documents provided for summarization.",
                'num_documents': 0
            }
        
        # Combine document content
        combined_text = "\n\n".join([doc.page_content for doc in documents])
        
        # Truncate if too long
        if len(combined_text) > max_input_length:
            combined_text = combined_text[:max_input_length] + "..."
        
        try:
            # Generate summary
            summary = self.summarization_chain.run(text=combined_text)
            
            # Clean up the response
            summary = self._clean_response(summary)
            
            return {
                'summary': summary,
                'num_documents': len(documents),
                'input_length': len(combined_text)
            }
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return {
                'summary': "I encountered an error while generating the summary.",
                'num_documents': len(documents)
            }
    
    def _prepare_context(self, documents: List[Document], max_length: int) -> str:
        """Prepare context string from retrieved documents."""
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(documents):
            doc_text = f"Document {i+1}:\n{doc.page_content}\n"
            
            if current_length + len(doc_text) > max_length:
                # Truncate the last document if needed
                remaining_length = max_length - current_length
                if remaining_length > 100:  # Only add if there's meaningful space
                    truncated_text = doc_text[:remaining_length] + "..."
                    context_parts.append(truncated_text)
                break
            
            context_parts.append(doc_text)
            current_length += len(doc_text)
        
        return "\n".join(context_parts)
    
    def _clean_response(self, response: str) -> str:
        """Clean and format the generated response."""
        # Remove extra whitespace
        response = response.strip()
        
        # Remove common artifacts
        response = re.sub(r'\n+', '\n', response)
        response = re.sub(r' +', ' ', response)
        
        return response
    
    def _estimate_confidence(self, answer: str, context: str) -> float:
        """Estimate confidence in the generated answer."""
        # Simple heuristic-based confidence estimation
        confidence = 0.5  # Base confidence
        
        # Increase confidence if answer is substantial
        if len(answer) > 50:
            confidence += 0.2
        
        # Increase confidence if answer doesn't contain uncertainty phrases
        uncertainty_phrases = ["i don't know", "not sure", "unclear", "cannot determine"]
        if not any(phrase in answer.lower() for phrase in uncertainty_phrases):
            confidence += 0.2
        
        # Decrease confidence if answer is very short
        if len(answer) < 20:
            confidence -= 0.3
        
        # Ensure confidence is between 0 and 1
        return max(0.0, min(1.0, confidence))


def create_generator(model_name: str = "google/flan-t5-small") -> RAGGenerator:
    """
    Create a RAG generator with the specified model.
    
    Args:
        model_name: Hugging Face model name
        
    Returns:
        Configured RAGGenerator instance
    """
    return RAGGenerator(model_name=model_name)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Create generator
        generator = create_generator()
        
        # Sample documents for testing
        if Document is not None:
            sample_docs = [
                Document(
                    page_content="Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.",
                    metadata={'source': 'ml_guide.txt'}
                ),
                Document(
                    page_content="Deep learning uses neural networks with multiple layers to automatically learn representations from data.",
                    metadata={'source': 'dl_guide.txt'}
                )
            ]
            
            # Test question answering
            question = "What is machine learning?"
            result = generator.answer_question(question, sample_docs)
            
            print(f"Question: {question}")
            print(f"Answer: {result['answer']}")
            print(f"Confidence: {result['confidence']:.2f}")
            print("---")
            
            # Test summarization
            summary_result = generator.summarize_documents(sample_docs)
            print(f"Summary: {summary_result['summary']}")
        else:
            print("Dependencies not installed. Skipping example run.")
            
    except Exception as e:
        print(f"Error (expected if dependencies not installed): {e}")