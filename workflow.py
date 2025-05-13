import json
from typing import Dict, List
import aiosqlite
from langgraph.graph import StateGraph, START, END
# from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # USE THIS WHEN LANGGRAPH UPDATED
from AsyncSqlLiteSaver_hack import HackAsyncSqliteSaver
from langchain_core.messages import AIMessage
import logging
from math import floor

from web_search_api import exa_ai
from states import State, ChatResponse, WebSearchResult
from model_manager import LLMModelManager
from utils import CONFIG, chat_system_prompt, decide_web_search_system_prompt, query_optimiser_system_prompt, \
    summerisation_system_prompt

logger = logging.getLogger(__name__)

exa_ai = exa_ai()


async def create_workflow(shared_manager: LLMModelManager | None):
    """Creates and compiles a LangGraph workflow for processing conversational queries.

    The workflow includes nodes for recalling conversation history, calling the LLM,
    and summarizing the conversation. It uses SQLite for state persistence.

    Args:
        shared_manager (LLMModelManager): Shared instance of LLMModelManager for model inference.

    Raises:
        RuntimeError: If SQLite connection or checkpointer initialization fails.
    """
    logger.info(f"Creating workflow")

    try:
        # Initialize AsyncSQLite checkpointer
        conn = await aiosqlite.connect(CONFIG['paths']['checkpoint_db_path'])
        await conn.execute("PRAGMA journal_mode=WAL;")  # Enable Write-Ahead Logging
        await conn.execute("PRAGMA synchronous=NORMAL;")
        checkpointer = HackAsyncSqliteSaver(conn)
        logger.info(f"AsyncSqliteSaver initialized with connection to {CONFIG['paths']['checkpoint_db_path']}")
    except aiosqlite.Error as e:
        logger.error(f"Failed to initialize SQLite connection: {e}")
        raise RuntimeError(f"SQLite initialization failed: {str(e)}")

    workflow = StateGraph(state_schema=State)

    async def recall_conversation(state, config: dict) -> dict:
        """Calculate context length limits and other preprocessing"""
        thread_id = config["configurable"].get("thread_id", "unknown")
        logger.debug(f"Processing conversation for thread_id: {thread_id}")

        if state.conversation_LLM not in CONFIG["model"]["openrouter"]["available_models"]:
            raise 'recall_conversation: Model not in available models'

        if state.conversation_LLM == "meta-llama/llama-4-scout":
            context_window = 10000000
        elif state.conversation_LLM == "meta-llama/llama-4-maverick":
            context_window = 1000000
        elif state.conversation_LLM == "meta-llama/llama-3.1-405b-instruct":
            context_window = 128000
        else:
            raise 'recall_conversation: Model context window not defined'

        return {"context_window": context_window}

    async def retrieve_documents(state: State, config: dict) -> Dict[str, str]:
        """Node to retrieve user-uploaded and RAG documents related to the user prompt"""
        thread_id = config["configurable"].get("thread_id", "unknown")
        user_id = state.user_id

        # Calculate max token allowed for RAG and Session to stay within context length
        def top_k_calc(context_window_percentage: int, config_top_k: int):
            context_window_share = state.context_window * context_window_percentage
            max_chunks_to_retrieve = floor(context_window_share / CONFIG["embeddings"]["chunk_size"])
            top_k = min(max_chunks_to_retrieve, config_top_k)
            return top_k

        top_k_session = top_k_calc(
            context_window_percentage=CONFIG["embeddings"]["session_context_percentage"],
            config_top_k=CONFIG["embeddings"]["session_top_k"]
        )
        top_k_rag = top_k_calc(
            context_window_percentage=CONFIG["embeddings"]["rag_context_percentage"],
            config_top_k=CONFIG["embeddings"]["rag_top_k"]
        )
        logger.debug(f'top_k_session: {top_k_session}, top_k_rag: {top_k_rag}')

        vsm = shared_manager.vector_store_manager
        # Retrieve session documents
        session_docs_string, session_docs_agg = await vsm.get_documents(
            collection_name='SESSION',
            question=state.question,
            thread_id=thread_id,
            user_id=user_id,
            top_k=top_k_session,
            min_similarity=0.2
        )
        logger.info(f"Retrieved {len(session_docs_agg)} aggregated session documents for thread_id: {thread_id}")

        # RAG documents
        rag_docs_string, rag_docs_agg = await vsm.get_documents(
            collection_name='RAG',
            question=state.question,
            thread_id=thread_id,
            user_id=user_id,
            base_path=state.rag_path,
            top_k=top_k_rag,
            min_similarity=CONFIG["embeddings"]["min_similarity"]
        )
        logger.info(f"Retrieved {len(rag_docs_agg)} aggregated RAG docs for question: {state.question}")

        return {
            "rag_documents": rag_docs_string,
            "session_documents": session_docs_string,
        }

    async def decide_web_search(state: State, config: dict):
        """Determine if a web search is needed by prompting the model.

        Args:
            state: Current conversation state.
            config: Configuration with thread_id.

        Returns:
            Dict with web_search_needed flag.
        """
        thread_id = config["configurable"].get("thread_id", "unknown")
        logger.debug(f"Deciding if web search is needed for thread_id: {thread_id}, question: {state.question}")
        result = None  # Type: ignore
        try:
            decision_state = State(
                question=state.question,
                conversation_LLM=state.conversation_LLM,
                system_prompt=decide_web_search_system_prompt,
                rag_documents=state.rag_documents,
                session_documents=state.session_documents,
                web_search_results=state.web_search_results
            )
            result: ChatResponse = await shared_manager.call_model(state=decision_state, step='decide_web_search')
            logger.info(f"decide_web_search response: {result}")

            decision = json.loads(result.answer.strip().lower())
            logger.info(f"decide_web_search decision parsed: {decision}")

            web_search_needed: bool = decision.get("needs_web_search", False)
            depth = None
            if web_search_needed and decision.get("search_depth", None):
                depth = decision.get("search_depth").lower().strip()
                if depth not in ["low", "medium", "high"]:
                    logger.error(f"decide_web_search invalid depth: {depth}")
                    logger.info(f"Defaulting to 'low' depth")
                    depth = "low"

            return {"web_search_needed": web_search_needed, "search_depth": depth}
        except json.JSONDecodeError as e:
            logger.error(
                f"Web search decision failed with JSONDecodeError for thread_id: {thread_id}: {e}, raw: {result}")
            # reset web search between queries
            return {"web_search_needed": False, "search_depth": None, 'web_search_results': None}
        except Exception as e:
            logger.error(f"Web search decision failed for thread_id {thread_id}: {e}")
            return {"web_search_needed": False, "search_depth": None, 'web_search_results': None}

    async def query_optimiser(state: State, config: dict):
        """Split user query into strings suitable for web search"""

        # ADD PREVIOUS MESSAGES
        query_optimiser_state = State(
            question=state.question,
            conversation_LLM=state.conversation_LLM,
            older_message_summary=state.older_message_summary,
            system_prompt=query_optimiser_system_prompt
        )
        response: ChatResponse = await shared_manager.call_model(state=query_optimiser_state, step='query_optimiser')
        try:
            optimised_queries = json.loads(response.answer)
            logger.debug(f"query_optimiser optimised_queries: {optimised_queries}")
        except json.JSONDecodeError:
            logger.error(f"query_optimiser incorrect json format: {response.answer}")
            # fallback: use original input as single query
            optimised_queries = [state.question]

        return {"optimised_queries": optimised_queries}

    async def perform_web_search(state: State, config: dict) -> Dict[str, List[WebSearchResult]]:
        """Perform a web search if needed using exa.ai API. Add to existing web search results before validation.

        Args:
            state: Current conversation state.
            config: Configuration with thread_id.

        Returns:
            Dict with web search results.
        """
        thread_id = config["configurable"].get("thread_id", "unknown")
        logger.debug(f"Checking web search for thread_id: {thread_id}")

        if not state.web_search_needed:
            return {}

        results = []
        try:
            if state.search_depth == 'low':
                results = exa_ai.simple_search(queries=state.optimised_queries)
            elif state.search_depth == 'medium':
                results = exa_ai.perform_web_search(queries=state.optimised_queries, text=False)
            elif state.search_depth == 'high':
                results = exa_ai.perform_web_search(queries=state.optimised_queries, text=True)

            logger.debug(f"web_search results: {len(results)}")
            return {"web_search_results": state.web_search_results + results}
        except Exception as e:
            logger.error(f"Web search failed for thread_id {thread_id}: {e}")
            return {"web_search_results": state.web_search_results}

    async def high_depth_web_search_summary(state: State, config: dict):
        """Summarise the full text retrieved for hig-depth web searches and chunk into smaller pieces"""
        return {}

    async def rank_web_search_results(state: State, config: dict):
        """
        Use embeddings to rank web search results (plus previous web searches) against the original user prompt
        Web searches should be contained within the WEB_SEARCH collection in the chroma vector store so all web searches
        in the conversation can be recalled at any point in the future conversation if relevant
        """
        if not state.web_search_results:
            return {}

        model = shared_manager.vector_store_manager.embedding_model
        # Embed user prompt
        user_prompt = state.messages[-1]['content']
        user_embedding = model.encode(user_prompt, convert_to_tensor=True)

        # Loop to embed web results and give relevance score
        web_search_results_text, web_search_results_highlights = [], []
        for x in state.web_search_results:
            web_search_results_text.append(x.content)
            web_search_results_highlights.append(x.content)

        # Compute cosine similarity
        similarities = util.cos_sim(user_embedding, result_embeddings)[0]

        # Filter on top X results above a similarity threshold
        threshold = config.get("similarity_threshold", 0.6)
        top_k = config.get("top_k", 5)
        filtered_results = [
            result for result, score in results_with_scores if score >= threshold
        ]
        filtered_results = sorted(
            filtered_results,
            key=lambda x: util.cos_sim(user_embedding, model.encode(x["snippet"], convert_to_tensor=True)).item(),
            reverse=True
        )[:top_k]

        return {"web_search_results": filtered_results}

    async def retrieve_conversation_metadata(state: State, config: dict):
        """Get conversation metadata so the model can answer simple questions like 'What files have been uploaded?'"""

        # List uploaded files
        thread_id = config["configurable"].get("thread_id", "unknown")
        filter_dict = {"$and": [
            {"user_id": {"$eq": state.user_id}},
            {"thread_id": {"$eq": thread_id}}
        ]}

        # List files in uploaded to vector db
        uploaded_files = await shared_manager.vector_store_manager.list_documents(
            collection_name='SESSION',
            filter=filter_dict
        )
        uploaded_file_count = uploaded_files.count if uploaded_files.status == 'success' else 0
        uploaded_files = uploaded_files.files if uploaded_files.status == 'success' else []

        logger.info(f"retrieve_conversation_metadata output: {uploaded_files}")

        # Number of messages?
        # Number of internet searches (if implemented)
        # Username
        # Number of conversations in history for this user
        # Conversation length (minutes)?

        return {
            "system_information": {"uploaded_file_count": uploaded_file_count, "uploaded_files": uploaded_files}
        }

    async def summarise_older_messages(state: State, config: dict) -> Dict[str, str]:
        """Node to generate a summary of older messages in the conversation to keep prompt length shorter."""
        thread_id = config["configurable"].get("thread_id", "unknown")
        limit = CONFIG['conversation']['num_messages_pre_older_message_summary']

        if not state.messages or len(state.messages) < limit:
            logger.debug(f"Skipping older messages summary for thread_id: {thread_id}")
            return {"older_message_summary": state.older_message_summary or ""}

        logger.debug(f"Generating older messages summary for thread_id: {thread_id}")
        try:
            # select oldest messages
            older_messages = state.messages[:limit]
            recent_messages = state.messages[limit:]

            summary_state = State(
                older_message_summary=state.older_message_summary,
                messages=older_messages,
                conversation_LLM=state.conversation_LLM,
                system_prompt=summerisation_system_prompt
            )

            # Call the model to generate the summary
            result: ChatResponse = await shared_manager.call_model(state=summary_state, step='summarisation')
            summary = result.answer.strip()

            logger.debug(f'Summarisation result: {result}')

            logger.info(f"Generated summary for thread_id {thread_id}, new messages length: {len(recent_messages)}")
            # update older message summary and trimmed messages
            return {"older_message_summary": summary, 'messages': recent_messages}

        except Exception as e:
            logger.error(f"Failed to generate summary for thread_id {thread_id}: {e}")
            return {"older_message_summary": ""}

    async def call_model(state: State, config: dict) -> Dict[str, ChatResponse | List[AIMessage]]:
        """Calls the LLM to generate a response for the given state.

        Args:
            state (State): Current conversation state with messages, question, etc.
            config (dict): Configuration with thread_id.

        Returns:
            Dict[str, ChatResponse | List[AIMessage]]: Model response and updated messages.
        """
        thread_id = config["configurable"].get("thread_id", "unknown")
        logger.debug(f"Calling model for thread_id: {thread_id}")
        try:

            state_for_model = State(
                messages=state.messages,
                question=state.question,
                rag_documents=state.rag_documents,
                session_documents=state.session_documents,
                system_information=state.system_information,
                system_prompt=chat_system_prompt,
                conversation_LLM=state.conversation_LLM,
                web_search_results=state.web_search_results,
                older_message_summary=state.older_message_summary
            )
            logger.debug(f'workflow call_model web_search_results: {state.web_search_results}')
            result: ChatResponse = await shared_manager.call_model(state=state_for_model, step='question')
            return {
                "model_response": result,
                "messages": [AIMessage(content=result.answer)]
            }
        except Exception as e:
            logger.error(f"Model call failed for thread_id {thread_id}: {e}")
            return {
                "model_response": ChatResponse(
                    status="error",
                    reasoning="",
                    answer="Error: Unable to process request",
                    error=str(e)
                ),
                "messages": [AIMessage(content="Error: Unable to process request")]
            }

    def web_search_routing(search_depth: str):
        if search_depth == 'high':
            return "high_depth_web_search_summary"
        elif search_depth == 'medium':
            return "rank_web_search_results"
        else:  # low
            return "retrieve_conversation_metadata"

    # Add nodes
    workflow.add_node("recall_conversation", recall_conversation)
    workflow.add_node("retrieve_documents", retrieve_documents)
    workflow.add_node("decide_web_search", decide_web_search)
    workflow.add_node("query_optimiser", query_optimiser)
    workflow.add_node("perform_web_search", perform_web_search)
    workflow.add_node("high_depth_web_search_summary", high_depth_web_search_summary)
    workflow.add_node("rank_web_search_results", rank_web_search_results)
    workflow.add_node("retrieve_conversation_metadata", retrieve_conversation_metadata)
    workflow.add_node("summarise_older_messages", summarise_older_messages)
    workflow.add_node("model", call_model)

    # Add edges
    workflow.add_edge(START, "recall_conversation")
    workflow.add_edge("recall_conversation", "retrieve_documents")

    # Routing based on state.online
    workflow.add_conditional_edges(
        "retrieve_documents",
        lambda state: "decide_web_search" if state.online else "retrieve_conversation_metadata",
        {
            "decide_web_search": "decide_web_search",
            "retrieve_conversation_metadata": "retrieve_conversation_metadata"
        }
    )
    # Web search path
    workflow.add_conditional_edges(
        "decide_web_search",
        lambda state: "query_optimiser" if state.web_search_needed else "retrieve_conversation_metadata",
        {
            "query_optimiser": "query_optimiser",
            "retrieve_conversation_metadata": "retrieve_conversation_metadata"
        }
    )
    workflow.add_edge("query_optimiser", "perform_web_search")
    # edges based on web search depth
    workflow.add_conditional_edges(
        "perform_web_search",
        lambda state: web_search_routing(state.search_depth),
        {
            "high_depth_web_search_summary": "high_depth_web_search_summary",
            "rank_web_search_results": "rank_web_search_results",
            "retrieve_conversation_metadata": "retrieve_conversation_metadata"
        }
    )

    workflow.add_edge("high_depth_web_search_summary", "rank_web_search_results")
    workflow.add_edge("rank_web_search_results", "retrieve_conversation_metadata")

    # Common path
    workflow.add_edge("retrieve_conversation_metadata", "summarise_older_messages")
    workflow.add_edge("summarise_older_messages", "model")
    workflow.add_edge("model", END)

    # Compile with checkpointer
    app = workflow.compile(checkpointer=checkpointer)
    return app
