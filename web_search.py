import os
from typing import List, Optional
from exa_py import Exa
from datetime import datetime
import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tenacity import retry, stop_after_attempt, wait_exponential

from workflow.states import State
from config import Settings


logger = logging.getLogger(__name__)


class exa_ai:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('EXA_AI_API_KEY')

        # Validate the API key
        if not self.api_key:
            logger.warning("Exa AI API key not provided; web search will be disabled unless configured later.")

        try:
            self.client = Exa(self.api_key)
        except Exception as e:
            logger.error(f"Failed to initialize Exa AI client: {e}")
            raise ValueError("Invalid Exa AI API key or configuration error")

    def _is_api_enabled(self) -> bool:
        if not self.api_key:
            logger.error("Exa AI API key is missing.")
            return False
        return True

    @staticmethod
    def _gather_metadata(state: State, query: str, url: str, source_title: str, retrieved: datetime,
                         chunk_index: int | None = None, total_chunks: int | None = None):
        metadata = {
            'user_id': state.user_id,
            'thread_id': state.thread_id,
            'search_depth': state.search_depth,
            'url': url,
            'retrieved_datetime': retrieved,
            'source_title': source_title,
            'chunk_index': chunk_index,
            'total_chunks': total_chunks,
            'highlight': False,
            'original_query': state.question,
            'optimised_query': query,
        }
        return metadata

    @staticmethod
    def _create_doc_id(state, url, chunk_index, highlight_idx):
        return 'web_' + state.user_id + '_' + state.thread_id + '_' + str(hash(url)) + '_' + str(chunk_index) + '_' + str(highlight_idx)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def simple_search(self, state: State) -> List[Document]:
        """
        Perform a simple search using Exa AI's answer endpoint.

        Args:
            state (State): The state object containing user queries and metadata.

        Returns:
            A list of Document objects containing search results.
        """
        try:
            logger.debug(f'exa_ai simple_search input: {state.optimised_queries}')
            if not self._is_api_enabled() or not state.optimised_queries:
                return []

            retrieved = datetime.now()
            results = []
            for query in state.optimised_queries:
                query_result = self.client.answer(query, text=False)
                url = query_result.citations[0].id
                metadata = self._gather_metadata(
                    state=state, query=query,
                    url=url,
                    source_title=query_result.citations[0].title,
                    retrieved=retrieved
                )
                print(state.user_id, state.thread_id)
                doc = Document(
                    id=self._create_doc_id(state, url, chunk_index='', highlight_idx=''),
                    page_content=query_result.answer,
                    metadata=metadata
                )

                results.append(doc)
            logger.debug(f'Simple search results:, {results}')

            return results
        except Exception as e:
            logger.error(f"Simple search failed for queries '{state.optimised_queries}': {e}")
            return []

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def perform_web_search(self, state: State) -> List[Document]:
        """
        Perform a comprehensive web search using Exa AI API.

        Args:
            state (State): State object.

        Returns:
            List[Document]: List of formatted search results in documents.
        """
        logger.debug(f'exa_ai perform_web_search input: {state.optimised_queries}')
        if not self._is_api_enabled() or not state.optimised_queries:
            return []

        text = True if state.search_depth == 'high' else False

        search_options = {
            "num_results": Settings.web_search_max_results,
            "use_autoprompt": True,
            "text": text,
            "livecrawl": "auto",
            "highlights": {
                "num_sentences": 5,
                "highlights_per_url": Settings.web_search_highlights_per_url,
            }
        }

        try:
            results = []
            retrieved = datetime.now()
            for query in state.optimised_queries:
                query_result = self.client.search_and_contents(query, **search_options)
                for source in query_result.results:
                    metadata = self._gather_metadata(
                        state=state, query=query,
                        url=source.url,
                        source_title=source.title,
                        retrieved=retrieved
                    )
                    if source.highlights:
                        metadata['highlight'] = True
                        for highlight_idx, highlight in enumerate(source.highlights):
                            doc = Document(
                                    id=self._create_doc_id(state, source.url, chunk_index='', highlight_idx=highlight_idx),
                                    page_content=highlight,
                                    metadata=metadata
                            )
                            results.append(doc)
                        metadata['highlight'] = False

                    if source.text and state.search_depth == 'high':
                        logger.debug(f"Exa AI search depth high returned source.text")
                        doc = Document(
                            id=None,
                            page_content=source.text,
                            metadata=metadata
                        )
                        # chunk full text of webpages for indexing
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
                        split_docs = text_splitter.split_documents([doc])

                        if not split_docs:
                            return results

                        # Add chunk_index to each split document
                        for i, split_doc in enumerate(split_docs):
                            chunk_index = i + 1
                            split_doc.metadata["chunk_index"] = chunk_index
                            split_doc.metadata["total_chunks"] = len(split_docs)
                            doc = Document(
                                id=self._create_doc_id(state, source.url, chunk_index=chunk_index, highlight_idx=''),
                                page_content=split_doc.page_content,
                                metadata=split_doc.metadata
                            )
                            results.append(doc)

            logger.debug(f"Exa AI returned {len(results)} results for queries: {state.optimised_queries}")
            return results

        except Exception as e:
            logger.error(f"Web search failed for query '{state.optimised_queries}': {e}")
            return []
