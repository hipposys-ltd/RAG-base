import chromadb
import os

from langchain.schema import Document
from langchain_chroma import Chroma as LangChroma
from typing import Iterable, Optional
from urllib.parse import urlparse

from app.databases.vector.base import BaseVectorDatabase
from app.utils.logger import Logger


class Chroma(BaseVectorDatabase, LangChroma):

    FILE_SYSTEM_SCHEMA = 'fs://'
    SERVER_SCHEMA = 'http://'
    SERVER_SECURE_SCHEMA = 'https://'

    def get_chroma_http_client(self, uri: str) -> chromadb.HttpClient:
        """Parse `uri` and return an HTTP client object for the Chroma database."""
        
        # Parse the URI.
        parsed_uri = urlparse(uri)
        host = parsed_uri.hostname
        port = parsed_uri.port
        username = parsed_uri.username
        password = parsed_uri.password
    
        # Construct and return the client object.
        Logger().get_logger().debug(f'Connecting to Chroma database at {host}:{port}')
        return chromadb.HttpClient(
            host=host,
            port=port,
            settings=chromadb.config.Settings(
                chroma_client_auth_provider='chromadb.auth.basic_authn.BasicAuthClientProvider',
                chroma_client_auth_credentials=f'{username}:{password}',
            ),
        )
    
    def __init__(self, drop_old: Optional[bool] = False, **kwargs):
        """Initialize a Chroma database client.
        
        The `CHROMA_DB_URI` environment variable should be set to the URI of the Chroma database.

        Supported URI schemas:
        - `http://`: Connect to a remote Chroma database.
        - `fs://`: Connect to a local file system database.
        """

        self.client = None

        # Parse the connection URI and set the HTTP client or local FS directory, accordingly.
        chroma_db_uri = os.environ.get('CHROMA_DB_URI', 'http://localhost:6000')
        if chroma_db_uri.startswith(self.SERVER_SCHEMA) or chroma_db_uri.startswith(self.SERVER_SECURE_SCHEMA):
            # HTTP client.
            self.client = self.get_chroma_http_client(chroma_db_uri)
        elif chroma_db_uri.startswith(self.FILE_SYSTEM_SCHEMA):
            # Local file system.
            self.client = chromadb.Client(chromadb.config.Settings(
                is_persistent=True,
                persist_directory=chroma_db_uri[len(self.FILE_SYSTEM_SCHEMA):],
            ))
        else:
            # Invalid schema.
            raise ValueError(f'Invalid CHROMA_DB_URI Schema: {chroma_db_uri}')

        kwargs['client'] = self.client

        # Drop the existing collection, if requested and exists.
        if drop_old:
            collection_name = kwargs.get('collection_name', self.get_default_collection_name())
            self._drop_collection(collection_name, ignore_non_exist=True)

        super().__init__(**kwargs)

    async def delete_embeddings(self, source_id: str) -> dict:
        """Delete the embeddings for the given text from the Chroma database."""
        
        # Must use the "low-level" API to delete using a condition (and not by ID).
        collection = self.client.get_collection(self.collection_name)
        collection.delete(where={'source_id': source_id})

        # Chroma DB doesn't provide statistics on deletion.
        return {
            'success': True,
            'error_count': 0,
        }

    def _drop_collection(self, collection_name: str, ignore_non_exist: bool = False) -> None:
        """Drop the collection from the Chroma database."""
        try:
            self.client.delete_collection(collection_name)
        except chromadb.errors.InvalidArgumentError as e:
            # Silence the error if the collection doesn't exist and we're asked to ignore it.
            if ignore_non_exist and f'Collection {collection_name} does not exist' in str(e):
                pass
            else:
                raise e

    async def drop_collection(self, collection_name: str, ignore_non_exist: bool = False) -> None:
        """Drop the collection from the Chroma database."""
        self._drop_collection(collection_name, ignore_non_exist)

    def add_documents(self, documents: Iterable[Document]) -> list[str]:
        """Override the general `add_documents` in order to convert `metadata.payload` to string.
        
        The problem is that `Chroma` doesn't accept dictionary metadata values.
        """

        def payload_to_str(doc: Document) -> Document:
            """Convert the `payload` field in the metadata to a string."""
            doc.metadata['payload'] = str(doc.metadata.get('payload', {}))
            return doc

        # Convert the `payload` field in the metadata to a string.
        documents = (payload_to_str(doc) for doc in documents)

        # Call the parent method.
        # We Convert to `list` because `add_documents` doesn't support a generator.
        return super().add_documents(list(documents))
