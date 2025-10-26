from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import hashlib
import uuid

class Schema2Chunks:

    def __init__(self,model_name="mistral-large-latest",chunk_size=500, chunk_overlap=50):

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def split_text(self,documents:dict,common_cols:dict):

        all_chunks = []

        meta_data = []

        for schame in documents:

            relation = ""

            ids = self._deterministic_uuid(schame['database']+schame['table_name'])

            for rels in common_cols:

                if (rels['database']== schame['database'] or rels['table_schema']== schame['table_schema']) and rels['table_name'] == schame['table_name']:

                    relation = rels['relation']

            chunks = self.splitter.split_text(schame['data_points'].strip())

            all_chunks.extend(chunks)

            for text in chunks:

                meta_data.append({"table_id":ids,"text_data":schame['data_points'],"common_columns":relation})

        return all_chunks, meta_data


    def _deterministic_uuid(self,content) -> str:
        """Creates deterministic UUID on hash value of string or byte content.
        Args:
            content: String or byte representation of data.
        Returns:
            UUID of the content.
        """
        if isinstance(content, str):
            content_bytes = content.encode("utf-8")
        elif isinstance(content, bytes):
            content_bytes = content
        else:
            raise ValueError(f"Content type {type(content)} not supported !")

        hash_object = hashlib.sha256(content_bytes)
        hash_hex = hash_object.hexdigest()
        namespace = uuid.UUID("00000000-0000-0000-0000-000000000000")
        content_uuid = str(uuid.uuid5(namespace, hash_hex))
        return content_uuid
    