import logging
from typing import List, Dict, Any, BinaryIO, Optional
import pandas as pd
import json
import io

from langchain.schema import Document

from app.services.parsers.base import BaseParser
from app.services.chunking_service import chunking_service

logger = logging.getLogger(__name__)

class StructuredDataParser(BaseParser):
    """Parser for structured data formats (CSV, Excel, JSON)"""
    
    def __init__(self):
        super().__init__()
        self.supported_mimetypes = [
            "text/csv",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/json"
        ]
        # Use centralized chunking service
        self.chunking_service = chunking_service
    
    def _parse_csv(self, file_data: BinaryIO, metadata: Dict[str, Any]):
        """Parse CSV data"""
        try:
            # Load CSV into DataFrame
            df = pd.read_csv(file_data)
            
            # Add metadata
            metadata["format"] = "csv"
            metadata["columns"] = df.columns.tolist()
            metadata["row_count"] = len(df)
            
            return self._process_dataframe(df, metadata)
        
        except Exception as e:
            logger.error(f"Error parsing CSV: {str(e)}")
            raise
    
    def _parse_excel(self, file_data: BinaryIO, metadata: Dict[str, Any]):
        """Parse Excel data"""
        try:
            # Load Excel into DataFrame(s)
            excel_file = pd.ExcelFile(file_data)
            sheet_names = excel_file.sheet_names
            
            metadata["format"] = "excel"
            metadata["sheets"] = sheet_names
            
            documents = []
            
            # Process each sheet
            for sheet in sheet_names:
                df = excel_file.parse(sheet)
                
                sheet_metadata = metadata.copy()
                sheet_metadata["sheet"] = sheet
                # JSON metadata requires scalar, serializable column names.
                sheet_metadata["columns"] = [str(column) for column in df.columns]
                sheet_metadata["row_count"] = len(df)
                
                documents.extend(self._process_dataframe(df, sheet_metadata))
            
            return documents
        
        except Exception as e:
            logger.error(f"Error parsing Excel: {str(e)}")
            raise
    
    def _parse_json(self, file_data: BinaryIO, metadata: Dict[str, Any]):
        """Parse JSON data"""
        try:
            # Load JSON
            json_data = json.load(file_data)
            
            metadata["format"] = "json"
            
            # Convert to DataFrame if possible
            if isinstance(json_data, list) and len(json_data) > 0 and isinstance(json_data[0], dict):
                df = pd.DataFrame(json_data)
                metadata["row_count"] = len(df)
                metadata["columns"] = df.columns.tolist()
                return self._process_dataframe(df, metadata)
            
            # Otherwise, just use the raw JSON as text
            json_text = json.dumps(json_data, indent=2)
            
            document = Document(
                page_content=json_text,
                metadata=metadata
            )
            
            return self.chunking_service.split_documents([document])
        
        except Exception as e:
            logger.error(f"Error parsing JSON: {str(e)}")
            raise
    
    def _process_dataframe(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> List[Document]:
        """Convert non-empty table regions into compact, non-duplicated chunks."""
        compact_df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
        if compact_df.empty or len(compact_df.columns) == 0:
            return []

        compact_df = compact_df.copy()
        compact_df.columns = [str(column) for column in compact_df.columns]

        documents = []
        batch_size = 20
        sheet_name = metadata.get("sheet")

        for start in range(0, len(compact_df), batch_size):
            batch = compact_df.iloc[start:start + batch_size]
            end = min(start + batch_size - 1, len(compact_df) - 1)

            # CSV is substantially more compact than DataFrame.to_string for
            # wide financial sheets and preserves the header/value relationship.
            table_text = batch.to_csv(index=False, na_rep="")
            context_lines = []
            if sheet_name:
                context_lines.append(f"Worksheet: {sheet_name}")
            context_lines.append(f"Rows: {start}-{end}")

            batch_metadata = metadata.copy()
            batch_metadata["columns"] = list(compact_df.columns)
            batch_metadata["non_empty_row_count"] = len(compact_df)
            batch_metadata["row_range"] = f"{start}-{end}"
            batch_metadata["representation"] = "table_rows"

            documents.append(
                Document(
                    page_content="\n".join(context_lines) + "\n" + table_text,
                    metadata=batch_metadata,
                )
            )

        result = []
        for document in documents:
            result.extend(self.chunking_service.split_documents([document]))

        return result
    
    def parse(self, file_data: BinaryIO, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Parse structured data into documents
        
        Args:
            file_data: File-like object containing the structured data
            metadata: Optional metadata to include with the document
            
        Returns:
            List of Document objects
        """
        if metadata is None:
            metadata = {}
        
        mime_type = metadata.get("mime_type", "")
        
        if mime_type == "text/csv":
            return self._parse_csv(file_data, metadata)
        
        elif mime_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            return self._parse_excel(file_data, metadata)
        
        elif mime_type == "application/json":
            return self._parse_json(file_data, metadata)
        
        else:
            logger.error(f"Unsupported MIME type for structured parser: {mime_type}")
            raise ValueError(f"Unsupported MIME type: {mime_type}")


structured_data_parser = StructuredDataParser()
