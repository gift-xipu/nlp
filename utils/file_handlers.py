"""
File handling utilities for various formats.
"""

import pandas as pd
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import chardet

class FileHandler:
    """Handles file I/O operations for various formats."""
    
    @staticmethod
    def detect_encoding(file_path: Union[str, Path]) -> str:
        """
        Detect file encoding.
        
        Args:
            file_path: Path to file
        
        Returns:
            Detected encoding string
        """
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
            return result['encoding'] or 'utf-8'
    
    @staticmethod
    def read_text_file(file_path: Union[str, Path], encoding: Optional[str] = None) -> str:
        """
        Read text file with encoding detection.
        
        Args:
            file_path: Path to text file
            encoding: Optional encoding (auto-detected if None)
        
        Returns:
            File contents as string
        """
        if encoding is None:
            encoding = FileHandler.detect_encoding(file_path)
        
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()
    
    @staticmethod
    def write_text_file(file_path: Union[str, Path], content: str, encoding: str = 'utf-8'):
        """Write content to text file."""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(content)
    
    @staticmethod
    def read_csv(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Read CSV file.
        
        Args:
            file_path: Path to CSV file
            **kwargs: Additional arguments for pd.read_csv
        
        Returns:
            DataFrame
        """
        encoding = kwargs.pop('encoding', None) or FileHandler.detect_encoding(file_path)
        return pd.read_csv(file_path, encoding=encoding, **kwargs)
    
    @staticmethod
    def write_csv(
        file_path: Union[str, Path],
        data: Union[pd.DataFrame, List[Dict]],
        **kwargs
    ):
        """
        Write data to CSV file.
        
        Args:
            file_path: Path to output CSV
            data: DataFrame or list of dictionaries
            **kwargs: Additional arguments for to_csv
        """
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(data, pd.DataFrame):
            data.to_csv(file_path, index=False, encoding='utf-8', **kwargs)
        else:
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False, encoding='utf-8', **kwargs)
    
    @staticmethod
    def read_json(file_path: Union[str, Path]) -> Any:
        """Read JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def write_json(file_path: Union[str, Path], data: Any, indent: int = 2):
        """Write data to JSON file."""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
    
    @staticmethod
    def read_excel(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Read Excel file."""
        return pd.read_excel(file_path, **kwargs)
    
    @staticmethod
    def write_excel(file_path: Union[str, Path], data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], **kwargs):
        """
        Write data to Excel file.
        
        Args:
            file_path: Path to output Excel file
            data: DataFrame or dict of DataFrames (for multiple sheets)
            **kwargs: Additional arguments for to_excel
        """
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(data, pd.DataFrame):
            data.to_excel(file_path, index=False, **kwargs)
        else:
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                for sheet_name, df in data.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    @staticmethod
    def export_lexicon(
        file_path: Union[str, Path],
        words: List[Dict],
        format: str = 'csv',
        metadata: Optional[Dict] = None
    ):
        """
        Export lexicon in specified format.
        
        Args:
            file_path: Output file path
            words: List of word dictionaries
            format: Export format ('csv', 'txt', 'json', 'xlsx')
            metadata: Optional metadata to include
        """
        file_path = Path(file_path)
        
        if format == 'csv':
            FileHandler.write_csv(file_path, words)
        
        elif format == 'txt':
            lines = []
            if metadata:
                lines.append(f"# Generated: {metadata.get('timestamp', datetime.now().isoformat())}")
                lines.append(f"# Language: {metadata.get('language', 'Unknown')}")
                lines.append(f"# Sentiment: {metadata.get('sentiment', 'Unknown')}")
                lines.append(f"# Total words: {len(words)}")
                lines.append("#" + "="*50)
                lines.append("")
            
            for word in words:
                if isinstance(word, dict):
                    word_str = f"{word.get('word', '')}: {word.get('translation', '')}"
                    if 'sentiment' in word:
                        word_str += f" [{word['sentiment']}]"
                    if 'score' in word:
                        word_str += f" (score: {word['score']:.2f})"
                    lines.append(word_str)
                else:
                    lines.append(str(word))
            
            FileHandler.write_text_file(file_path, '\n'.join(lines))
        
        elif format == 'json':
            export_data = {
                'metadata': metadata or {},
                'words': words,
                'count': len(words)
            }
            FileHandler.write_json(file_path, export_data)
        
        elif format == 'xlsx':
            df = pd.DataFrame(words)
            
            if metadata:
                # Create a multi-sheet Excel with metadata
                metadata_df = pd.DataFrame([metadata])
                FileHandler.write_excel(
                    file_path,
                    {'Lexicon': df, 'Metadata': metadata_df}
                )
            else:
                FileHandler.write_excel(file_path, df)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    @staticmethod
    def generate_filename(
        prefix: str,
        language: str,
        sentiment: Optional[str] = None,
        extension: str = 'csv'
    ) -> str:
        """
        Generate standardized filename.
        
        Args:
            prefix: Filename prefix
            language: Language name
            sentiment: Optional sentiment type
            extension: File extension
        
        Returns:
            Generated filename
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        parts = [prefix, language.lower()]
        
        if sentiment:
            parts.append(sentiment.lower())
        
        parts.append(timestamp)
        
        filename = '_'.join(parts)
        return f"{filename}.{extension.lstrip('.')}"