import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class Document:
    """
    Standard Document Schema for MLRL02.
    """
    def __init__(self, content: str, metadata: Dict[str, Any]):
        self.page_content = content
        self.metadata = metadata

    def __repr__(self):
        source = self.metadata.get('source', 'Unknown')
        return f"Document(source='{source}', len={len(self.page_content)})"

def extract_frontmatter(content: str) -> tuple[Dict[str, Any], str]:
    """
    Extracts YAML frontmatter from the start of a markdown string.
    Returns (metadata_dict, remaining_content).
    """
    frontmatter_pattern = re.compile(r'^---\s*\n(.*?)\n---\s*\n', re.DOTALL)
    match = frontmatter_pattern.match(content)
    
    metadata = {}
    if match:
        raw_yaml = match.group(1)
        # Simple YAML-ish parser (key: value)
        for line in raw_yaml.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                # Clean up values (strip spaces, remove [ ] for lists)
                key = key.strip()
                value = value.strip()
                if value.startswith('[') and value.endswith(']'):
                    value = [v.strip() for v in value[1:-1].split(',')]
                metadata[key] = value
        
        # Return metadata and content without frontmatter
        return metadata, content[match.end():]
    
    return {}, content

def chunk_by_headers(content: str) -> List[tuple[str, str]]:
    """
    Splits content by markdown headers (# Header).
    Returns a list of (header_name, section_content).
    """
    # Split on newline followed by a markdown header (using positive lookahead)
    sections = re.split(r'\n(?=#{1,6}\s)', content)
    
    chunks = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
            
        # Parse the header and content from the section
        lines = section.split('\n', 1)
        header_line = lines[0].strip()
        
        # Check if the first line is actually a header
        if re.match(r'^#{1,6}\s', header_line):
            header = header_line.lstrip('#').strip()
            content_section = lines[1].strip() if len(lines) > 1 else ""
            chunks.append((header, content_section))
        else:
            # If no header (e.g., intro text before first header)
            chunks.append(("Intro", section))
            
    return chunks

def _clean_text(text: str) -> str:
    """
    Sanitizes text for AI processing (Task 1.10).
    - Removes image links
    - Normalizes whitespace
    """
    # Remove markdown image links: ![alt](url)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    # Normalize multiple newlines/spaces
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()

def load_markdown(folder_path: str, chunk: bool = True) -> List[Document]:
    """
    Advanced Loader for MLRL02.
    - Recursive Loading
    - Frontmatter Extraction (Task 1.9)
    - Smart Chunking (Task 1.7)
    - Metadata Handling (Task 1.4)
    """
    documents = []
    base_path = Path(folder_path)

    if not base_path.exists():
        logger.error(f"Directory not found: {folder_path}")
        return []

    for file_path in base_path.rglob("*.md"):
        if file_path.is_file():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    raw_content = f.read()
                
                # 1. Extract Metadata from Frontmatter
                file_metadata, clean_content = extract_frontmatter(raw_content)
                
                # 1.5 Clean content (Task 1.10)
                clean_content = _clean_text(clean_content)
                
                # 2. Base Metadata
                base_metadata = {
                    "source": file_path.name,
                    "path": str(file_path),
                    "last_modified": file_path.stat().st_mtime,
                    **file_metadata # Merge frontmatter tags/status
                }
                
                if chunk:
                    # 3. Smart Chunking by Headers
                    sections = chunk_by_headers(clean_content)
                    for header, text in sections:
                        if not text: continue # Skip empty sections
                        
                        chunk_metadata = base_metadata.copy()
                        chunk_metadata["header"] = header
                        
                        documents.append(Document(content=text, metadata=chunk_metadata))
                    logger.info(f"Loaded {file_path.name} ({len(sections)} chunks)")
                else:
                    documents.append(Document(content=clean_content, metadata=base_metadata))
                    logger.info(f"Loaded {file_path.name} (full)")
                
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")

    return documents

if __name__ == "__main__":
    docs = load_markdown("workspace/markdown")
    for d in docs[:5]:
        print(f"\nSource: {d.metadata['source']} | Header: {d.metadata.get('header')}")
        print(f"Content: {d.page_content[:50]}...")
