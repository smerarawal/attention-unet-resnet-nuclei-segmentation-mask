from typing import Optional

from pydantic import BaseModel


class FileResponse(BaseModel):
    file_id: int
    folder_id: int
    filename: str
    path: str
    extension: str
    size_bytes: int
    created_at: Optional[str]
    modified_at: Optional[str]
    content_hash: str
    indexed_at: Optional[str]
    status: str
