from pydantic import BaseModel


class FolderCreateRequest(BaseModel):
    path: str


class FolderResponse(BaseModel):
    folder_id: int
    path: str
    enabled: bool
    created_at: str


class FolderCreateResponse(BaseModel):
    folder: FolderResponse
    new_files: int
    updated_files: int
    unchanged_files: int
    skipped_files: int
