"""
Personal Assistant Backend - Note Service
Fresh implementation

Business logic for note operations.
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

from sqlalchemy.orm import Session
from app.database.session import get_db
from app.models.note import Note
from app.models.group import Group

logger = logging.getLogger(__name__)


def create_note(
    title: str,
    content: str,
    group_id: int,
    user_id: Optional[str] = None,
    metadata: Optional[Dict] = None,
    db: Session = None,
) -> Dict[str, Any]:
    """
    Create a new note.
    
    Args:
        title: Note title
        content: Note content (markdown)
        group_id: Group ID
        user_id: User ID
        metadata: Optional metadata
        db: Database session (optional, will create if not provided)
    
    Returns:
        Created note data
    """
    # Create db session if not provided
    from app.database.session import get_db
    if db is None:
        db = next(get_db())
        should_close = True
    else:
        should_close = False
    
    try:
        # Verify group exists
        group = db.query(Group).filter(Group.id == group_id).first()
        if not group:
            return {"error": f"Group {group_id} not found"}
        
        # Create note
        note = Note(
            title=title,
            content=content,
            group_id=group_id,
            metadata_json=metadata or {},
        )
        
        db.add(note)
        db.commit()
        db.refresh(note)
        
        logger.info(f"Created note {note.id}: {title}")
        
        return note.to_dict()
    
    except Exception as e:
        if should_close:
            db.rollback()
        logger.error(f"Failed to create note: {e}", exc_info=True)
        return {"error": str(e)}
    finally:
        if should_close:
            db.close()


def get_note(
    note_id: int,
    user_id: Optional[str] = None,
    db: Session = None,
) -> Dict[str, Any]:
    """Get a note by ID."""
    from app.database.session import get_db
    if db is None:
        db = next(get_db())
        should_close = True
    else:
        should_close = False
    
    try:
        note = db.query(Note).filter(Note.id == note_id).first()
        
        if not note:
            return {"error": f"Note {note_id} not found"}
        
        # TODO: Add access control check
        
        return note.to_dict()
    
    except Exception as e:
        logger.error(f"Failed to get note: {e}", exc_info=True)
        return {"error": str(e)}
    finally:
        if should_close:
            db.close()


def update_note(
    note_id: int,
    title: Optional[str] = None,
    content: Optional[str] = None,
    metadata: Optional[Dict] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Update a note."""
    db = next(get_db())
    
    try:
        note = db.query(Note).filter(Note.id == note_id).first()
        
        if not note:
            return {"error": f"Note {note_id} not found"}
        
        # TODO: Add access control check
        
        # Update fields
        if title is not None:
            note.title = title
        if content is not None:
            note.content = content
        if metadata is not None:
            note.metadata_json = metadata
        
        note.updated_at = datetime.utcnow()
        
        db.commit()
        db.refresh(note)
        
        logger.info(f"Updated note {note_id}")
        
        return note.to_dict()
    
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to update note: {e}", exc_info=True)
        return {"error": str(e)}
    finally:
        db.close()


def delete_note(
    note_id: int,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Delete a note."""
    db = next(get_db())
    
    try:
        note = db.query(Note).filter(Note.id == note_id).first()
        
        if not note:
            return {"error": f"Note {note_id} not found"}
        
        # TODO: Add access control check
        
        db.delete(note)
        db.commit()
        
        logger.info(f"Deleted note {note_id}")
        
        return {"deleted": True, "note_id": note_id}
    
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to delete note: {e}", exc_info=True)
        return {"error": str(e)}
    finally:
        db.close()


def search_notes(
    query: str,
    group_id: Optional[int] = None,
    limit: int = 20,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Search notes by keyword."""
    db = next(get_db())
    
    try:
        # Build query
        db_query = db.query(Note).filter(
            (Note.title.ilike(f"%{query}%")) |
            (Note.content.ilike(f"%{query}%"))
        )
        
        # Apply filters
        if group_id:
            db_query = db_query.filter(Note.group_id == group_id)
        
        if user_id:
            # TODO: Filter by user_id for access control
            pass
        
        # Execute
        notes = db_query.limit(limit).all()
        
        return {
            "query": query,
            "notes": [note.to_dict() for note in notes],
            "count": len(notes),
        }
    
    except Exception as e:
        logger.error(f"Failed to search notes: {e}", exc_info=True)
        return {"error": str(e)}
    finally:
        db.close()


def list_notes(
    group_id: Optional[int] = None,
    limit: int = 50,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """List notes."""
    db = next(get_db())
    
    try:
        db_query = db.query(Note)
        
        if group_id:
            db_query = db_query.filter(Note.group_id == group_id)
        
        if user_id:
            # TODO: Filter by user_id
            pass
        
        notes = db_query.order_by(Note.updated_at.desc()).limit(limit).all()
        
        return {
            "notes": [note.to_dict() for note in notes],
            "count": len(notes),
        }
    
    except Exception as e:
        logger.error(f"Failed to list notes: {e}", exc_info=True)
        return {"error": str(e)}
    finally:
        db.close()
