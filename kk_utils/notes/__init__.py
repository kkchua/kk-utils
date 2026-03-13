"""
kk_utils.notes — Notes service

Notes management with group organization:
- Create, read, update, delete notes
- Group organization
- Keyword search
- List notes by group

Usage:
    from kk_utils.notes.service import (
        create_note,
        get_note,
        update_note,
        delete_note,
        search_notes,
        list_notes,
    )
    
    # Create a note
    note = create_note(title="My Note", content="Content...", group_id=1)
    
    # Get a note
    note = get_note(note_id=1)
    
    # Search notes
    results = search_notes(query="keyword")
"""

from .service import (
    create_note,
    get_note,
    update_note,
    delete_note,
    search_notes,
    list_notes,
)

__all__ = [
    'create_note',
    'get_note',
    'update_note',
    'delete_note',
    'search_notes',
    'list_notes',
]
