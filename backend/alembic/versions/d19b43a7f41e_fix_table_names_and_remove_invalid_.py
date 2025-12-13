"""fix_table_names_and_remove_invalid_relationship

Revision ID: d19b43a7f41e
Revises: 5dd682b5b5ea
Create Date: 2025-12-13 11:59:43.205389

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd19b43a7f41e'
down_revision: Union[str, Sequence[str], None] = '5dd682b5b5ea'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Rename tables to match explicit __tablename__ attributes."""
    # Rename user table to users
    op.rename_table('user', 'users')
    
    # Rename userprofile table to user_profiles  
    op.rename_table('userprofile', 'user_profiles')
    
    # Rename personalizationrecord table to personalization_records
    op.rename_table('personalizationrecord', 'personalization_records')
    
    # Rename aisummary table to ai_summaries
    op.rename_table('aisummary', 'ai_summaries')
    
    # Note: The invalid UserProfile.personalization_records relationship
    # has been removed from the model. No database changes needed for this.


def downgrade() -> None:
    """Revert table names back to auto-generated lowercase."""
    # Rename tables back
    op.rename_table('users', 'user')
    op.rename_table('user_profiles', 'userprofile')
    op.rename_table('personalization_records', 'personalizationrecord')
    op.rename_table('ai_summaries', 'aisummary')
