"""fix_foreign_key_constraints

Revision ID: f56f53cbe90e
Revises: d19b43a7f41e
Create Date: 2025-12-13 12:03:42.531934

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f56f53cbe90e'
down_revision: Union[str, Sequence[str], None] = 'd19b43a7f41e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Fix foreign key constraint names after table renames."""
    
    # Drop old foreign key constraint on user_profiles
    op.drop_constraint('userprofile_user_id_fkey', 'user_profiles', type_='foreignkey')
    
    # Create new foreign key constraint with correct name
    op.create_foreign_key(
        'user_profiles_user_id_fkey',
        'user_profiles', 'users',
        ['user_id'], ['id'],
        ondelete='CASCADE'
    )
    
    # Drop old foreign key constraint on personalization_records
    op.drop_constraint('personalizationrecord_user_id_fkey', 'personalization_records', type_='foreignkey')
    
    # Create new foreign key constraint with correct name
    op.create_foreign_key(
        'personalization_records_user_id_fkey',
        'personalization_records', 'users',
        ['user_id'], ['id'],
        ondelete='CASCADE'
    )


def downgrade() -> None:
    """Revert foreign key constraint names."""
    
    # Drop new foreign key constraints
    op.drop_constraint('user_profiles_user_id_fkey', 'user_profiles', type_='foreignkey')
    op.drop_constraint('personalization_records_user_id_fkey', 'personalization_records', type_='foreignkey')
    
    # Recreate old foreign key constraints
    op.create_foreign_key(
        'userprofile_user_id_fkey',
        'user_profiles', 'users',
        ['user_id'], ['id']
    )
    
    op.create_foreign_key(
        'personalizationrecord_user_id_fkey',
        'personalization_records', 'users',
        ['user_id'], ['id']
    )
