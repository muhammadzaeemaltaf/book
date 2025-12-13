"""Initial migration for user profile and related tables

Revision ID: 5dd682b5b5ea
Revises:
Create Date: 2025-12-13 00:27:24.316028

"""
from typing import Sequence, Union
from datetime import datetime
import uuid

from alembic import op
import sqlalchemy as sa
from sqlalchemy import JSON, Text


# revision identifiers, used by Alembic.
revision: str = '5dd682b5b5ea'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create UserProfile table
    op.create_table(
        'userprofile',
        sa.Column('id', sa.UUID, primary_key=True),
        sa.Column('user_id', sa.String, nullable=False, unique=True),
        sa.Column('python_experience', sa.String, nullable=False, default='none'),
        sa.Column('cpp_experience', sa.String, nullable=False, default='none'),
        sa.Column('js_ts_experience', sa.String, nullable=False, default='none'),
        sa.Column('ai_ml_familiarity', sa.String, nullable=False, default='none'),
        sa.Column('ros2_experience', sa.String, nullable=False, default='none'),
        sa.Column('gpu_details', sa.String, nullable=False, default='none'),
        sa.Column('ram_capacity', sa.String, nullable=False, default='4GB'),
        sa.Column('operating_system', sa.String, nullable=False, default='linux'),
        sa.Column('jetson_ownership', sa.Boolean, nullable=False, default=False),
        sa.Column('realsense_lidar_availability', sa.Boolean, nullable=False, default=False),
        sa.Column('created_at', sa.DateTime, nullable=False, default=datetime.utcnow),
        sa.Column('updated_at', sa.DateTime, nullable=False, default=datetime.utcnow),
    )
    op.create_foreign_key('fk_userprofile_user_id', 'userprofile', 'user', ['user_id'], ['id'])

    # Create PersonalizationRecord table
    op.create_table(
        'personalizationrecord',
        sa.Column('id', sa.UUID, primary_key=True),
        sa.Column('user_id', sa.String, nullable=False),
        sa.Column('chapter_id', sa.String, nullable=False),
        sa.Column('personalized_content', sa.Text, nullable=False),  # 10MB max as per model
        sa.Column('personalization_metadata', JSON, nullable=False, default=dict),
        sa.Column('created_at', sa.DateTime, nullable=False, default=datetime.utcnow),
        sa.Column('updated_at', sa.DateTime, nullable=False, default=datetime.utcnow),
    )
    op.create_foreign_key('fk_personalizationrecord_user_id', 'personalizationrecord', 'user', ['user_id'], ['id'])
    op.create_index('ix_personalizationrecord_chapter_id', 'personalizationrecord', ['chapter_id'])

    # Create AISummary table
    op.create_table(
        'aisummary',
        sa.Column('id', sa.UUID, primary_key=True),
        sa.Column('chapter_id', sa.String, nullable=False),
        sa.Column('summary_content', sa.String(10000), nullable=False),  # min_length=100, max_length=10000 as per model
        sa.Column('summary_metadata', JSON, nullable=False, default=dict),
        sa.Column('access_count', sa.Integer, nullable=False, default=0),
        sa.Column('last_accessed', sa.DateTime, nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=False, default=datetime.utcnow),
        sa.Column('updated_at', sa.DateTime, nullable=False, default=datetime.utcnow),
    )

    # Add index for chapter_id in AISummary
    op.create_index('ix_aisummary_chapter_id', 'aisummary', ['chapter_id'])


def downgrade() -> None:
    """Downgrade schema."""
    # Drop indexes
    op.drop_index('ix_aisummary_chapter_id', table_name='aisummary')
    op.drop_index('ix_personalizationrecord_chapter_id', table_name='personalizationrecord')

    # Drop foreign key constraints
    op.drop_constraint('fk_personalizationrecord_user_id', 'personalizationrecord', type_='foreignkey')
    op.drop_constraint('fk_userprofile_user_id', 'userprofile', type_='foreignkey')

    # Drop tables in reverse order to respect foreign key constraints
    op.drop_table('aisummary')
    op.drop_table('personalizationrecord')
    op.drop_table('userprofile')
