"""initial

Revision ID: 0001_initial
Revises: 
Create Date: 2026-03-26 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0001_initial'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'users',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('email', sa.String(), nullable=False, unique=True),
        sa.Column('password', sa.String(), nullable=False),
        sa.Column('role', sa.String(), nullable=True, server_default='user'),
        sa.Column('is_active', sa.Boolean(), nullable=True, server_default=sa.text('1')),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )

    op.create_table(
        'documents',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('filename', sa.String(), nullable=False),
        sa.Column('storage_path', sa.String(), nullable=False),
        sa.Column('content_type', sa.String(), nullable=False),
        sa.Column('file_size', sa.BigInteger(), nullable=True, server_default='0'),
        sa.Column('status', sa.String(), nullable=True),
        sa.Column('chunk_count', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('error_msg', sa.String(), nullable=True),
        sa.Column('modality', sa.String(), nullable=True, server_default='text'),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
    )

    op.create_table(
        'chat_messages',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('session_id', sa.String(), nullable=False),
        sa.Column('role', sa.String(), nullable=False),
        sa.Column('content', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )

    op.create_table(
        'eval_records',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('run_id', sa.String(), nullable=False),
        sa.Column('document_id', sa.String(), nullable=False),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('mode', sa.String(), nullable=False),
        sa.Column('sample_index', sa.Integer(), nullable=True, server_default='-1'),
        sa.Column('question', sa.String(), nullable=True),
        sa.Column('answer', sa.String(), nullable=True),
        sa.Column('ground_truth', sa.String(), nullable=True),
        sa.Column('faithfulness', sa.Integer(), nullable=True),
        sa.Column('answer_relevancy', sa.Integer(), nullable=True),
        sa.Column('context_precision', sa.Integer(), nullable=True),
        sa.Column('context_recall', sa.Integer(), nullable=True),
        sa.Column('passed', sa.Boolean(), nullable=True, server_default=sa.text('0')),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table('eval_records')
    op.drop_table('chat_messages')
    op.drop_table('documents')
    op.drop_table('users')
