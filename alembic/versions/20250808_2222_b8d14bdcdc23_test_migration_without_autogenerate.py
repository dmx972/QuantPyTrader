"""Initial migration: create all tables and constraints (manual)

Revision ID: b8d14bdcdc23
Revises: 
Create Date: 2025-08-08 22:22:42.874025

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b8d14bdcdc23'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Import our models to ensure metadata is loaded
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    from core.database.models import Base
    from core.database import models, trading_models, kalman_models
    
    # Get current connection from Alembic context
    from alembic import context
    connection = context.get_bind()
    
    # Create all tables using SQLAlchemy
    Base.metadata.create_all(bind=connection)


def downgrade() -> None:
    """Downgrade schema."""
    # Import our models to ensure metadata is loaded
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    from core.database.models import Base
    from core.database import models, trading_models, kalman_models
    
    # Get current connection from Alembic context
    from alembic import context
    connection = context.get_bind()
    
    # Drop all tables using SQLAlchemy
    Base.metadata.drop_all(bind=connection)
