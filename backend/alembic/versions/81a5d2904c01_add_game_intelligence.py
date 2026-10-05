"""Game probabilities, audit signals and paper positions.

Frozen DDL snapshot. Existing tables and rows are preserved.
Revision ID: 81a5d2904c01
Revises: 5f1f68be9221
"""
from alembic import op

revision = "81a5d2904c01"
down_revision = "5f1f68be9221"
branch_labels = None
depends_on = None


STATEMENTS = (
    """
    ALTER TABLE games ADD COLUMN tipoff_time_estimated BOOLEAN NOT NULL DEFAULT false
    """,
    """
    CREATE TABLE game_predictions (
    	game_id UUID NOT NULL, 
    	as_of TIMESTAMP WITH TIME ZONE NOT NULL, 
    	home_probability FLOAT NOT NULL, 
    	model_version VARCHAR(50) NOT NULL, 
    	features JSONB NOT NULL, 
    	output JSONB NOT NULL, 
    	id UUID NOT NULL, 
    	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    	updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    	PRIMARY KEY (id), 
    	FOREIGN KEY(game_id) REFERENCES games (id)
    )
    """,
    """
    CREATE INDEX ix_game_predictions_game_id ON game_predictions (game_id)
    """,
    """
    CREATE TABLE game_signals (
    	prediction_id UUID NOT NULL, 
    	snapshot_id UUID, 
    	selection VARCHAR(10) NOT NULL, 
    	status VARCHAR(30) NOT NULL, 
    	decision JSONB NOT NULL, 
    	id UUID NOT NULL, 
    	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    	updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    	PRIMARY KEY (id), 
    	FOREIGN KEY(prediction_id) REFERENCES game_predictions (id), 
    	FOREIGN KEY(snapshot_id) REFERENCES market_snapshots (id)
    )
    """,
    """
    CREATE INDEX ix_game_signals_prediction_id ON game_signals (prediction_id)
    """,
    """
    CREATE TABLE paper_positions (
    	game_id UUID NOT NULL, 
    	signal_id UUID NOT NULL, 
    	selection VARCHAR(10) NOT NULL, 
    	stake FLOAT NOT NULL, 
    	entry_price FLOAT NOT NULL, 
    	shares FLOAT NOT NULL, 
    	status VARCHAR(30) NOT NULL, 
    	pnl FLOAT, 
    	settled_at TIMESTAMP WITH TIME ZONE, 
    	id UUID NOT NULL, 
    	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    	updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    	PRIMARY KEY (id), 
    	CONSTRAINT uq_paper_game_selection UNIQUE (game_id, selection), 
    	FOREIGN KEY(game_id) REFERENCES games (id), 
    	FOREIGN KEY(signal_id) REFERENCES game_signals (id)
    )
    """,
    """
    CREATE INDEX ix_paper_positions_game_id ON paper_positions (game_id)
    """,
)


def upgrade():
    for statement in STATEMENTS:
        op.execute(statement)


def downgrade():
    op.drop_table("paper_positions")
    op.drop_table("game_signals")
    op.drop_table("game_predictions")
    op.drop_column("games", "tipoff_time_estimated")
