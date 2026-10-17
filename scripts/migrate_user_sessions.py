from pathlib import Path
import sys

from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.session import get_engine


def main() -> int:
    engine = get_engine()
    if engine is None:
        print("DATABASE_URL is not configured; migration was not run.")
        return 1

    try:
        columns = {
            column["name"] for column in inspect(engine).get_columns("chat_sessions")
        }
        with engine.begin() as connection:
            if "user_id" not in columns:
                connection.execute(
                    text(
                        "ALTER TABLE chat_sessions "
                        "ADD COLUMN user_id UUID NULL REFERENCES users(id)"
                    )
                )
                print("Added chat_sessions.user_id.")
            else:
                print("chat_sessions.user_id already exists.")

            connection.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_chat_sessions_user_id "
                    "ON chat_sessions(user_id)"
                )
            )
    except SQLAlchemyError as error:
        print(f"Session ownership migration failed: {error}")
        return 1

    print("Session ownership migration completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())