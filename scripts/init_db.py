from pathlib import Path
import sys

from sqlalchemy.exc import SQLAlchemyError


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.models import Base
from src.db.session import get_engine


def main() -> int:
    engine = get_engine()
    if engine is None:
        print("DATABASE_URL is not configured; no database tables were created.")
        return 1

    try:
        Base.metadata.create_all(engine)
    except SQLAlchemyError as error:
        print(f"Database initialization failed: {error}")
        return 1

    print("Database tables created successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
