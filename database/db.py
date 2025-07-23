from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from database.models import Base

# Database is hosted locally in the file system, not over the local net
DATABASE_URL = "sqlite+aiosqlite:///./driftx.db"

# Entry point to the database
engine = create_async_engine(DATABASE_URL, echo=True)

# Session factory: basically a meeseeks box 
SessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)