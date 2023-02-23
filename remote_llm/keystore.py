from typing import Optional
from sqlalchemy import String, Integer, Boolean, Column
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Mapped
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
import uuid

Base = declarative_base()

class ApiKeys(Base):
    __tablename__ = "user_account"
    id: Mapped[int] = Column(Integer, primary_key=True)
    name: Mapped[str] = Column(String)
    key: Mapped[str] = Column(String)
    admin: Mapped[bool] = Column(Boolean)
    
    def __repr__(self) -> str:
        return f"User(id={self.id!r}, name={self.name!r}, key={self.key!r})"

class ApiKeystore():
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.engine = create_engine(db_path, echo=False)
        Base.metadata.create_all(self.engine)
    
    def check_key(self, *, key: str) -> Optional[str]:
        with Session(self.engine) as session:
            results = session.query(ApiKeys.name).filter(ApiKeys.key == key).first()
            if results is None:
                return None
            return results[0] # results is a tuple, so we need to index it.

    def add_admin_key(self, *, name: str, key: str) -> Optional[str]:
        with Session(self.engine) as session:
            if session.query(ApiKeys).filter(ApiKeys.name == name).first():
                return None

            session.add(ApiKeys(key=key, name=name, admin=True))
            session.commit()
            return key
        
    def add_key(self, *, name: str) -> Optional[str]:
        with Session(self.engine) as session:
            key = str(uuid.uuid4())
            if session.query(ApiKeys).filter(ApiKeys.name == name).first():
                return None

            session.add(ApiKeys(key=key, name=name, admin=False))
            session.commit()
            return key

    def remove_key(self, *, name: str) -> bool:
        with Session(self.engine) as session:
            session.query(ApiKeys).filter(ApiKeys.name == name).delete()
            session.commit()
            return True
        
    def get_all_keys(self) -> list[str]:
        with Session(self.engine) as session:
            return session.query(ApiKeys).all()
        
    def get_key(self, *, name: str) -> Optional[str]:
        with Session(self.engine) as session:
            return session.query(ApiKeys.key).filter(ApiKeys.name == name).first()[0]

# A simple CLI to add/remove keys.
def main():
    print("Editing Keystore")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("db_path", type=str, help="The path to the database.")
    parser.add_argument("--add", type=str, required=False, help="Add a key.")
    parser.add_argument("--remove", type=str, required=False, help="Remove a key.")
    parser.add_argument("--list", action="store_true", help="List all keys.")
    parser.add_argument("--get", type=str, required=False, help="Get a key.")
    args = parser.parse_args()
    api_key_store = ApiKeystore(args.db_path)
    if args.add is not None:
        key = api_key_store.add_key(name=args.add)
        if key is None:
            print("Failed to add key.")
        else:
            print(f"Added key {key}")
    elif args.remove is not None:
        if api_key_store.remove_key(name=args.remove):
            print("Removed key.")
        else:
            print("Failed to remove key.")
    elif args.list:
        print("Keys:")
        for key in api_key_store.get_all_keys():
            print(f" - {key}")
    elif args.get is not None:
        key = api_key_store.get_key(name=args.get)
        if key is None:
            print("Failed to get key.")
        else:
            print(f"Key: {key}")

if __name__ == '__main__':
    main()