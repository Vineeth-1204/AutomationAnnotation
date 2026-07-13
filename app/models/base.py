from sqlalchemy.orm import DeclarativeBase, declared_attr

class Base(DeclarativeBase):
    # Generate __tablename__ automatically
    @declared_attr.directive
    def __tablename__(cls) -> str:
        # Convert CamelCase to lowercase with 's' suffix
        # Handles simple cases like User -> users, Item -> items
        name = cls.__name__
        # A simple snake_case-ish pluralizer:
        import re
        parts = re.findall(r'[A-Z][a-z0-9]*', name)
        if not parts:
            return name.lower() + "s"
        snake = "_".join(part.lower() for part in parts)
        return snake + "s"
