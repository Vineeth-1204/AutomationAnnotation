import logging
import sys
import json
from datetime import datetime, timezone
from app.core.config import settings

class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "filename": record.filename,
            "line_number": record.lineno,
        }
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)

def setup_logging() -> None:
    logger = logging.getLogger()
    # Clear existing handlers
    logger.handlers.clear()
    
    # Select log level based on environment
    log_level = logging.INFO
    if settings.ENVIRONMENT == "local" or settings.ENVIRONMENT == "test":
        log_level = logging.DEBUG
        
    logger.setLevel(log_level)
    
    # Configure handler
    handler = logging.StreamHandler(sys.stdout)
    if settings.ENVIRONMENT == "production":
        handler.setFormatter(JSONFormatter())
    else:
        # Standard formatted output for local development
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s in %(module)s (%(filename)s:%(lineno)d): %(message)s"
        )
        handler.setFormatter(formatter)
        
    logger.addHandler(handler)
    
    # Suppress verbose library logs unless needed
    logging.getLogger("uvicorn.access").handlers = [handler]
    logging.getLogger("uvicorn.error").handlers = [handler]
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
