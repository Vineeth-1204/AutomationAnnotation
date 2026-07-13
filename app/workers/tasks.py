import logging
import asyncio

logger = logging.getLogger(__name__)

async def send_welcome_email(email: str, name: str | None = None) -> None:
    logger.info(f"Starting background email job for {email}")
    # Simulate a network delay (e.g., SMTP transport)
    await asyncio.sleep(2)
    display_name = name or email
    logger.info(f"Welcome email successfully sent to {display_name} ({email})")
