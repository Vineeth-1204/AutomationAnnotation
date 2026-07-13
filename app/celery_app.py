"""
Celery application instance for clanker.ai.

Broker  : Redis (REDIS_URL from settings)
Backend : Redis (same URL, separate DB key namespace)

IMPORTANT: The broker and backend URLs are set lazily via on_after_configure
so that tests can set task_always_eager=True BEFORE any Redis connection is
attempted. If you import this module during tests, no socket is opened.
"""
import os
from celery import Celery

# Build the app with NO broker/backend — they're injected lazily below
celery_app = Celery("clanker")

celery_app.conf.update(
    # Serialisation
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Reliability
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_reject_on_worker_lost=True,

    # Visibility
    task_track_started=True,
    result_expires=86400,

    # Retry defaults
    task_max_retries=3,
    task_default_retry_delay=5,

    # Timezone
    timezone="UTC",
    enable_utc=True,

    # Task discovery
    include=["app.workers.celery_tasks"],
)


@celery_app.on_after_configure.connect
def _setup_broker(sender, **kwargs):
    """
    Set broker and backend URLs AFTER configuration is finalised.
    In test mode (task_always_eager=True) this signal still fires but
    no actual TCP connection is made because eager mode skips the broker.
    """
    from app.core.config import settings
    sender.conf.broker_url = settings.REDIS_URL
    sender.conf.result_backend = settings.REDIS_URL
