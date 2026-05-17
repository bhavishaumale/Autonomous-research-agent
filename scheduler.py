# scheduler.py

from apscheduler.schedulers.blocking import BlockingScheduler
from main import run_agent
import logging

logging.basicConfig(level=logging.INFO)
scheduler = BlockingScheduler()


@scheduler.scheduled_job("cron", day_of_week="mon", hour=8, minute=0)
def scheduled_run():
    run_agent(topics=[
        "retrieval augmented generation",
        "large language model agents",
        "efficient llm inference",
        "multimodal language models",
        "AI safety alignment"
    ])


if __name__ == "__main__":
    print("Scheduler started. Agent runs every Monday at 8:00 AM.")
    print("Press Ctrl+C to stop.")
    scheduler.start()