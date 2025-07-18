import logging

from celery import shared_task  # type: ignore

from extensions.ext_database import db
from models.account import Account
from services.billing_service import BillingService
from tasks.mail_account_deletion_task import send_deletion_success_task

logger = logging.getLogger(__name__)

# cdg: 删除账户任务，具体实现为：从数据库中获取账户；如果账户不存在，则抛出异常；删除账户；提交数据库。
# cdg: 这里引用celery的@shared_task装饰器，将任务添加到celery的dataset队列中，实现异步执行。
@shared_task(queue="dataset")
def delete_account_task(account_id):
    account = db.session.query(Account).filter(Account.id == account_id).first()
    try:
        BillingService.delete_account(account_id)
    except Exception as e:
        logger.exception(f"Failed to delete account {account_id} from billing service.")
        raise

    if not account:
        logger.error(f"Account {account_id} not found.")
        return
    # send success email
    send_deletion_success_task.delay(account.email)
