from typing import Any

from constants import UUID_NIL

# cdg:提取消息的线程消息，线程消息的应用场景是：对于历史消息，如果消息有父消息，则父消息是线程消息的开始，子消息是线程消息的结束。
# cdg:会话消息和线程消息的区别是：会话消息是会话中所有的消息，而线程消息是会话中与最新消息相关的消息。
def extract_thread_messages(messages: list[Any]):
    thread_messages = []
    next_message = None
    # cdg:遍历消息，提取线程消息。
    for message in messages:
        # cdg:如果没有上一条消息
        if not message.parent_message_id:
            # cdg:如果消息没有父消息，则将消息添加到线程消息列表中，并结束线程消息的提取。
            # If the message is regenerated and does not have a parent message, it is the start of a new thread
            thread_messages.append(message)
            break

        # 如果没有下一条消息的ID
        if not next_message:
            thread_messages.append(message)
            next_message = message.parent_message_id
        else:
            if next_message in {message.id, UUID_NIL}:
                thread_messages.append(message)
                next_message = message.parent_message_id

    return thread_messages
