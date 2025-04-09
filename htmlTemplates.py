css = '''
<style>
.chat-message {
    padding: 0.75rem;
    margin: 0.5rem;
    border-radius: 0.5rem;
    display: flex;
    max-width: 75%;
}

.chat-message.user {
    background-color: #2b313e;
    margin-left: auto;
    margin-right: 1rem;
}

.chat-message.bot {
    background-color: #475063;
    margin-right: auto;
    margin-left: 1rem;
}

.chat-message .message {
    color: #fff;
    padding: 0.25rem;
    line-height: 1.4;
}

.chat-container {
    display: flex;
    flex-direction: column;
    padding: 0.5rem;
}

</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="message">{{MSG}}</div>
</div>
'''