css = '''
<style>
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: #1e1e2f;
    color: white;
    margin: 0;
    padding: 0;
}

.chat-message {
    display: flex;
    align-items: flex-start;
    margin: 1rem 0;
    padding: 1rem;
    border-radius: 1rem;
    max-width: 800px;
    width: 100%;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.chat-message.bot {
    background-color: #2d3748;
    margin-left: auto;
    margin-right: 0;
}

.chat-message.user {
    background-color: #4a5568;
    margin-left: 0;
    margin-right: auto;
}

.chat-message .avatar {
    flex-shrink: 0;
    width: 60px;
    height: 60px;
    border-radius: 50%;
    overflow: hidden;
    margin-right: 1rem;
    transition: transform 0.3s ease;
}

.chat-message .avatar:hover {
    transform: scale(1.05);
}

.chat-message .avatar img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}

.chat-message .message {
    flex: 1;
    color: #f1f1f1;
    font-size: 1rem;
    line-height: 1.6;
    word-wrap: break-word;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png" alt="Bot Avatar">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.ibb.co/zdhB8DX/cool-question-image.jpg" alt="User Avatar">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''
