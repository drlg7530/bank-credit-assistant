/**
 * ============================================
 * 智能信贷业务辅助系统 - 主JavaScript文件
 * ============================================
 */

// ========== 全局变量 ==========
// 当前session_id
let currentSessionId = null;

// ========== 页面初始化 ==========
// 页面加载时初始化
window.onload = function() {
    loadQuickQuestions();
    // 延迟加载历史记录，等待session_id生成
    setTimeout(() => {
        if (currentSessionId) {
            loadConversationHistory();
        }
    }, 1000);
};

// ========== 对话历史管理 ==========
/**
 * 加载对话历史记录
 */
async function loadConversationHistory() {
    if (!currentSessionId) {
        document.getElementById('conversationHistory').innerHTML = '<div class="empty-history">暂无对话记录</div>';
        return;
    }

    try {
        const response = await fetch(`/api/conversation-history?session_id=${encodeURIComponent(currentSessionId)}&limit=50`);
        const data = await response.json();
        
        const historyContainer = document.getElementById('conversationHistory');
        
        if (data.success && data.history && data.history.length > 0) {
            // 按turn_id分组，每轮对话包含user和assistant
            const groupedHistory = {};
            data.history.forEach(item => {
                const turnId = item.turn_id;
                if (!groupedHistory[turnId]) {
                    groupedHistory[turnId] = { user: null, assistant: null };
                }
                if (item.role === 'user') {
                    groupedHistory[turnId].user = item;
                } else if (item.role === 'assistant') {
                    groupedHistory[turnId].assistant = item;
                }
            });

            // 生成HTML
            let html = '';
            const turnIds = Object.keys(groupedHistory).sort((a, b) => parseInt(b) - parseInt(a)); // 倒序显示，最新的在前
            
            turnIds.forEach(turnId => {
                const turn = groupedHistory[turnId];
                if (turn.user) {
                    // 用户问题
                    const time = new Date(turn.user.timestamp).toLocaleString('zh-CN');
                    const userContent = escapeHtml(turn.user.content);
                    html += `
                        <div class="history-item" onclick="reuseHistoryQuery(${JSON.stringify(turn.user.content)})">
                            <div class="history-item-header">
                                <span class="history-item-turn">第${turnId}轮</span>
                                <span class="history-item-time">${time}</span>
                            </div>
                            <div>
                                <span class="history-item-role user">用户</span>
                                <span class="history-item-content">${userContent}</span>
                            </div>
                        </div>
                    `;
                }
                if (turn.assistant) {
                    // 助手回答
                    const time = new Date(turn.assistant.timestamp).toLocaleString('zh-CN');
                    const content = turn.assistant.content.length > 100 
                        ? turn.assistant.content.substring(0, 100) + '...' 
                        : turn.assistant.content;
                    html += `
                        <div class="history-item">
                            <div class="history-item-header">
                                <span class="history-item-turn">第${turnId}轮</span>
                                <span class="history-item-time">${time}</span>
                            </div>
                            <div>
                                <span class="history-item-role assistant">助手</span>
                                <span class="history-item-content">${escapeHtml(content)}</span>
                            </div>
                        </div>
                    `;
                }
            });

            historyContainer.innerHTML = html;
        } else {
            historyContainer.innerHTML = '<div class="empty-history">暂无对话记录</div>';
        }
    } catch (error) {
        console.error('加载对话历史失败:', error);
        document.getElementById('conversationHistory').innerHTML = '<div class="empty-history">加载失败</div>';
    }
}

/**
 * 重用历史查询
 * @param {string} question - 历史问题
 */
function reuseHistoryQuery(question) {
    document.getElementById('questionInput').value = question;
    submitQuery();
}

// ========== 快捷问题管理 ==========
/**
 * 加载快捷问题
 */
async function loadQuickQuestions() {
    const role = document.getElementById('roleSelect').value;
    const container = document.getElementById('quickQuestions');
    
    try {
        const response = await fetch(`/api/quick-questions?role=${encodeURIComponent(role)}`);
        const data = await response.json();
        
        if (data.success) {
            const questions = data.questions;
            container.innerHTML = '';
            
            questions.forEach(question => {
                const tag = document.createElement('div');
                tag.className = 'question-tag';
                tag.textContent = question;
                tag.onclick = () => {
                    document.getElementById('questionInput').value = question;
                    submitQuery();
                };
                container.appendChild(tag);
            });
        }
    } catch (error) {
        console.error('加载快捷问题失败:', error);
    }
}

// ========== 查询提交 ==========
/**
 * 提交查询
 */
async function submitQuery() {
    const question = document.getElementById('questionInput').value.trim();
    const role = document.getElementById('roleSelect').value;
    // 默认启用query改写，不启用重排序
    const enableRewrite = true;
    const enableRerank = false;

    if (!question) {
        alert('请输入查询问题');
        return;
    }

    // 显示加载状态
    document.getElementById('loading').classList.add('show');
    document.getElementById('error').classList.remove('show');
    document.getElementById('submitBtn').disabled = true;

    // 获取结果区域
    const resultContent = document.getElementById('resultContent');
    
    // 移除空状态提示
    const emptyState = resultContent.querySelector('.empty-state');
    if (emptyState) {
        emptyState.remove();
    }

    // 添加用户消息
    const userMessageTime = new Date().toLocaleTimeString('zh-CN');
    const userMessage = document.createElement('div');
    userMessage.className = 'message user';
    userMessage.innerHTML = `
        <div class="message-content">
            <div class="message-header">👤 您</div>
            <div class="message-text">${escapeHtml(question)}</div>
            <div class="message-time">${userMessageTime}</div>
        </div>
    `;
    resultContent.appendChild(userMessage);
    
    // 添加系统回复占位符（显示加载中）
    const assistantMessage = document.createElement('div');
    assistantMessage.className = 'message assistant';
    assistantMessage.id = 'currentAssistantMessage';
    assistantMessage.innerHTML = `
        <div class="message-content">
            <div class="message-header">🤖 智能助手</div>
            <div class="message-loading">
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        </div>
    `;
    resultContent.appendChild(assistantMessage);
    
    // 滚动到底部
    resultContent.scrollTop = resultContent.scrollHeight;

    try {
        // 使用 /api/query 端点，根据配置决定是否使用流式输出
        // 后端会根据 RAG_CONFIG['enable_streaming'] 配置决定返回流式还是非流式响应
        const response = await fetch('/api/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: question,
                role: role,
                enable_rewrite: enableRewrite,
                enable_rerank: enableRerank,
                session_id: currentSessionId  // 传递当前session_id
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        // 检查响应类型：流式（text/event-stream）还是普通JSON
        const contentType = response.headers.get('content-type') || '';
        const isStreaming = contentType.includes('text/event-stream');

        // 获取当前助手消息元素
        const assistantMessage = document.getElementById('currentAssistantMessage');
        const messageContent = assistantMessage.querySelector('.message-content');
        let metadata = {};

        if (isStreaming) {
            // 处理流式响应
            await handleStreamingResponse(response, messageContent, assistantMessage, resultContent, metadata);
        } else {
            // 处理普通JSON响应
            await handleJsonResponse(response, messageContent, assistantMessage, resultContent, metadata);
        }
    } catch (error) {
        document.getElementById('loading').classList.remove('show');
        document.getElementById('submitBtn').disabled = false;
        resultContent.innerHTML = `
            <div class="error show">
                请求失败: ${escapeHtml(error.message)}
            </div>
        `;
        document.getElementById('error').textContent = '请求失败: ' + error.message;
        document.getElementById('error').classList.add('show');
    }
}

/**
 * 处理流式响应
 */
async function handleStreamingResponse(response, messageContent, assistantMessage, resultContent, metadata) {
    let fullAnswer = '';

    // 读取流式响应
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
        const { done, value } = await reader.read();
        if (done) {
            break;
        }

        buffer += decoder.decode(value, { stream: true });
        
        // SSE格式：每个消息以 \n\n 结尾
        let parts = buffer.split('\n\n');
        buffer = parts.pop() || ''; // 保留最后一个不完整的消息

        for (const part of parts) {
            if (!part.trim()) continue; // 跳过空行
            
            // 查找 data: 开头的行
            const lines = part.split('\n');
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const jsonStr = line.slice(6);
                        const data = JSON.parse(jsonStr);
                        
                        if (data.type === 'start') {
                            // 开始处理 - 保持加载指示器显示，等待流式内容
                            // 不在这里移除加载指示器，等收到第一个chunk时再移除
                        } else if (data.type === 'metadata') {
                            // 元数据（保存，但不显示在对话中）
                            metadata.intent = data.intent;
                            metadata.route_to = data.route_to;
                            metadata.module = data.module;
                        } else if (data.type === 'chunk') {
                            // 流式内容片段
                            const loadingDiv = messageContent.querySelector('.message-loading');
                            if (loadingDiv) {
                                loadingDiv.remove();
                            }
                            
                            // 获取或创建message-text元素
                            let textDiv = messageContent.querySelector('.message-text');
                            if (!textDiv) {
                                textDiv = document.createElement('div');
                                textDiv.className = 'message-text';
                                const header = messageContent.querySelector('.message-header');
                                messageContent.insertBefore(textDiv, header.nextSibling);
                            }
                            
                            fullAnswer += data.content;
                            textDiv.textContent = fullAnswer;
                            
                            // 自动滚动到底部
                            resultContent.scrollTop = resultContent.scrollHeight;
                        } else if (data.type === 'done') {
                            // 完成
                            metadata.session_id = data.session_id;
                            metadata.turn_id = data.turn_id;
                            metadata.monitor = data.monitor;
                            
                            // 完成流式响应处理
                            finishStreamingResponse(messageContent, assistantMessage, resultContent, metadata, fullAnswer);
                            return;
                        } else if (data.type === 'error') {
                            // 错误
                            handleStreamingError(messageContent, data.message);
                            return;
                        }
                    } catch (e) {
                        console.error('解析SSE数据失败:', e, '原始行:', line);
                    }
                }
            }
        }
    }

    // 处理剩余的buffer
    if (buffer.trim()) {
        const lines = buffer.split('\n');
        for (const line of lines) {
            if (line.startsWith('data: ')) {
                try {
                    const data = JSON.parse(line.slice(6));
                    if (data.type === 'chunk') {
                        const loadingDiv = messageContent.querySelector('.message-loading');
                        if (loadingDiv) {
                            loadingDiv.remove();
                        }
                        let textDiv = messageContent.querySelector('.message-text');
                        if (!textDiv) {
                            textDiv = document.createElement('div');
                            textDiv.className = 'message-text';
                            const header = messageContent.querySelector('.message-header');
                            messageContent.insertBefore(textDiv, header.nextSibling);
                        }
                        fullAnswer += data.content;
                        textDiv.textContent = fullAnswer;
                    } else if (data.type === 'done') {
                        metadata.session_id = data.session_id;
                        metadata.turn_id = data.turn_id;
                        metadata.monitor = data.monitor;
                        finishStreamingResponse(messageContent, assistantMessage, resultContent, metadata, fullAnswer);
                        return;
                    }
                } catch (e) {
                    console.error('解析最后数据失败:', e);
                }
            }
        }
    }
    
    // 如果最终没有收到done消息，显示错误
    if (!metadata.session_id) {
        const loadingDiv = messageContent.querySelector('.message-loading');
        if (loadingDiv) {
            loadingDiv.remove();
        }
        let textDiv = messageContent.querySelector('.message-text');
        if (!textDiv) {
            textDiv = document.createElement('div');
            textDiv.className = 'message-text';
            const header = messageContent.querySelector('.message-header');
            messageContent.insertBefore(textDiv, header.nextSibling);
        }
        textDiv.textContent = fullAnswer || '响应中断，请重试';
        textDiv.style.color = '#d32f2f';
        document.getElementById('loading').classList.remove('show');
        document.getElementById('submitBtn').disabled = false;
    }
}

/**
 * 完成流式响应处理
 */
function finishStreamingResponse(messageContent, assistantMessage, resultContent, metadata, fullAnswer) {
    // 移除加载指示器
    const loadingDiv = messageContent.querySelector('.message-loading');
    if (loadingDiv) {
        loadingDiv.remove();
    }
    
    // 确保有message-text元素
    let textDiv = messageContent.querySelector('.message-text');
    if (!textDiv) {
        textDiv = document.createElement('div');
        textDiv.className = 'message-text';
        const header = messageContent.querySelector('.message-header');
        messageContent.insertBefore(textDiv, header.nextSibling);
    }
    
    // 添加时间戳
    if (!messageContent.querySelector('.message-time')) {
        const timeDiv = document.createElement('div');
        timeDiv.className = 'message-time';
        timeDiv.textContent = new Date().toLocaleTimeString('zh-CN');
        messageContent.appendChild(timeDiv);
    }
    
    // 添加RAG域信息（意图、路由、模块）
    if (!messageContent.querySelector('.message-rag-info')) {
        const ragInfoDiv = document.createElement('div');
        ragInfoDiv.className = 'message-rag-info';
        const ragInfo = [];
        if (metadata.intent) ragInfo.push(`意图: ${metadata.intent}`);
        if (metadata.route_to) ragInfo.push(`路由: ${metadata.route_to}`);
        if (metadata.module) ragInfo.push(`模块: ${metadata.module}`);
        if (ragInfo.length > 0) {
            ragInfoDiv.innerHTML = ragInfo.join(' | ');
            messageContent.appendChild(ragInfoDiv);
        }
    }
    
    // 添加Token信息（在RAG信息之后）
    if (metadata.monitor && !messageContent.querySelector('.message-info')) {
        const infoDiv = document.createElement('div');
        infoDiv.className = 'message-info';
        infoDiv.innerHTML = `Token: ${metadata.monitor.total_tokens || 0} | 耗时: ${Math.round(metadata.monitor.total_latency_ms || 0)}ms | 调用: ${metadata.monitor.call_count || 0}次`;
        messageContent.appendChild(infoDiv);
    }
    
    // 移除id，使其成为普通消息
    assistantMessage.removeAttribute('id');
    
    // 更新全局session_id
    if (metadata.session_id) {
        currentSessionId = metadata.session_id;
        loadConversationHistory();
    }
    
    document.getElementById('loading').classList.remove('show');
    document.getElementById('submitBtn').disabled = false;
    
    // 滚动到底部
    resultContent.scrollTop = resultContent.scrollHeight;
}

/**
 * 处理流式响应错误
 */
function handleStreamingError(messageContent, errorMessage) {
    const loadingDiv = messageContent.querySelector('.message-loading');
    if (loadingDiv) {
        loadingDiv.remove();
    }
    const errorDiv = document.createElement('div');
    errorDiv.textContent = '❌ 错误: ' + errorMessage;
    errorDiv.style.color = '#d32f2f';
    messageContent.appendChild(errorDiv);
    
    document.getElementById('error').textContent = '错误: ' + errorMessage;
    document.getElementById('error').classList.add('show');
    document.getElementById('loading').classList.remove('show');
    document.getElementById('submitBtn').disabled = false;
}

/**
 * 处理普通JSON响应
 */
async function handleJsonResponse(response, messageContent, assistantMessage, resultContent, metadata) {
    const data = await response.json();
    
    // 隐藏加载状态
    document.getElementById('loading').classList.remove('show');
    document.getElementById('submitBtn').disabled = false;

    if (data.success) {
        // 更新全局session_id
        if (data.session_id) {
            currentSessionId = data.session_id;
            loadConversationHistory();
        }

        // 移除加载指示器
        const loadingDiv = messageContent.querySelector('.message-loading');
        if (loadingDiv) {
            loadingDiv.remove();
        }
        
        // 获取或创建内容div（使用message-text类）
        let contentDiv = messageContent.querySelector('.message-text');
        if (!contentDiv) {
            contentDiv = document.createElement('div');
            contentDiv.className = 'message-text';
            const header = messageContent.querySelector('.message-header');
            messageContent.insertBefore(contentDiv, header.nextSibling);
        }
        
        // 显示答案
        contentDiv.textContent = data.answer || '';
        
        // 添加时间戳
        if (!messageContent.querySelector('.message-time')) {
            const timeDiv = document.createElement('div');
            timeDiv.className = 'message-time';
            timeDiv.textContent = new Date().toLocaleTimeString('zh-CN');
            messageContent.appendChild(timeDiv);
        }
        
        // 添加RAG域信息（意图、路由、模块）
        if (!messageContent.querySelector('.message-rag-info')) {
            const ragInfoDiv = document.createElement('div');
            ragInfoDiv.className = 'message-rag-info';
            const ragInfo = [];
            if (data.intent) ragInfo.push(`意图: ${data.intent}`);
            if (data.route_to) ragInfo.push(`路由: ${data.route_to}`);
            if (data.module) ragInfo.push(`模块: ${data.module}`);
            if (ragInfo.length > 0) {
                ragInfoDiv.innerHTML = ragInfo.join(' | ');
                messageContent.appendChild(ragInfoDiv);
            }
        }
        
        // 添加Token信息（在RAG信息之后）
        if (data.monitor && !messageContent.querySelector('.message-info')) {
            const infoDiv = document.createElement('div');
            infoDiv.className = 'message-info';
            infoDiv.innerHTML = `Token: ${data.monitor.total_tokens || 0} | 耗时: ${Math.round(data.monitor.total_latency_ms || 0)}ms | 调用: ${data.monitor.call_count || 0}次`;
            messageContent.appendChild(infoDiv);
        }
        
        // 移除id，使其成为普通消息
        assistantMessage.removeAttribute('id');
        
        // 滚动到底部
        resultContent.scrollTop = resultContent.scrollHeight;
    } else {
        // 显示错误
        const loadingDiv = messageContent.querySelector('.message-loading');
        if (loadingDiv) {
            loadingDiv.remove();
        }
        const errorDiv = document.createElement('div');
        errorDiv.textContent = '❌ 查询失败: ' + (data.error || '未知错误');
        errorDiv.style.color = '#d32f2f';
        messageContent.appendChild(errorDiv);
        
        document.getElementById('error').textContent = '查询失败: ' + (data.error || '未知错误');
        document.getElementById('error').classList.add('show');
    }
}

// ========== 工具函数 ==========
/**
 * HTML转义函数
 * @param {string} text - 需要转义的文本
 * @returns {string} 转义后的HTML
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ========== 事件监听 ==========
// 支持回车键提交（Ctrl+Enter）
document.getElementById('questionInput').addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && e.ctrlKey) {
        submitQuery();
    }
});

