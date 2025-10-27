class ChatApp {
    constructor() {
        this.chatMessages = document.getElementById('chatMessages');
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.typingIndicator = document.getElementById('typingIndicator');
        this.searchResults = document.getElementById('searchResults');
        this.vectorMemoryBtn = document.getElementById('vectorMemoryBtn');
        this.graphMemoryBtn = document.getElementById('graphMemoryBtn');
        
        // 新增：页面切换相关元素
        this.chatMenuItem = document.getElementById('chatMenuItem');
        this.memoryMenuItem = document.getElementById('memoryMenuItem');
        this.scheduleMenuItem = document.getElementById('scheduleMenuItem');
        this.notificationMenuItem = document.getElementById('notificationMenuItem');
        this.chatView = document.getElementById('chatView');
        this.memoryPanel = document.getElementById('memoryPanel');
        this.schedulePanel = document.getElementById('schedulePanel');
        this.notificationPanel = document.getElementById('notificationPanel');
        
        // 新增：Memory导航按钮相关元素
        this.memoryBankBtn = document.getElementById('memoryBankBtn');
        this.importMemoryBtn = document.getElementById('importMemoryBtn');
        this.memoryBankView = document.getElementById('memoryBankView');
        this.importMemoryView = document.getElementById('importMemoryView');
        
        // 多层视图相关状态
        this.currentViewMode = 'flat'; // 'flat' or 'layered'
        this.currentLayerMode = 'month'; // 'week' or 'month'
        this.layeredGraphData = null;
        this.originalGraphData = null;
        this.mainNetwork = null;
        this.mainNodes = null;
        this.mainEdges = null;
        
        this.initEventListeners();
    }

    initEventListeners() {
        // 发送按钮点击事件
        this.sendButton.addEventListener('click', () => this.sendMessage());
        
        // 清除按钮点击事件
        document.getElementById('clearButton').addEventListener('click', () => this.clearAll());
        
        // 回车键发送消息
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // 输入框焦点
        this.messageInput.focus();
        
        // 内存模式按钮点击事件
        this.vectorMemoryBtn.addEventListener('click', () => this.switchMemoryMode('vector'));
        this.graphMemoryBtn.addEventListener('click', () => this.switchMemoryMode('graph'));
        
        // 新增：菜单栏点击事件
        this.chatMenuItem.addEventListener('click', () => this.switchToView('chat'));
        this.memoryMenuItem.addEventListener('click', () => this.switchToView('memory'));
        this.scheduleMenuItem.addEventListener('click', () => this.switchToView('schedule'));
        this.notificationMenuItem.addEventListener('click', () => this.switchToView('notification'));
        
        // 新增：Memory导航按钮点击事件
        this.memoryBankBtn.addEventListener('click', () => this.switchMemoryView('bank'));
        this.importMemoryBtn.addEventListener('click', () => this.switchMemoryView('import'));
        
        // 初始化Import Memory页面的交互功能
        this.initImportMemoryInteractions();
        
        // 初始化日程模块
        this.initScheduleModule();
    }

    // 新增：初始化Import Memory页面的交互功能
    initImportMemoryInteractions() {
        const importColumns = document.querySelectorAll('.import-column');
        
        importColumns.forEach(column => {
            const dropZone = column.querySelector('.drop-zone');
            const fileInput = column.querySelector('.file-input');
            const loadingOverlay = column.querySelector('.loading-overlay');
            
            // 点击拖放区域触发文件选择器
            dropZone.addEventListener('click', () => {
                fileInput.click();
            });
            
            // 文件选择器变化事件
            fileInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    this.handleFileUpload(column, e.target.files[0]);
                }
            });
            
            // 拖放事件处理
            dropZone.addEventListener('dragover', (e) => {
                e.preventDefault();
                dropZone.classList.add('drag-over');
            });
            
            dropZone.addEventListener('dragleave', (e) => {
                e.preventDefault();
                dropZone.classList.remove('drag-over');
            });
            
            dropZone.addEventListener('drop', (e) => {
                e.preventDefault();
                dropZone.classList.remove('drag-over');
                
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    this.handleFileUpload(column, files[0]);
                }
            });
        });
    }

    // 新增：处理文件上传
    async handleFileUpload(column, file) {
        const loadingOverlay = column.querySelector('.loading-overlay');
        const dropZone = column.querySelector('.drop-zone');
        const fileInput = column.querySelector('.file-input');
        const progressBar = loadingOverlay.querySelector('.progress-fill');
        
        // 显示进度条
        loadingOverlay.classList.add('active');
        
        try {
            // 创建FormData对象
            const formData = new FormData();
            formData.append('file', file);
            formData.append('type', dropZone.dataset.type);
            
            // 发送文件到后端
            const response = await fetch('/api/upload-file', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.success) {
                const taskId = result.task_id;
                
                // 开始轮询进度
                await this.pollProgress(taskId, progressBar);
                
                // 处理完成后的操作
                this.handleUploadComplete(dropZone, file);
                
            } else {
                throw new Error(result.error || '文件上传失败');
            }
            
        } catch (error) {
            console.error('File upload error:', error);
            progressBar.style.width = '0%';
            
            // 3秒后隐藏错误信息
            setTimeout(() => {
                loadingOverlay.classList.remove('active');
                this.resetProgressBar(progressBar);
            }, 3000);
        }
        
        // 重置文件输入
        fileInput.value = '';
    }
    
    async pollProgress(taskId, progressBar) {
        return new Promise((resolve, reject) => {
            const pollInterval = setInterval(async () => {
                try {
                    const response = await fetch(`/api/upload-progress/${taskId}`);
                    const data = await response.json();
                    
                    if (data.success) {
                        // 更新进度条
                        const progress = data.progress;
                        progressBar.style.width = `${progress}%`;
                        
                        // 如果完成，停止轮询
                        if (data.is_complete || progress >= 100) {
                            clearInterval(pollInterval);
                            resolve();
                        }
                    } else {
                        clearInterval(pollInterval);
                        reject(new Error(data.error || '获取进度失败'));
                    }
                } catch (error) {
                    clearInterval(pollInterval);
                    reject(error);
                }
            }, 500); // 每500ms查询一次进度
        });
    }
    
    handleUploadComplete(dropZone, file) {
        const loadingOverlay = dropZone.querySelector('.loading-overlay');
        const dropContent = dropZone.querySelector('.drop-content');
        const dropText = dropZone.querySelector('.drop-text');
        const originalHTML = dropText.innerHTML;
        
        // 隐藏进度条，显示成功信息
        loadingOverlay.classList.remove('active');
        
        // 在drop-content区域显示成功信息，避免被进度条遮挡
        dropText.innerHTML = `<strong>Processed: ${file.name}</strong><br><span style="color: #4a90e2;">Import Successful!</span>`;
        
        // 更新用户状态为已导入文件
        this.updateImportStatus(true);
        
        // 文件上传完成后，重新加载图谱
        if (this.loadDefaultGraph) {
            this.loadDefaultGraph();
        }
        
        // 如果当前在Memory Bank视图，确保图谱可见
        const memoryBankView = document.getElementById('memoryBankView');
        const memoryPanel = document.getElementById('memoryPanel');
        if (memoryPanel && memoryPanel.style.display !== 'none' && 
            memoryBankView && memoryBankView.classList.contains('active')) {
            // 确保图谱容器可见
            const graphContainer = document.getElementById('mainGraphContainer');
            if (graphContainer) {
                const loadingDiv = graphContainer.querySelector('.graph-loading');
                if (loadingDiv) {
                    loadingDiv.style.display = 'flex';
                }
            }
        }
        
        // 2秒后恢复原始状态
        setTimeout(() => {
            dropText.innerHTML = originalHTML;
            
            // 重置进度条
            const progressBar = loadingOverlay.querySelector('.progress-fill');
            this.resetProgressBar(progressBar);
        }, 2000);
    }
    
    resetProgressBar(progressBar) {
        progressBar.style.width = '0%';
    }

    // 新增：页面切换功能
    switchToView(viewName) {
        // 移除所有菜单项的active状态
        this.chatMenuItem.classList.remove('active');
        this.memoryMenuItem.classList.remove('active');
        this.scheduleMenuItem.classList.remove('active');
        this.notificationMenuItem.classList.remove('active');
        
        // 隐藏所有视图
        this.chatView.style.display = 'none';
        this.memoryPanel.classList.remove('active');
        this.schedulePanel.style.display = 'none';
        this.notificationPanel.style.display = 'none';
        
        if (viewName === 'chat') {
            // 显示聊天视图
            this.chatMenuItem.classList.add('active');
            this.chatView.style.display = 'flex';
            // 重新聚焦输入框
            setTimeout(() => {
                this.messageInput.focus();
            }, 100);
        } else if (viewName === 'memory') {
            // 显示Memory页面
            this.memoryMenuItem.classList.add('active');
            this.memoryPanel.classList.add('active');
        } else if (viewName === 'schedule') {
            // 显示Schedule页面
            this.scheduleMenuItem.classList.add('active');
            this.schedulePanel.style.display = 'block';
        } else if (viewName === 'notification') {
            // 显示Notification页面
            this.notificationMenuItem.classList.add('active');
            this.notificationPanel.style.display = 'block';
        }
        
        console.log(`Switched to ${viewName} view`);
    }

    // 新增：Memory内部视图切换功能
    switchMemoryView(viewName) {
        // 移除所有Memory导航按钮的active状态
        this.memoryBankBtn.classList.remove('active');
        this.importMemoryBtn.classList.remove('active');
        
        // 隐藏所有Memory视图
        this.memoryBankView.classList.remove('active');
        this.importMemoryView.classList.remove('active');
        
        if (viewName === 'bank') {
            // 显示Memory Bank视图
            this.memoryBankBtn.classList.add('active');
            this.memoryBankView.classList.add('active');
        } else if (viewName === 'import') {
            // 显示Import Memory视图
            this.importMemoryBtn.classList.add('active');
            this.importMemoryView.classList.add('active');
        }
        
        console.log(`Switched to memory ${viewName} view`);
    }

    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message) return;

        // 禁用输入
        this.setInputEnabled(false);
        
        // 显示用户消息
        this.addMessage(message, 'user');
        
        // 清空输入框
        this.messageInput.value = '';
        
        // 显示打字指示器
        this.showTypingIndicator();
        
        try {
            // 获取当前搜索模式
            const currentMode = this.vectorMemoryBtn.classList.contains('active') ? 'vector' : 'graph';
            
            // 发送请求到后端
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    message: message,
                    search_mode: currentMode
                })
            });
            
            const data = await response.json();
            
            // 检查是否为简单问题，如果不是简单问题才显示搜索结果
            if (!data.is_simple_question) {
                // 等待搜索结果完全显示完毕
                await this.showSearchResults(message);
            }
            
            // 检查是否为简单问题，如果是则减少延迟时间
            let thinkingTime;
            if (data.is_simple_question) {
                thinkingTime = 200 + Math.random() * 300; // 简单问题：200-500ms
            } else {
                thinkingTime = 1000 + Math.random() * 2000; // 普通问题：1-3秒
            }
            
            // 模拟思考时间
            await this.delay(thinkingTime);
            
            // 隐藏打字指示器
            this.hideTypingIndicator();
            
            if (data.success) {
                // 显示AI回复，传递是否为简单问题的标识
                this.addMessage(data.reply, 'bot', data.is_simple_question);
            } else {
                // 显示错误消息
                this.addMessage('Sorry, an error occurred: ' + data.error, 'bot');
            }
        } catch (error) {
            // 隐藏打字指示器
            this.hideTypingIndicator();
            
            // 显示网络错误
            this.addMessage('Sorry, there was a network connection issue. Please try again later.', 'bot');
            console.error('Chat error:', error);
        } finally {
            // 重新启用输入
            this.setInputEnabled(true);
            this.messageInput.focus();
        }
    }

    addMessage(content, type, isSimpleQuestion = false) {
        // 移除欢迎消息
        const welcomeMessage = this.chatMessages.querySelector('.welcome-message');
        if (welcomeMessage) {
            welcomeMessage.remove();
        }
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        
        const now = new Date();
        const timeString = now.toLocaleTimeString('zh-CN', { 
            hour: '2-digit', 
            minute: '2-digit' 
        });
        
        // 如果是机器人回复，使用打字机效果
        if (type === 'bot') {
            // 只有非简单问题才显示"Add to Schedule"按钮
            const actionsHtml = isSimpleQuestion ? '' : `
                <div class="message-actions" style="display: none;">
                    <button class="add-to-schedule-btn" onclick="chatApp.addToSchedule('${this.escapeHtml(content).replace(/'/g, "\\'")}')">
                        📅 Add to Schedule
                    </button>
                </div>
            `;
            
            messageDiv.innerHTML = `
                <div class="message-content">
                    <div class="message-text"></div>
                    <div class="message-time">${timeString}</div>
                    ${actionsHtml}
                </div>
            `;
            
            this.chatMessages.appendChild(messageDiv);
            this.scrollToBottom();
            
            // 开始打字机效果，传递按钮容器以便在完成后显示（如果存在的话）
            const actionsDiv = messageDiv.querySelector('.message-actions');
            this.typewriterEffect(messageDiv.querySelector('.message-text'), content, 4, actionsDiv);
        } else {
            // 用户消息直接显示
            messageDiv.innerHTML = `
                <div class="message-content">
                    ${this.escapeHtml(content)}
                    <div class="message-time">${timeString}</div>
                </div>
            `;
            
            this.chatMessages.appendChild(messageDiv);
            this.scrollToBottom();
        }
    }

    // 新增：打字机效果函数
    async typewriterEffect(element, text, baseSpeed = 4, actionsDiv = null) {
        element.innerHTML = '';
        
        // 定义几种不同的打字速度（毫秒）
        const speeds = [
            baseSpeed * 0.5,  // 很快
            baseSpeed * 0.8,  // 较快
            baseSpeed,        // 正常
            baseSpeed * 1.5,  // 较慢
            baseSpeed * 3   // 慢
        ];
        
        for (let i = 0; i < text.length; i++) {
            element.innerHTML += this.escapeHtml(text.charAt(i));
            this.scrollToBottom();
            
            // 随机选择一个速度
            const randomSpeed = speeds[Math.floor(Math.random() * speeds.length)];
            
            // 如果遇到标点符号，使用较长的停顿
            const char = text.charAt(i);
            const pauseChars = ['.', '!', '?', '。', '！', '？', '，', ',', '；', ';'];
            const delay = pauseChars.includes(char) ? randomSpeed * 3 : randomSpeed;
            
            await this.delay(delay);
        }
        
        // 打字完成后移除光标
        element.classList.add('typing-complete');
        
        // 如果传递了按钮容器，显示按钮
        if (actionsDiv) {
            // 添加一个短暂的延迟，让用户感知到打字已完成
            await this.delay(1500);
            actionsDiv.style.display = 'block';
            // 添加淡入动画效果
            actionsDiv.style.opacity = '0';
            actionsDiv.style.transition = 'opacity 0.3s ease-in';
            setTimeout(() => {
                actionsDiv.style.opacity = '1';
            }, 50);
        }
    }

    // 新增：添加到日程功能
    async addToSchedule(content) {
        try {
            // 发送请求到后端API
            const response = await fetch('/api/add-to-schedule', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    content: content
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                // 显示成功提示
                this.showNotification(data.message, 'success');
                
                // 重新加载通知数据，以便显示Memory Update通知
                if (this.loadNotifications) {
                    await this.loadNotifications('all');
                }
                
                // 重新加载schedule数据，以便显示Friends Gathering事件
                if (this.loadScheduleData) {
                    await this.loadScheduleData();
                }
            } else {
                this.showNotification('Failed to add to schedule: ' + data.error, 'error');
            }
        } catch (error) {
            console.error('Error adding to schedule:', error);
            this.showNotification('Failed to add to schedule', 'error');
        }
    }

    // 新增：显示网页内提示
    showNotification(message, type = 'info') {
        // 创建提示元素
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        
        // 添加到页面
        document.body.appendChild(notification);
        
        // 显示动画
        setTimeout(() => {
            notification.classList.add('show');
        }, 100);
        
        // 3秒后自动消失
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 3000);
    }

    showTypingIndicator() {
        this.typingIndicator.style.display = 'block';
        this.scrollToBottom();
    }

    hideTypingIndicator() {
        this.typingIndicator.style.display = 'none';
    }

    setInputEnabled(enabled) {
        this.messageInput.disabled = !enabled;
        this.sendButton.disabled = !enabled;
    }

    scrollToBottom() {
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    renderGraph(containerId, graphData) {
        // 等待DOM元素创建完成
        setTimeout(() => {
            const container = document.getElementById(containerId);
            if (!container) {
                console.error('Graph container not found:', containerId);
                return;
            }

            // 转换节点数据格式
            const defaultColor = '#4A90E2'; // 默认蓝色
            const nodes = new vis.DataSet(graphData.nodes.map(node => ({
                id: node.id,
                label: node.id, // 显示节点ID
                title: this.createNodeTooltip(node), // 美化的节点浮窗
                color: {
                    background: defaultColor,
                    border: this.darkenColor(defaultColor, 0.2),
                    highlight: {
                        background: this.lightenColor(defaultColor, 0.2),
                        border: defaultColor
                    }
                },
                font: {
                    color: '#333',
                    size: 30,
                    face: 'Arial'
                },
                shape: 'dot',
                size: 25
            })));

            // 转换边数据格式
        const edges = new vis.DataSet(graphData.edges.map((edge, index) => ({
            id: edge.id || `${edge.from}-${edge.to}-${index}`,
            from: edge.from,
            to: edge.to,
            label: '',
            title: this.createEdgeTooltip(edge), // 美化的边浮窗
            description: edge.description,
            timestamp: edge.timestamp,
            color: {
                color: '#848484',
                highlight: '#333'
            },
            font: {
                color: '#666',
                size: 12,
                strokeWidth: 2,
                strokeColor: '#fff'
            },
            arrows: {
                to: {
                    enabled: false
                }
            },
            smooth: {
                type: 'continuous',
                roundness: 0.2
            }
        })));

            // 创建网络图
            const data = { nodes: nodes, edges: edges };
            const options = {
                layout: {
                    improvedLayout: true,
                    hierarchical: false
                },
                physics: {
                    enabled: true,
                    stabilization: { iterations: 150 },
                    barnesHut: {
                        gravitationalConstant: -800,
                        centralGravity: 0.05,
                        springLength: 180,
                        springConstant: 0.02,
                        damping: 0.15,
                        avoidOverlap: 0.3
                    }
                },
                interaction: {
                    hover: true,
                    tooltipDelay: 200,
                    hideEdgesOnDrag: false,
                    hideNodesOnDrag: false
                },
                nodes: {
                    borderWidth: 2,
                    shadow: {
                        enabled: true,
                        color: 'rgba(0,0,0,0.2)',
                        size: 5,
                        x: 2,
                        y: 2
                    }
                },
                edges: {
                    width: 2,
                    shadow: {
                        enabled: true,
                        color: 'rgba(0,0,0,0.1)',
                        size: 3,
                        x: 1,
                        y: 1
                    }
                }
            };

            const network = new vis.Network(container, data, options);

            // 添加交互事件
            network.on('click', (params) => {
                if (params.nodes.length > 0) {
                    const nodeId = params.nodes[0];
                    const node = nodes.get(nodeId);
                    console.log('Clicked node:', node);
                } else if (params.edges.length > 0) {
                    const edgeId = params.edges[0];
                    const edge = edges.get(edgeId);
                    this.showEdgePopup(edge, params.pointer.DOM);
                }
            });

            // 添加边的悬停事件
            network.on('hoverEdge', (params) => {
                const edge = edges.get(params.edge);
                // vis.js 已经通过 title 属性显示浮窗，这里可以添加额外的交互
            });

            // 网络稳定后调整视图
            network.once('stabilizationIterationsDone', () => {
                network.fit({
                    animation: {
                        duration: 1000,
                        easingFunction: 'easeInOutQuad'
                    }
                });
            });
        }, 100);
    }

    // 创建美化的节点浮窗
    createNodeTooltip(node) {
        const description = node.description || '暂无描述';
        // 限制描述长度，避免浮窗过长
        const truncatedDescription = description.length > 100 ? 
            description.substring(0, 100) + '...' : description;
        return `Node id: ${node.id}\n\nDescription: ${truncatedDescription}`;
    }

    // 创建美化的边浮窗
    createEdgeTooltip(edge) {
        const description = edge.description || '暂无描述';
        // 限制描述长度，避免浮窗过长
        const truncatedDescription = description.length > 80 ? 
            description.substring(0, 80) + '...' : description;
        const timestamp = edge.timestamp ? new Date(edge.timestamp).toLocaleString('zh-CN') : '未知时间';
        return `🔗 ${edge.from} → ${edge.to}\n\nDescription: ${truncatedDescription}\nTimeStamp: ${timestamp}`;
    }

    // 显示边的详细信息弹窗
    showEdgePopup(edge, position) {
        // 移除已存在的弹窗
        const existingPopup = document.getElementById('edge-popup');
        if (existingPopup) {
            existingPopup.remove();
        }

        const description = edge.description || '暂无描述';
        const timestamp = edge.timestamp ? new Date(edge.timestamp).toLocaleString('zh-CN') : '未知时间';

        const popup = document.createElement('div');
        popup.id = 'edge-popup';
        popup.innerHTML = `
            <div style="
                position: fixed;
                background: white;
                border: 1px solid #ddd;
                border-radius: 12px;
                padding: 20px;
                box-shadow: 0 8px 25px rgba(0,0,0,0.15);
                z-index: 10000;
                max-width: 320px;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                left: ${position.x + 10}px;
                top: ${position.y - 10}px;
            ">
                <div style="
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 15px;
                ">
                    <h3 style="
                        margin: 0;
                        color: #333;
                        font-size: 16px;
                        font-weight: 600;
                    ">🔗 边详情</h3>
                    <button onclick="document.getElementById('edge-popup').remove()" style="
                        background: none;
                        border: none;
                        font-size: 18px;
                        cursor: pointer;
                        color: #999;
                        padding: 0;
                        width: 24px;
                        height: 24px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                    ">×</button>
                </div>
                <div style="margin-bottom: 12px;">
                    <strong style="color: #555;">连接:</strong>
                    <span style="
                        background: #003f88;
                        color: white;
                        padding: 4px 8px;
                        border-radius: 4px;
                        font-size: 12px;
                        margin-left: 8px;
                    ">${edge.from} → ${edge.to}</span>
                </div>
                <div style="margin-bottom: 12px;">
                    <strong style="color: #555;">描述:</strong>
                    <div style="
                        margin-top: 6px;
                        padding: 10px;
                        background: #f8f9fa;
                        border-radius: 6px;
                        color: #333;
                        line-height: 1.4;
                        word-wrap: break-word;
                    ">${description}</div>
                </div>
                <div>
                    <strong style="color: #555;">时间:</strong>
                    <span style="
                        color: #666;
                        font-size: 14px;
                        margin-left: 8px;
                    ">${timestamp}</span>
                </div>
            </div>
        `;

        document.body.appendChild(popup);

        // 点击外部关闭弹窗
        setTimeout(() => {
            document.addEventListener('click', function closePopup(e) {
                if (!popup.contains(e.target)) {
                    popup.remove();
                    document.removeEventListener('click', closePopup);
                }
            });
        }, 100);
    }

    // 颜色处理工具函数
    darkenColor(color, factor) {
        const hex = color.replace('#', '');
        const r = parseInt(hex.substr(0, 2), 16);
        const g = parseInt(hex.substr(2, 2), 16);
        const b = parseInt(hex.substr(4, 2), 16);
        
        const newR = Math.round(r * (1 - factor));
        const newG = Math.round(g * (1 - factor));
        const newB = Math.round(b * (1 - factor));
        
        return `#${newR.toString(16).padStart(2, '0')}${newG.toString(16).padStart(2, '0')}${newB.toString(16).padStart(2, '0')}`;
    }

    lightenColor(color, factor) {
        const hex = color.replace('#', '');
        const r = parseInt(hex.substr(0, 2), 16);
        const g = parseInt(hex.substr(2, 2), 16);
        const b = parseInt(hex.substr(4, 2), 16);
        
        const newR = Math.round(r + (255 - r) * factor);
        const newG = Math.round(g + (255 - g) * factor);
        const newB = Math.round(b + (255 - b) * factor);
        
        return `#${newR.toString(16).padStart(2, '0')}${newG.toString(16).padStart(2, '0')}${newB.toString(16).padStart(2, '0')}`;
    }

    async showSearchResults(query) {
        return new Promise(async (resolve) => {
            // 移除欢迎消息
            const welcomeMessage = this.searchResults.querySelector('.welcome-message');
            if (welcomeMessage) {
                welcomeMessage.remove();
            }

            // 显示加载状态
            this.searchResults.innerHTML = '<div class="loading-message">🔍 Searching database...</div>';
            
            // 添加搜索延迟，模拟真实的搜索过程
            const searchDelay = 300 + Math.random() * 300; // 300-600ms的随机延迟
            await this.delay(searchDelay);
            
            try {
                // 获取当前选中的检索模式
                const currentMode = this.vectorMemoryBtn.classList.contains('active') ? 'vector' : 'graph';
                
                // 调用后端检索API
                const response = await fetch('/api/search', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        query: query,
                        mode: currentMode,
                        top_k: 5
                    })
                });
                
                const data = await response.json();
                
                // 清空加载状态
                this.searchResults.innerHTML = '';
                
                if (data.success && data.results.length > 0) {
                    // 逐个显示检索结果，增加动画效果
                    for (let index = 0; index < data.results.length; index++) {
                        const result = data.results[index];
                        await this.delay(100 + Math.random() * 100); // 每个结果间隔50-100ms
                        
                        const resultDiv = document.createElement('div');
                        resultDiv.className = 'search-result-item';
                        resultDiv.style.opacity = '0';
                        resultDiv.style.transform = 'translateY(10px)';
                        resultDiv.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
                        
                        // 根据检索类型显示不同内容
                        let contentHtml = '';
                        if (result.retrieval_type === 'graph' && result.content.nodes && result.content.edges) {
                            // 图形检索结果：显示图形可视化
                            const graphId = `graph-${result.id}-${index}`;
                            contentHtml = `<div id="${graphId}" class="graph-container"></div>`;
                            
                            // 构建时间区间显示
                            const timeInterval = result.time_interval ? 
                                `[${result.time_interval.t_start}, ${result.time_interval.t_end}]` : 
                                result.created_at;
                            
                            resultDiv.innerHTML = `
                                <div class="search-result-title">
                                    Result: ${index + 1}
                                    <div class="search-result-meta">
                                        <span class="search-result-query-type">${result.query_type || 'N/A'}</span>
                                    </div>
                                </div>
                                <div class="search-result-query">Query: ${this.escapeHtml(result.query || 'N/A')}</div>
                                <div class="search-result-query">Query Time Interval: ${timeInterval}</div>
                                <div class="search-result-content">${contentHtml}</div>
                            `;
                            this.searchResults.appendChild(resultDiv);
                            
                            // 动态设置query_type的背景色
                            const queryTypeElement = resultDiv.querySelector('.search-result-query-type');
                            if (queryTypeElement && result.query_type) {
                                if (result.query_type.includes('short-term')) {
                                    queryTypeElement.style.backgroundColor = '#4a90e2';
                                } else if (result.query_type.includes('long-term')) {
                                    queryTypeElement.style.backgroundColor = '#ff9800';
                                } else {
                                    queryTypeElement.style.backgroundColor = '#9e9e9e';
                                }
                            }
                            
                            // 渲染图形
                            this.renderGraph(graphId, result.content);
                        } else {
                            // Vector检索结果：根据类型显示不同内容
                            const typeIcon = this.getTypeIcon(result.type);
                            const typeLabel = this.getTypeLabel(result.type);
                            
                            if (result.type === 'conversation' && Array.isArray(result.content)) {
                                // 对话类型
                                contentHtml = result.content.map(msg => {
                                    const roleIcon = msg.role === 'user' ? '👤' : '🤖';
                                    const roleClass = msg.role === 'user' ? 'user-msg' : 'assistant-msg';
                                    return `<div class="conversation-message ${roleClass}">
                                        <span class="role-icon">${roleIcon}</span>
                                        <span class="message-text">${this.escapeHtml(msg.content)}</span>
                                    </div>`;
                                }).join('');
                            } else if (result.type === 'email') {
                                // 邮件类型
                                contentHtml = `
                                    <div class="email-content">
                                        <div class="email-header">
                                            <div class="email-subject">📧 ${this.escapeHtml(result.subject || 'No Subject')}</div>
                                            <div class="email-meta">
                                                <div class="email-sender">发件人: ${this.escapeHtml(result.sender || '未知发件人')}</div>
                                    <div class="email-recipient">收件人: ${this.escapeHtml(result.recipient || '未知收件人')}</div>
                                            </div>
                                        </div>
                                        <div class="email-body">${this.escapeHtml(result.content || '').replace(/\n/g, '<br>')}</div>
                                    </div>
                                `;
                            } else if (result.type === 'activity') {
                                // 活动类型 - 简化为单一英文文本描述
                                const title = result.title || 'Activity';
                                const duration = result.duration || '';
                                const location = result.location || '';
                                const content = result.content || '';
                                
                                // 构建简洁的英文描述
                                let description = `${title}`;
                                if (duration) {
                                    description += ` for ${duration}`;
                                }
                                if (location) {
                                    description += ` at ${location}`;
                                }
                                if (content) {
                                    description += `. ${content}`;
                                }
                                
                                contentHtml = `
                                    <div class="activity-content">
                                        <div class="activity-simple-description">${this.escapeHtml(description)}</div>
                                    </div>
                                `;
                            } else {
                                // 其他类型或旧格式兼容
                                if (Array.isArray(result.content)) {
                                    contentHtml = result.content.map(msg => {
                                        const roleIcon = msg.role === 'user' ? '👤' : '🤖';
                                        const roleClass = msg.role === 'user' ? 'user-msg' : 'assistant-msg';
                                        return `<div class="conversation-message ${roleClass}">
                                            <span class="role-icon">${roleIcon}</span>
                                            <span class="message-text">${this.escapeHtml(msg.content)}</span>
                                        </div>`;
                                    }).join('');
                                } else {
                                    contentHtml = this.escapeHtml(result.content || '');
                                }
                            }
                            
                            resultDiv.innerHTML = `
                                <div class="search-result-title">
                                    <span class="result-type-badge" data-type="${result.type || 'unknown'}">
                                        ${typeIcon} ${typeLabel}
                                    </span>
                                    Result: ${index + 1}
                                    <span class="search-result-date">${result.created_at}</span>
                                </div>
                                <div class="search-result-content">${contentHtml}</div>
                            `;
                            this.searchResults.appendChild(resultDiv);
                        }
                        
                        // 添加动画效果，并等待动画完成
                        setTimeout(() => {
                            resultDiv.style.opacity = '1';
                            resultDiv.style.transform = 'translateY(0)';
                        }, 50);
                        
                        // 等待动画完成
                        await this.delay(200); // 等待动画完成 (50ms + 300ms transition)
                    }
                    
                    // 所有搜索结果都显示完毕后，再等待一小段时间确保用户能看到完整结果
                    await this.delay(100);
                    resolve();
                } else {
                    // 显示无结果消息
                    this.searchResults.innerHTML = '<div class="no-results-message">📭 No relevant results found for your query.</div>';
                    resolve();
                }
            } catch (error) {
                console.error('Search API error:', error);
                // 显示错误消息
                this.searchResults.innerHTML = '<div class="error-message">❌ Search service temporarily unavailable. Please try again later.</div>';
                resolve();
            }
        });
    }

    // 注意：模拟检索功能已迁移到后端 app.py 中
    // 前端现在通过 /api/search API 调用后端的检索服务
    
    switchMemoryMode(mode) {
        // 移除所有按钮的active状态
        this.vectorMemoryBtn.classList.remove('active');
        this.graphMemoryBtn.classList.remove('active');
        
        // 为选中的按钮添加active状态
        if (mode === 'vector') {
            this.vectorMemoryBtn.classList.add('active');
        } else if (mode === 'graph') {
            this.graphMemoryBtn.classList.add('active');
        }
        
        console.log(`Switched to ${mode} retrieval mode`);
        
        // 如果当前有查询内容，重新执行检索以显示不同模式的结果
        const currentQuery = this.messageInput.value.trim();
        if (currentQuery) {
            this.showSearchResults(currentQuery);
        }
    }

    // 新增：获取类型图标的辅助函数
    getTypeIcon(type) {
        switch (type) {
            case 'conversation':
                return '💬';
            case 'email':
                return '📧';
            case 'activity':
                return '🏃‍♂️';
            case 'unknown':
                return '❓';
            default:
                return '📄';
        }
    }

    // 新增：获取类型标签的辅助函数
    getTypeLabel(type) {
        switch (type) {
            case 'conversation':
                return 'Conversation';
            case 'email':
                return 'Email';
            case 'activity':
                return 'Activity';
            case 'unknown':
                return '未知';
            default:
                return '其他';
        }
    }

    clearAll() {
         // 清空聊天消息
         this.chatMessages.innerHTML = `
             <div class="welcome-message">
                 👋 Welcome to AI Chatbot!<br>
                 Please enter your message to start the conversation
             </div>
         `;
         
         // 清空搜索结果
         this.searchResults.innerHTML = `
             <div class="welcome-message" style="text-align: center; color: #888; padding: 40px 20px; font-size: 14px;">
                 💡 When you ask a question, relevant<br>database search results will appear here
             </div>
         `;
         
         // 清空输入框
         this.messageInput.value = '';
         
         // 隐藏打字指示器（如果存在）
         const typingIndicator = document.querySelector('.typing-indicator');
         if (typingIndicator) {
             typingIndicator.remove();
         }
         
         // 重新聚焦输入框
         this.messageInput.focus();
         
         console.log('All content cleared - returned to initial state');
     }

    // Memory Repository 相关功能
    async initializeMemoryRepository() {
        // 初始化图谱控制按钮事件
        this.setupGraphControls();
        
        // 初始化时间范围滑动条
        this.initTimeRangeSlider();
        
        // 检查用户状态，只有在已导入文件时才显示图谱
        await this.checkUserStateAndLoadGraph();
    }

    async checkUserStateAndLoadGraph() {
        try {
            const response = await fetch('/api/user-state', {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const data = await response.json();
            
            if (data.success && data.userState && data.userState.hasImportedMemory) {
                // 如果用户已导入文件，显示图谱
                this.loadDefaultGraph();
            } else {
                // 如果用户未导入文件，显示提示信息
                this.showNoDataMessage();
            }
        } catch (error) {
            console.error('Error checking user state:', error);
            // 出错时显示提示信息
            this.showNoDataMessage();
        }
    }

    showNoDataMessage() {
        const graphContainer = document.getElementById('mainGraphContainer');
        if (!graphContainer) return;

        graphContainer.innerHTML = `
            <div class="no-data-message">
                <div class="no-data-title"> </div>
            </div>
        `;
    }

    async updateImportStatus(hasImported) {
        try {
            const response = await fetch('/api/update-import-status', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    hasImportedMemory: hasImported
                })
            });
            
            const data = await response.json();
            
            if (!data.success) {
                console.error('Failed to update import status:', data.error);
            }
        } catch (error) {
            console.error('Error updating import status:', error);
        }
    }

    initTimeRangeSlider() {
        // 时间范围相关变量
        this.minTimestamp = new Date('2025-05-01T09:15:00').getTime();
        this.maxTimestamp = new Date().getTime(); // 当前时间
        this.currentStartTime = this.minTimestamp;
        this.currentEndTime = this.maxTimestamp;
        this.originalGraphData = null; // 存储原始图谱数据
        
        // 获取滑动条元素
        const startSlider = document.getElementById('startTimeSlider');
        const endSlider = document.getElementById('endTimeSlider');
        const resetBtn = document.getElementById('resetTimeRangeBtn');
        
        if (!startSlider || !endSlider) return;
        
        // 初始化滑动条值
        startSlider.value = 0;
        endSlider.value = 100;
        
        // 更新显示标签
        this.updateTimeLabels();
        this.updateSliderRange();
        
        // 绑定滑动条事件
        startSlider.addEventListener('input', (e) => {
            const value = parseInt(e.target.value);
            const endValue = parseInt(endSlider.value);
            
            // 确保起始时间不超过结束时间
            if (value >= endValue) {
                e.target.value = endValue - 1;
                return;
            }
            
            this.updateTimeRange();
        });
        
        endSlider.addEventListener('input', (e) => {
            const value = parseInt(e.target.value);
            const startValue = parseInt(startSlider.value);
            
            // 确保结束时间不小于起始时间
            if (value <= startValue) {
                e.target.value = startValue + 1;
                return;
            }
            
            this.updateTimeRange();
        });
        
        // 重置按钮事件
        if (resetBtn) {
            resetBtn.addEventListener('click', () => {
                startSlider.value = 0;
                endSlider.value = 100;
                this.updateTimeRange();
                // 重新加载完整的原始图谱数据
                this.loadDefaultGraph(); 
            });
        }
    }
    
    updateTimeRange() {
        const startSlider = document.getElementById('startTimeSlider');
        const endSlider = document.getElementById('endTimeSlider');
        
        if (!startSlider || !endSlider) return;
        
        const startPercent = parseInt(startSlider.value);
        const endPercent = parseInt(endSlider.value);
        
        // 计算实际时间戳
        const timeRange = this.maxTimestamp - this.minTimestamp;
        this.currentStartTime = this.minTimestamp + (timeRange * startPercent / 100);
        this.currentEndTime = this.minTimestamp + (timeRange * endPercent / 100);
        
        // 更新显示
        this.updateTimeLabels();
        this.updateSliderRange();
        
        // 使用防抖机制避免频繁API调用
        this.debouncedFilterAndRender();
    }
    
    debouncedFilterAndRender() {
        // 清除之前的定时器
        if (this.filterTimeout) {
            clearTimeout(this.filterTimeout);
        }
        
        // 设置新的定时器，300ms后执行
        this.filterTimeout = setTimeout(() => {
            this.filterAndRenderGraph();
        }, 300);
    }
    
    updateTimeLabels() {
        const startLabel = document.getElementById('startTimeLabel');
        const endLabel = document.getElementById('endTimeLabel');
        
        const startDate = new Date(this.currentStartTime);
        const endDate = new Date(this.currentEndTime);
        const isToday = Math.abs(this.currentEndTime - this.maxTimestamp) < 24 * 60 * 60 * 1000;
        
        const formatDate = (date) => {
            return date.toLocaleDateString('en-US', {
                year: 'numeric',
                month: 'short',
                day: '2-digit'
            });
        };
        
        if (startLabel) startLabel.textContent = formatDate(startDate);
        if (endLabel) endLabel.textContent = isToday ? 'Today' : formatDate(endDate);
    }
    
    updateSliderRange() {
        const startSlider = document.getElementById('startTimeSlider');
        const endSlider = document.getElementById('endTimeSlider');
        const sliderRange = document.getElementById('sliderRange');
        
        if (!startSlider || !endSlider || !sliderRange) return;
        
        const startPercent = parseInt(startSlider.value);
        const endPercent = parseInt(endSlider.value);
        
        sliderRange.style.left = startPercent + '%';
        sliderRange.style.width = (endPercent - startPercent) + '%';
    }
    
    getTimeRange() {
        if (!this.originalGraphData || !this.originalGraphData.edges) {
            return null;
        }
        
        const timestamps = this.originalGraphData.edges
            .map(edge => edge.timestamp)
            .filter(timestamp => timestamp)
            .map(timestamp => new Date(timestamp).getTime());
        
        if (timestamps.length === 0) {
            return null;
        }
        
        return {
            min: Math.min(...timestamps),
            max: Math.max(...timestamps)
        };
    }

    filterAndRenderGraph() {
        if (!this.originalGraphData) {
            console.warn('No original graph data available for filtering');
            return;
        }

        const startSlider = document.getElementById('startTimeSlider');
        const endSlider = document.getElementById('endTimeSlider');
        
        if (!startSlider || !endSlider) {
            console.warn('Time range sliders not found');
            return;
        }

        // 获取滑动条的值（0-100的百分比）
        const startPercent = parseInt(startSlider.value);
        const endPercent = parseInt(endSlider.value);
        
        // 计算实际的时间范围
        const timeRange = this.getTimeRange();
        if (!timeRange) {
            console.warn('Unable to determine time range');
            return;
        }
        
        const totalDuration = timeRange.max - timeRange.min;
        const startTime = new Date(timeRange.min + (totalDuration * startPercent / 100));
        const endTime = new Date(timeRange.min + (totalDuration * endPercent / 100));
        
        // 调用后端API获取过滤后的数据
        const startTimeStr = startTime.toISOString();
        const endTimeStr = endTime.toISOString();
        
        fetch(`/api/base-graph?start_time=${encodeURIComponent(startTimeStr)}&end_time=${encodeURIComponent(endTimeStr)}`)
            .then(response => response.json())
            .then(data => {
                if (data.success && data.graph_data) {
                    this.renderMainGraph(data.graph_data, true); // 传递isUpdate=true
                } else {
                    console.error('Failed to fetch filtered graph data:', data.error);
                }
            })
            .catch(error => {
                console.error('Error fetching filtered graph data:', error);
            });
    }

    setupGraphControls() {
        const searchGraphBtn = document.getElementById('searchGraphBtn');
        const resetGraphBtn = document.getElementById('resetGraphBtn');
        const fullscreenGraphBtn = document.getElementById('fullscreenGraphBtn');
        const exportDetailsBtn = document.getElementById('exportDetailsBtn');
        const refreshDetailsBtn = document.getElementById('refreshDetailsBtn');

        // 搜索功能
        if (searchGraphBtn) {
            searchGraphBtn.addEventListener('click', () => {
                this.toggleSearchInput();
            });
        }

        // 初始化搜索输入框功能
        this.initSearchInput();

        if (resetGraphBtn) {
            resetGraphBtn.addEventListener('click', () => {
                this.loadDefaultGraph();
            });
        }

        if (fullscreenGraphBtn) {
            fullscreenGraphBtn.addEventListener('click', () => {
                const graphContainer = document.getElementById('mainGraphContainer');
                if (graphContainer.requestFullscreen) {
                    graphContainer.requestFullscreen();
                }
            });
        }

        if (exportDetailsBtn) {
            exportDetailsBtn.addEventListener('click', () => {
                this.exportNodeDetails();
            });
        }

        if (refreshDetailsBtn) {
            refreshDetailsBtn.addEventListener('click', () => {
                this.refreshDetailsPanel();
            });
        }

        // 多层视图控制按钮
        const flatViewBtn = document.getElementById('flatViewBtn');
        const layeredViewBtn = document.getElementById('layeredViewBtn');
        const layerModeSelect = document.getElementById('layerModeSelect');
        const layerOptions = document.getElementById('layerOptions');

        if (flatViewBtn) {
            flatViewBtn.addEventListener('click', () => {
                this.switchViewMode('flat');
            });
        }

        if (layeredViewBtn) {
            layeredViewBtn.addEventListener('click', () => {
                this.switchViewMode('layered');
            });
        }

        if (layerModeSelect) {
            layerModeSelect.addEventListener('change', (e) => {
                this.currentLayerMode = e.target.value;
                if (this.currentViewMode === 'layered') {
                    this.renderLayeredGraph();
                }
            });
        }
    }

    async loadDefaultGraph() {
        const graphContainer = document.getElementById('mainGraphContainer');
        if (!graphContainer) return;

        // 显示加载状态
        graphContainer.innerHTML = `
            <div class="graph-loading">
                <div class="loading-spinner"></div>
                <div class="loading-text">Loading knowledge graph...</div>
            </div>
        `;

        try {
            // 使用独立的基础图谱API端点
            const response = await fetch('/api/base-graph', {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const data = await response.json();
            
            if (data.success && data.graph_data) {
                // 直接使用基础图谱数据
                this.renderMainGraph(data.graph_data);
            } else {
                // 如果基础图谱加载失败，显示错误信息
                graphContainer.innerHTML = `
                    <div class="graph-error">
                        <div class="error-icon">⚠️</div>
                        <div class="error-text">Failed to load base knowledge graph</div>
                        <div class="error-detail">${data.error || '未知错误'}</div>
                    </div>
                `;
            }
        } catch (error) {
            console.error('Error loading base graph:', error);
            graphContainer.innerHTML = `
                <div class="graph-error">
                    <div class="error-icon">⚠️</div>
                    <div class="error-text">Network error loading knowledge graph</div>
                    <div class="error-detail">${error.message}</div>
                </div>
            `;
        }
    }

    renderMainGraph(graphData, isUpdate = false) {
        const container = document.getElementById('mainGraphContainer');
        if (!container) return;

        // 如果不是更新操作，则保存原始图谱数据
        if (!isUpdate) {
            this.originalGraphData = graphData;
        }

        // 清空容器
        container.innerHTML = '';

        // 转换节点数据格式
        const nodes = new vis.DataSet(graphData.nodes.map(node => {
            // 根据节点类型提供默认颜色
            let defaultColor = '#8594c8 '; // 默认蓝色
            
            
            // 使用节点的颜色属性，如果没有则使用默认颜色
            const nodeColor = defaultColor;
            
            return {
                id: node.id,
                label: node.id,
                description: node.description,
                color: {
                    background: nodeColor,
                    border: this.darkenColor(nodeColor, 0.2),
                    highlight: {
                        background: this.lightenColor(nodeColor, 0.2),
                        border: nodeColor
                    }
                },
                font: {
                    color: '#333',
                    size: 14,
                    face: 'Arial'
                },
                shape: 'dot',
                size: 15
            };
        }));

        // 转换边数据格式
        const edges = new vis.DataSet(graphData.edges.map((edge, index) => ({
            id: edge.id || `${edge.from}-${edge.to}-${index}`,
            from: edge.from,
            to: edge.to,
            label: '',
            description: edge.description,
            timestamp: edge.timestamp,
            color: {
                color: '#848484',
                highlight: '#333'
            },
            font: {
                color: '#666',
                size: 12,
                strokeWidth: 2,
                strokeColor: '#fff'
            },
            arrows: {
                to: {
                    enabled: false
                }
            },
            smooth: {
                type: 'continuous',
                roundness: 0.2
            }
        })));

        // 创建网络图
        const data = { nodes: nodes, edges: edges };
        const options = {
            layout: {
                improvedLayout: true,
                hierarchical: false
            },
            physics: {
                enabled: true,
                stabilization: { iterations: 200 },
                barnesHut: {
                    gravitationalConstant: -3000,
                    centralGravity: 0.03,
                    springLength: 300,
                    springConstant: 0.015,
                    damping: 0.2,
                    avoidOverlap: 0.8
                }
            },
            interaction: {
                hover: true,
                tooltipDelay: 200,
                hideEdgesOnDrag: false,
                hideNodesOnDrag: false,
                selectConnectedEdges: false
            },
            nodes: {
                borderWidth: 2,
                shadow: {
                    enabled: true,
                    color: 'rgba(0,0,0,0.2)',
                    size: 5,
                    x: 2,
                    y: 2
                }
            },
            edges: {
                width: 2,
                shadow: {
                    enabled: true,
                    color: 'rgba(0,0,0,0.1)',
                    size: 3,
                    x: 1,
                    y: 1
                }
            }
        };

        this.mainNetwork = new vis.Network(container, data, options);
        this.mainNodes = nodes;
        this.mainEdges = edges;

        // 添加交互事件
        this.mainNetwork.on('click', (params) => {
            if (params.nodes.length > 0) {
                const nodeId = params.nodes[0];
                const node = nodes.get(nodeId);
                this.showNodeDetails(node);
            } else if (params.edges.length > 0) {
                const edgeId = params.edges[0];
                const edge = edges.get(edgeId);
                this.showEdgeDetails(edge);
            } else {
                this.clearDetailsPanel();
            }
        });

        // 网络稳定后调整视图
        this.mainNetwork.once('stabilizationIterationsDone', () => {
            this.mainNetwork.fit({
                animation: {
                    duration: 1000,
                    easingFunction: 'easeInOutQuad'
                }
            });
        });
    }

    renderDefaultGraph() {
        const defaultGraphData = {
            nodes: [
                {id: "center", label: "Memory Repository", type: "system", color: "#b6b0d5"},
                {id: "personal", label: "Personal Info", type: "category", color: "#8195cc"},
                {id: "activities", label: "Activities", type: "category", color: "#8195cc"},
                {id: "relationships", label: "Relationships", type: "category", color: "#8195cc"},
                {id: "knowledge", label: "Knowledge", type: "category", color: "#8195cc"}
            ],
            edges: [
                {from: "center", to: "personal", label: "contains", type: "contains"},
                {from: "center", to: "activities", label: "contains", type: "contains"},
                {from: "center", to: "relationships", label: "contains", type: "contains"},
                {from: "center", to: "knowledge", label: "contains", type: "contains"}
            ]
        };
        
        this.renderMainGraph(defaultGraphData);
    }

    showNodeDetails(node) {
        const detailsContent = document.getElementById('detailsContent');
        if (!detailsContent) return;

        // vis.js节点对象使用id属性作为节点标识符
        const nodeId = node.id;
        
        // 生成节点描述信息
        const description = node.description || this.generateNodeDescription(node);
        const connectionData = this.getNodeConnections(nodeId);

        // 生成邻居节点列表HTML
        let neighborsHtml = '';
        if (connectionData.neighbors && connectionData.neighbors.length > 0) {
            neighborsHtml = `
                <div class="neighbors-list">
                    ${connectionData.neighbors.map(neighbor => {
                        const escapedId = neighbor.id.replace(/'/g, "\\'");
                        return `
                        <div class="neighbor-item" onclick="window.chatApp.showNeighborDetails('${escapedId}')">
                            <span class="neighbor-name">${neighbor.label}</span>
                            ${neighbor.description ? `<span class="neighbor-description">${neighbor.description}</span>` : ''}
                        </div>
                        `;
                    }).join('')}
                </div>
            `;
        } else {
            neighborsHtml = '<span class="no-neighbors">No connections</span>';
        }

        detailsContent.innerHTML = `
            <div class="node-details">
                <div class="node-details-header">
                    <h3 class="node-label"><span class="node-prefix">🔵 Node:</span> ${nodeId}</h3>
                    <button class="delete-node-btn" onclick="window.chatApp.deleteNodeWithConfirmation('${nodeId}')">
                        🗑️ Delete Node
                    </button>
                </div>
                <div class="node-properties">
                    <div class="property-item">
                        <span class="property-label">ID:</span>
                        <span class="property-value">${nodeId}</span>
                    </div>
                    <div class="property-item">
                        <span class="property-label">Description:</span>
                        <span class="property-value">${description}</span>
                    </div>
                    <div class="property-item connections-section">
                        <span class="property-label">Connections:</span>
                        <span class="property-value">${connectionData.count} neighbors</span>
                    </div>
                    ${connectionData.neighbors && connectionData.neighbors.length > 0 ? `
                    <div class="neighbors-section">
                        ${neighborsHtml}
                    </div>
                    ` : ''}
                </div>
            </div>
        `;
    }

    showEdgeDetails(edge) {
        const detailsContent = document.getElementById('detailsContent');
        if (!detailsContent) return;

        const fromNode = this.mainNodes.get(edge.from);
        const toNode = this.mainNodes.get(edge.to);
        
        // 查找两个节点间的所有edge
        const allEdgesBetweenNodes = [];
        if (this.mainEdges) {
            this.mainEdges.forEach(e => {
                if ((e.from === edge.from && e.to === edge.to) || 
                    (e.from === edge.to && e.to === edge.from)) {
                    allEdgesBetweenNodes.push(e);
                }
            });
        }
        
        // 格式化时间戳显示
        const formatTimestamp = (timestamp) => {
            if (!timestamp) return 'Null';
            try {
                const date = new Date(timestamp);
                return date.toLocaleString('zh-CN', {
                    year: 'numeric',
                    month: '2-digit',
                    day: '2-digit',
                    hour: '2-digit',
                    minute: '2-digit',
                    second: '2-digit'
                });
            } catch (e) {
                return timestamp;
            }
        };

        // 生成edge详细信息HTML - 新的简化格式，为每条边添加删除按钮
        const edgeDetailsHtml = allEdgesBetweenNodes.map((e, index) => {
            const description = e.description || this.generateEdgeDescription(e);
            // 为每条边生成唯一标识符，用于删除操作
            const edgeId = e.id || `${e.from}_${e.to}_${index}_${e.timestamp}`;
            return `
                <div class="edge-event-item">
                    <div class="edge-event-content">
                        <div class="event-description">${description}</div>
                        <div class="event-time">🕙 ${formatTimestamp(e.timestamp)}</div>
                    </div>
                    <button class="delete-edge-btn" onclick="window.chatApp.deleteEdgeWithConfirmation('${edgeId}', '${e.from}', '${e.to}', ${index})">
                        🗑️ Delete Edge
                    </button>
                </div>
            `;
        }).join('');

        detailsContent.innerHTML = `
            <div class="edge-details">
                <div class="edge-connection">
                    <strong>${fromNode ? fromNode.label : edge.from}</strong>
                    <span class="connection-arrow">↔</span>
                    <strong>${toNode ? toNode.label : edge.to}</strong>
                </div>
                <div class="edge-summary">
                    <div class="property-item">
                        <span class="property-label">From Node:</span>
                        <span class="property-value">${edge.from}</span>
                    </div>
                    <div class="property-item">
                        <span class="property-label">To Node:</span>
                        <span class="property-value">${edge.to}</span>
                    </div>
                    <div class="property-item">
                        <span class="property-label">Total Connections:</span>
                        <span class="property-value">${allEdgesBetweenNodes.length}</span>
                    </div>
                </div>
                <div class="edge-events-container">
                    ${edgeDetailsHtml}
                </div>
            </div>
        `;
    }

    clearDetailsPanel() {
        const detailsContent = document.getElementById('detailsContent');
        if (!detailsContent) return;

        detailsContent.innerHTML = `
            <div class="no-selection">
                <div class="no-selection-icon">👆</div>
                <div class="no-selection-text">
                    <h4>Select a node or edge</h4>
                    <p>Click on any node or edge in the knowledge graph to view its detailed information here.</p>
                </div>
            </div>
        `;
    }

    getNodeConnections(nodeId) {
        if (!this.mainEdges) return { count: 0, neighbors: [] };
        
        const neighbors = new Set();
        this.mainEdges.forEach(edge => {
            if (edge.from === nodeId) {
                neighbors.add(edge.to);
            } else if (edge.to === nodeId) {
                neighbors.add(edge.from);
            }
        });
        
        // 获取邻居节点的详细信息
        const neighborDetails = Array.from(neighbors).map(neighborId => {
            const neighborNode = this.mainNodes ? this.mainNodes.get(neighborId) : null;
            return {
                id: neighborId,
                label: neighborNode ? neighborNode.label : neighborId,
                description: neighborNode ? neighborNode.description : null
            };
        });
        
        return {
            count: neighbors.size,
            neighbors: neighborDetails
        };
    }

    showNeighborDetails(neighborId) {
        // 查找邻居节点
        const neighborNode = this.mainNodes ? this.mainNodes.get(neighborId) : null;
        
        if (neighborNode) {
            // 如果找到节点，显示其详细信息
            this.showNodeDetails(neighborNode);
            
            // 可选：在图中高亮显示该节点
            if (this.mainNetwork) {
                this.mainNetwork.selectNodes([neighborId]);
                this.mainNetwork.focus(neighborId, {
                    scale: 1.2,
                    animation: {
                        duration: 500,
                        easingFunction: 'easeInOutQuad'
                    }
                });
            }
        } else {
            console.warn(`Neighbor node with ID ${neighborId} not found`);
        }
    }

    generateNodeDescription(node) {
        // 生成节点的描述信息
        if (node.description) {
            return node.description;
        }
        
        // 根据节点类型和标签生成默认描述
        const type = node.type || '未知类型';
        const label = node.label || node.id;
        
        return `This is a ${type.toLowerCase()} node named "${label}". It represents an important concept or entity in the knowledge graph.`;
    }

    generateEdgeDescription(edge) {
        // 生成边的描述信息
        if (edge.description) {
            return edge.description;
        }
        
        // 根据边的类型和标签生成默认描述
        const type = edge.type || 'connection';
        const label = edge.label || 'relationship';
        
        return `This ${type.toLowerCase()} represents a "${label}" relationship between the connected nodes.`;
    }

    exportNodeDetails() {
        const detailsContent = document.getElementById('detailsContent');
        if (!detailsContent) return;

        const nodeDetails = detailsContent.querySelector('.node-details, .edge-details');
        if (!nodeDetails) {
            alert('No node or edge selected to export.');
            return;
        }

        const content = nodeDetails.innerText;
        const blob = new Blob([content], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'node-details.txt';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    refreshDetailsPanel() {
        this.clearDetailsPanel();
        if (this.mainNetwork) {
            const selectedNodes = this.mainNetwork.getSelectedNodes();
            const selectedEdges = this.mainNetwork.getSelectedEdges();
            
            if (selectedNodes.length > 0) {
                const node = this.mainNodes.get(selectedNodes[0]);
                this.showNodeDetails(node);
            } else if (selectedEdges.length > 0) {
                const edge = this.mainEdges.get(selectedEdges[0]);
                this.showEdgeDetails(edge);
            }
        }
    }

    deleteNodeWithConfirmation(nodeId) {
        // 获取节点信息用于确认对话框
        const node = this.mainNodes.get(nodeId);
        const connections = this.getNodeConnections(nodeId);
        
        // 显示确认对话框
        const confirmMessage = `Are you sure you want to delete node "${nodeId}"?\n\nThis will also delete:\n- ${connections} connected edges\n- All relationships with this node\n\nThis action cannot be undone.`;
        
        if (confirm(confirmMessage)) {
            this.deleteNode(nodeId);
        }
    }

    deleteNode(nodeId) {
        if (!this.mainNodes || !this.mainEdges) return;
        
        try {
            // 1. 删除与该节点相关的所有边
            const edgesToDelete = [];
            this.mainEdges.forEach(edge => {
                if (edge.from === nodeId || edge.to === nodeId) {
                    edgesToDelete.push(edge.id);
                }
            });
            
            // 删除边
            if (edgesToDelete.length > 0) {
                this.mainEdges.remove(edgesToDelete);
            }
            
            // 2. 删除节点
            this.mainNodes.remove(nodeId);
            
            // 3. 清空详情面板
            this.clearDetailsPanel();
            
            // 4. 显示删除成功消息
            console.log(`Node "${nodeId}" and ${edgesToDelete.length} related edges have been deleted.`);
            
        } catch (error) {
            console.error('Error deleting node:', error);
            alert('Failed to delete node. Please try again.');
        }
    }

    deleteEdgeWithConfirmation(edgeId, fromNode, toNode, edgeIndex) {
        // 显示确认对话框
        const confirmMessage = `Are you sure you want to delete this edge?\n\nFrom: ${fromNode}\nTo: ${toNode}\n\nThis action cannot be undone.`;
        
        if (confirm(confirmMessage)) {
            this.deleteEdge(fromNode, toNode, edgeIndex);
        }
    }

    deleteEdge(fromNode, toNode, edgeIndex) {
        if (!this.mainEdges) return;
        
        try {
            // 查找两个节点间的所有边
            const allEdgesBetweenNodes = [];
            this.mainEdges.forEach(edge => {
                if ((edge.from === fromNode && edge.to === toNode) || 
                    (edge.from === toNode && edge.to === fromNode)) {
                    allEdgesBetweenNodes.push(edge);
                }
            });
            
            // 确保索引有效
            if (edgeIndex >= 0 && edgeIndex < allEdgesBetweenNodes.length) {
                const edgeToDelete = allEdgesBetweenNodes[edgeIndex];
                
                // 删除指定的边
                this.mainEdges.remove(edgeToDelete.id);
                
                // 检查是否还有其他边连接这两个节点
                const remainingEdges = [];
                this.mainEdges.forEach(edge => {
                    if ((edge.from === fromNode && edge.to === toNode) || 
                        (edge.from === toNode && edge.to === fromNode)) {
                        remainingEdges.push(edge);
                    }
                });
                
                // 刷新详情面板
                if (remainingEdges.length > 0) {
                    // 如果还有其他边，显示更新后的边详情
                    this.showEdgeDetails(remainingEdges[0]);
                } else {
                    // 如果没有其他边了，清空详情面板
                    this.clearDetailsPanel();
                }
                
                console.log(`Edge between "${fromNode}" and "${toNode}" has been deleted.`);
            } else {
                console.error('Invalid edge index:', edgeIndex);
                alert('Failed to delete edge: Invalid edge reference.');
            }
            
        } catch (error) {
            console.error('Error deleting edge:', error);
            alert('Failed to delete edge. Please try again.');
        }
    }

    // Notification 相关功能
    async initNotificationFeatures() {
        // 获取通知相关元素
        this.markAllReadBtn = document.getElementById('markAllReadBtn');
        this.clearAllBtn = document.getElementById('clearAllBtn');
        this.notificationList = document.getElementById('notificationList');
        this.notificationEmpty = document.getElementById('notificationEmpty');
        this.notificationDetail = document.getElementById('notificationDetail');
        this.filterBtns = document.querySelectorAll('.filter-btn');

        // 初始化通知数据
        this.notifications = [];
        this.currentFilter = 'all';

        // 绑定事件监听器
        if (this.markAllReadBtn) {
            this.markAllReadBtn.addEventListener('click', () => this.markAllNotificationsRead());
        }

        if (this.clearAllBtn) {
            this.clearAllBtn.addEventListener('click', () => this.clearAllNotifications());
        }

        // 绑定过滤器按钮事件
        this.filterBtns.forEach(btn => {
            btn.addEventListener('click', async (e) => {
                const filter = e.target.dataset.filter;
                this.currentFilter = filter;
                await this.loadNotifications(filter);
                
                // 更新按钮状态
                this.filterBtns.forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
            });
        });

        // 加载初始通知数据
        await this.loadNotifications('all');
    }

    async loadNotifications(filter = 'all') {
        try {
            const response = await fetch(`/api/notifications?filter=${filter}`);
            const data = await response.json();
            
            if (data.success) {
                this.notifications = data.notifications;
                this.renderNotifications(data.notifications);
                this.updateNotificationCount(data.stats);
            } else {
                console.error('Failed to load notifications:', data.error);
            }
        } catch (error) {
            console.error('Error loading notifications:', error);
        }
    }

    renderNotifications(notifications) {
        if (!this.notificationList) return;

        if (notifications.length === 0) {
            this.showEmptyState();
            return;
        }

        this.hideEmptyState();
        
        // 清空现有通知
        this.notificationList.innerHTML = '';
        
        // 渲染通知卡片
        notifications.forEach(notification => {
            const card = this.createNotificationCard(notification);
            this.notificationList.appendChild(card);
        });

        // 重新绑定事件
        this.bindNotificationActions();
    }

    createNotificationCard(notification) {
        const card = document.createElement('div');
        card.className = `notification-card ${!notification.isRead ? 'unread' : ''}`;
        card.dataset.id = notification.id;
        card.dataset.type = notification.type;

        const avatarClass = notification.type;
        
        card.innerHTML = `
            <div class="notification-avatar ${avatarClass}">${notification.avatar}</div>
            <div class="notification-body">
                <div class="notification-header-content">
                    <span class="notification-title">${notification.title}</span>
                    <span class="notification-time">${notification.time}</span>
                    ${!notification.isRead ? '<span class="unread-dot"></span>' : ''}
                </div>
                <div class="notification-message">${notification.message}</div>
                <div class="notification-actions">
                    ${notification.actions.map(action => 
                        `<button class="action-btn ${action.action}" data-action="${action.action}">${action.text}</button>`
                    ).join('')}
                </div>
            </div>
        `;

        return card;
    }

    bindNotificationActions() {
        const notificationCards = document.querySelectorAll('.notification-card');
        
        notificationCards.forEach(card => {
            // 点击通知卡片显示详情
            card.addEventListener('click', async () => {
                await this.showNotificationDetail(card);
                
                // 标记为已读
                if (card.classList.contains('unread')) {
                    await this.markNotificationRead(card);
                }
                
                // 更新选中状态
                notificationCards.forEach(c => c.classList.remove('selected'));
                card.classList.add('selected');
            });
        });
    }

    async showNotificationDetail(card) {
        const notificationId = card.dataset.id;
        const type = card.dataset.type;
        
        // 从本地数据获取通知详情
        const notificationData = this.notifications.find(n => n.id == notificationId);
        
        if (notificationData) {
            // 渲染详情面板
            this.renderNotificationDetail(notificationData);
        }
    }

    async getNotificationData(id, type) {
        console.log('getNotificationData called with ID:', id, 'type:', type);
        
        // 首先尝试从已加载的通知数据中获取
        if (this.notificationData && this.notificationData.length > 0) {
            console.log('Searching in loaded notifications:', this.notificationData.length);
            const notification = this.notificationData.find(n => n.id == id);
            if (notification) {
                console.log('Found notification in loaded data:', notification);
                return notification;
            }
        }

        // 如果没有找到，尝试从服务器获取
        try {
            console.log('Fetching from server API...');
            const response = await fetch('/api/notifications');
            if (response.ok) {
                const data = await response.json();
                console.log('API response:', data);
                if (data.success && data.notifications) {
                    const notification = data.notifications.find(n => n.id == id);
                    if (notification) {
                        console.log('Found notification in API response:', notification);
                        return notification;
                    }
                }
            }
        } catch (error) {
            console.error('Error fetching notification data:', error);
        }

        console.log('Notification not found in API, returning null');
        return null;
    }

    renderNotificationDetail(data) {
        if (!data || !this.notificationDetail) return;
        
        const avatarClass = `${data.type}`;
        
        // Format detailed info with better structure
        const formattedDetailedInfo = data.detailed_info ? 
            data.detailed_info.replace(/(\d+\))/g, '<br><strong>$1</strong>').replace(/\. /g, '.<br><br>') : 
            'No additional details available.';
        
        // Check if this is a report notification with knowledge graph data
        const hasKnowledgeGraph = data.knowledge_graph && data.knowledge_graph.nodes && data.knowledge_graph.edges;
        const knowledgeGraphButton = hasKnowledgeGraph ? 
            `<button class="knowledge-graph-btn" onclick="window.chatApp.showKnowledgeGraphModal('${data.id}')">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="3"/>
                    <circle cx="12" cy="5" r="1"/>
                    <circle cx="12" cy="19" r="1"/>
                    <circle cx="5" cy="12" r="1"/>
                    <circle cx="19" cy="12" r="1"/>
                    <line x1="12" y1="9" x2="12" y2="15"/>
                    <line x1="9" y1="12" x2="15" y2="12"/>
                </svg>
                View Knowledge Graph
            </button>` : '';
        
        // Check if this is a suggestion notification and add Apply Suggestion button
        const isSuggestion = data.type === 'suggestion';
        const applySuggestionButton = isSuggestion ? 
            `<button class="apply-suggestion-btn" onclick="window.chatApp.applySuggestion('${data.id}')">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="20,6 9,17 4,12"/>
                </svg>
                Apply Suggestion
            </button>` : '';
        
        this.notificationDetail.innerHTML = `
            <div class="detail-content">
                <div class="detail-header">
                    <div class="detail-avatar ${avatarClass}">${data.avatar}</div>
                    <div class="detail-info">
                        <h3>${data.title}</h3>
                        <div class="detail-meta">${data.sender} • ${data.time}</div>
                    </div>
                </div>
                <div class="detail-message">${data.message}</div>
                <div class="detail-info-section">
                    <h4>Detailed Information</h4>
                    <div class="detail-description">${formattedDetailedInfo}</div>
                    <div class="detail-actions">
                        ${knowledgeGraphButton}
                        ${applySuggestionButton}
                    </div>
                </div>
            </div>
        `;
    }

    handleNotificationDetailAction(action, notificationId) {
        switch (action) {
            case 'view_details':
                console.log('Viewing details for notification:', notificationId);
                break;
            case 'view_chat':
                this.switchToView('chat');
                break;
            case 'view_memory':
                this.switchToView('memory');
                break;
            case 'view_schedule':
                this.switchToView('schedule');
                break;
            case 'view_report':
                console.log('Viewing report for notification:', notificationId);
                break;
            case 'view_knowledge_graph':
                this.showKnowledgeGraphModal(notificationId);
                break;
            case 'download_pdf':
                console.log('Downloading PDF for notification:', notificationId);
                break;
            case 'mark_read':
                const card = document.querySelector(`[data-id="${notificationId}"]`);
                if (card) this.markNotificationRead(card);
                break;
            case 'dismiss':
            case 'snooze':
                console.log(`${action} notification:`, notificationId);
                break;
            default:
                console.log('未知操作:', action);
        }
    }

    handleNotificationAction(action, notificationId, type, item) {
        console.log(`处理通知操作: ${action}, ID: ${notificationId}, 类型: ${type}`);
        
        switch (action) {
            case 'View Details':
            case 'view_details':
                console.log('Viewing details for notification:', notificationId);
                // 可以在这里添加查看详情的逻辑
                break;
                
            case 'Dismiss':
            case 'dismiss':
                console.log('Dismissing notification:', notificationId);
                // 可以在这里添加忽略通知的逻辑
                break;
                
            default:
                console.log('未知操作:', action);
        }
    }

    async markNotificationRead(item) {
        const notificationId = item.dataset.id;
        
        try {
            const response = await fetch(`/api/notifications/${notificationId}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ action: 'mark_read' })
            });
            
            const data = await response.json();
            
            if (data.success) {
                item.classList.remove('unread');
                item.classList.add('read');
                // 重新加载通知以更新计数
                await this.loadNotifications(this.currentFilter);
            }
        } catch (error) {
            console.error('Error marking notification as read:', error);
        }
    }

    async markAllNotificationsRead() {
        try {
            const response = await fetch('/api/notifications/batch', {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ action: 'mark_all_read' })
            });
            
            const data = await response.json();
            
            if (data.success) {
                // 重新加载通知
                await this.loadNotifications(this.currentFilter);
                alert(`已将 ${data.affected_count} 条通知标记为已读`);
            }
        } catch (error) {
            console.error('Error marking all notifications as read:', error);
        }
    }

    async clearAllNotifications() {
        const confirmClear = confirm('Are you sure you want to clear all notifications? This action cannot be undone.');
        if (confirmClear) {
            try {
                const response = await fetch('/api/notifications/batch', {
                    method: 'PUT',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ action: 'clear_all' })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    // 重新加载通知
                    await this.loadNotifications(this.currentFilter);
                }
            } catch (error) {
                console.error('Error clearing all notifications:', error);
            }
        }
    }

    filterNotifications(filter) {
        const notificationCards = document.querySelectorAll('.notification-card');
        let visibleCount = 0;
        
        notificationCards.forEach(card => {
            const type = card.dataset.type;
            const isUnread = card.classList.contains('unread');
            let shouldShow = false;
            
            switch (filter) {
                case 'all':
                    shouldShow = true;
                    break;
                case 'unread':
                    shouldShow = isUnread;
                    break;
                case 'report':
                case 'reminder':
                case 'suggestion':
                    shouldShow = type === filter;
                    break;
            }
            
            if (shouldShow) {
                card.style.display = 'flex';
                visibleCount++;
            } else {
                card.style.display = 'none';
            }
        });
        
        // 如果没有可见的通知，显示空状态
        if (visibleCount === 0) {
            this.showEmptyState();
        } else {
            this.hideEmptyState();
        }
    }

    showEmptyState() {
        if (this.notificationEmpty) {
            this.notificationEmpty.style.display = 'block';
        }
        if (this.notificationList) {
            this.notificationList.style.display = 'none';
        }
    }

    hideEmptyState() {
        if (this.notificationEmpty) {
            this.notificationEmpty.style.display = 'none';
        }
        if (this.notificationList) {
            this.notificationList.style.display = 'block';
        }
    }

    // 新增：初始化日程模块
    initScheduleModule() {
        // 日程数据存储
        this.scheduleData = {};
        this.currentDate = new Date();
        this.selectedDate = null;
        this.editingEvent = null;
        
        // 从服务器加载日程数据
        this.loadScheduleData();
        
        // 获取日程相关元素
        this.scheduleView = document.getElementById('scheduleView');
        this.calendarMain = document.getElementById('calendarMain');
        this.dateDetail = document.getElementById('dateDetail');
        this.eventModal = document.getElementById('eventModal');
        
        // 日历导航元素
        this.prevMonthBtn = document.getElementById('prevMonth');
        this.nextMonthBtn = document.getElementById('nextMonth');
        this.todayBtn = document.getElementById('todayBtn');
        this.currentMonthDisplay = document.getElementById('currentMonth');
        this.calendarBody = document.getElementById('calendarBody');
        
        // 日期详情元素
        this.backToCalendarBtn = document.getElementById('backToCalendar');
        this.selectedDateDisplay = document.getElementById('selectedDate');
        this.addEventBtn = document.getElementById('addEventBtn');
        this.eventsContainer = document.getElementById('eventsContainer');
        this.noEvents = document.getElementById('noEvents');
        this.eventsList = document.getElementById('eventsList');
        this.createEventBtn = document.getElementById('createEventBtn');
        
        // 模态框元素
        this.modalTitle = document.getElementById('modalTitle');
        this.closeModalBtn = document.getElementById('closeModal');
        this.eventForm = document.getElementById('eventForm');
        this.cancelEventBtn = document.getElementById('cancelEvent');
        this.saveEventBtn = document.getElementById('saveEvent');
        
        // 绑定事件监听器
        this.bindScheduleEvents();
        
        // 初始化日历
        this.renderCalendar();
    }
    
    // 绑定日程相关事件
    bindScheduleEvents() {
        // 日历导航事件
        this.prevMonthBtn.addEventListener('click', () => this.navigateMonth(-1));
        this.nextMonthBtn.addEventListener('click', () => this.navigateMonth(1));
        this.todayBtn.addEventListener('click', () => this.goToToday());
        
        // 日期详情事件
        this.backToCalendarBtn.addEventListener('click', () => this.showCalendarView());
        this.addEventBtn.addEventListener('click', () => this.showEventModal());
        this.createEventBtn.addEventListener('click', () => this.showEventModal());
        
        // 模态框事件
        this.closeModalBtn.addEventListener('click', () => this.hideEventModal());
        this.cancelEventBtn.addEventListener('click', () => this.hideEventModal());
        this.eventForm.addEventListener('submit', (e) => this.handleEventSubmit(e));
        
        // 点击模态框外部关闭
        this.eventModal.addEventListener('click', (e) => {
            if (e.target === this.eventModal) {
                this.hideEventModal();
            }
        });
    }
    
    // 渲染日历
    renderCalendar() {
        const year = this.currentDate.getFullYear();
        const month = this.currentDate.getMonth();
        
        // 更新月份显示
        const monthNames = [
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ];
        this.currentMonthDisplay.textContent = `${monthNames[month]} ${year}`;
        
        // 获取月份第一天和最后一天
        const firstDay = new Date(year, month, 1);
        const lastDay = new Date(year, month + 1, 0);
        const startDate = new Date(firstDay);
        startDate.setDate(startDate.getDate() - firstDay.getDay());
        
        // 清空日历体
        this.calendarBody.innerHTML = '';
        
        // 生成6周的日期
        const today = new Date(2025, 9, 20); // 固定为2025年10月20日
        for (let week = 0; week < 6; week++) {
            for (let day = 0; day < 7; day++) {
                const currentDate = new Date(startDate);
                currentDate.setDate(startDate.getDate() + (week * 7) + day);
                
                const dayElement = this.createDayElement(currentDate, month, today);
                this.calendarBody.appendChild(dayElement);
            }
        }
    }
    
    // 创建日期元素
    createDayElement(date, currentMonth, today) {
        const dayDiv = document.createElement('div');
        dayDiv.className = 'calendar-day';
        
        // 添加样式类
        if (date.getMonth() !== currentMonth) {
            dayDiv.classList.add('other-month');
        }
        
        if (this.isSameDay(date, today)) {
            dayDiv.classList.add('today');
        }
        
        // 检查是否有事件
        const dateKey = this.formatDateKey(date);
        if (this.scheduleData[dateKey] && this.scheduleData[dateKey].length > 0) {
            dayDiv.classList.add('has-events');
        }
        
        // 日期数字
        const dayNumber = document.createElement('div');
        dayNumber.className = 'day-number';
        dayNumber.textContent = date.getDate();
        dayDiv.appendChild(dayNumber);
        
        // 事件预览
        if (this.scheduleData[dateKey]) {
            const eventsDiv = document.createElement('div');
            eventsDiv.className = 'day-events';
            
            const eventsToShow = this.scheduleData[dateKey].slice(0, 3);
            eventsToShow.forEach(event => {
                const eventPreview = document.createElement('div');
                eventPreview.className = 'event-preview';
                eventPreview.textContent = event.title;
                eventsDiv.appendChild(eventPreview);
            });
            
            if (this.scheduleData[dateKey].length > 3) {
                const moreEvents = document.createElement('div');
                moreEvents.className = 'event-preview';
                moreEvents.textContent = `+${this.scheduleData[dateKey].length - 3} more`;
                eventsDiv.appendChild(moreEvents);
            }
            
            dayDiv.appendChild(eventsDiv);
        }
        
        // 点击事件
        dayDiv.addEventListener('click', () => this.selectDate(date));
        
        return dayDiv;
    }
    
    // 选择日期
    selectDate(date) {
        this.selectedDate = new Date(date);
        this.showDateDetail();
    }
    
    // 显示日期详情
    showDateDetail() {
        this.calendarMain.style.display = 'none';
        this.dateDetail.style.display = 'block';
        
        // 更新选中日期显示
        const options = { 
            weekday: 'long', 
            year: 'numeric', 
            month: 'long', 
            day: 'numeric' 
        };
        this.selectedDateDisplay.textContent = this.selectedDate.toLocaleDateString('en-US', options);
        
        // 渲染事件列表
        this.renderEventsList();
    }
    
    // 显示日历视图
    showCalendarView() {
        this.dateDetail.style.display = 'none';
        this.calendarMain.style.display = 'block';
    }
    
    // 渲染事件列表
    renderEventsList() {
        const dateKey = this.formatDateKey(this.selectedDate);
        const events = this.scheduleData[dateKey] || [];
        
        if (events.length === 0) {
            this.noEvents.style.display = 'block';
            this.eventsList.style.display = 'none';
        } else {
            this.noEvents.style.display = 'none';
            this.eventsList.style.display = 'block';
            
            this.eventsList.innerHTML = '';
            events.forEach((event, index) => {
                const eventElement = this.createEventElement(event, index);
                this.eventsList.appendChild(eventElement);
            });
        }
    }
    
    // 创建事件元素
    createEventElement(event, index) {
        const eventDiv = document.createElement('div');
        eventDiv.className = `event-item ${event.category}`;
        
        const eventHeader = document.createElement('div');
        eventHeader.className = 'event-header';
        
        const eventTitle = document.createElement('h4');
        eventTitle.className = 'event-title';
        eventTitle.textContent = event.title;
        
        const eventActions = document.createElement('div');
        eventActions.className = 'event-actions';
        
        const editBtn = document.createElement('button');
        editBtn.className = 'event-action-btn';
        editBtn.innerHTML = '✏️';
        editBtn.title = 'Edit';
        editBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            this.editEvent(event, index);
        });
        
        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'event-action-btn';
        deleteBtn.innerHTML = '🗑️';
        deleteBtn.title = 'Delete';
        deleteBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            this.deleteEvent(index);
        });
        
        eventActions.appendChild(editBtn);
        eventActions.appendChild(deleteBtn);
        eventHeader.appendChild(eventTitle);
        eventHeader.appendChild(eventActions);
        eventDiv.appendChild(eventHeader);
        
        // 时间信息
        if (event.time || event.startTime || event.endTime) {
            const eventTime = document.createElement('div');
            eventTime.className = 'event-time';
            let timeText = '';
            
            // 优先使用time字段（用于现有数据）
            if (event.time) {
                timeText = event.time;
            } else {
                // 兼容startTime和endTime字段（用于新创建的事件）
                if (event.startTime) timeText += event.startTime;
                if (event.endTime) timeText += ` - ${event.endTime}`;
            }
            
            eventTime.textContent = timeText;
            eventDiv.appendChild(eventTime);
        }
        
        // 描述
        if (event.description) {
            const eventDesc = document.createElement('div');
            eventDesc.className = 'event-description';
            eventDesc.textContent = event.description;
            eventDiv.appendChild(eventDesc);
        }
        
        // 分类标签
        const eventCategory = document.createElement('span');
        eventCategory.className = 'event-category';
        eventCategory.textContent = event.category.charAt(0).toUpperCase() + event.category.slice(1);
        eventDiv.appendChild(eventCategory);
        
        return eventDiv;
    }
    
    // 显示事件模态框
    showEventModal(event = null, index = null) {
        this.editingEvent = event ? { ...event, index } : null;
        
        if (this.editingEvent) {
            this.modalTitle.textContent = 'Edit Event';
            this.populateEventForm(event);
        } else {
            this.modalTitle.textContent = 'Add New Event';
            this.clearEventForm();
        }
        
        this.eventModal.style.display = 'flex';
    }
    
    // 隐藏事件模态框
    hideEventModal() {
        this.eventModal.style.display = 'none';
        this.editingEvent = null;
        this.clearEventForm();
    }
    
    // 填充事件表单
    populateEventForm(event) {
        document.getElementById('eventTitle').value = event.title || '';
        document.getElementById('eventDate').value = event.date || '';
        
        // 处理时间字段的兼容性
        if (event.time) {
            // 如果有time字段，尝试解析为startTime和endTime
            const timeParts = event.time.split(' - ');
            document.getElementById('eventStartTime').value = timeParts[0] || '';
            document.getElementById('eventEndTime').value = timeParts[1] || '';
        } else {
            // 使用原有的startTime和endTime字段
            document.getElementById('eventStartTime').value = event.startTime || '';
            document.getElementById('eventEndTime').value = event.endTime || '';
        }
        
        document.getElementById('eventDescription').value = event.description || '';
        document.getElementById('eventCategory').value = event.category || 'other';
    }
    
    // 清空事件表单
    clearEventForm() {
        document.getElementById('eventTitle').value = '';
        document.getElementById('eventDate').value = this.formatDateForInput(this.selectedDate);
        document.getElementById('eventStartTime').value = '';
        document.getElementById('eventEndTime').value = '';
        document.getElementById('eventDescription').value = '';
        document.getElementById('eventCategory').value = 'other';
    }
    
    // 处理事件提交
    async handleEventSubmit(e) {
        e.preventDefault();
        
        const formData = new FormData(this.eventForm);
        const eventData = {
            title: formData.get('title'),
            date: formData.get('date'),
            startTime: formData.get('startTime'),
            endTime: formData.get('endTime'),
            description: formData.get('description'),
            category: formData.get('category'),
            id: this.editingEvent ? this.editingEvent.id : Date.now()
        };
        
        // 为了兼容现有数据格式，如果有startTime，将其映射到time字段
        if (eventData.startTime) {
            eventData.time = eventData.startTime;
            if (eventData.endTime) {
                eventData.time += ` - ${eventData.endTime}`;
            }
        }
        
        if (this.editingEvent) {
            const success = await this.updateEvent(eventData, this.editingEvent.index);
            if (success) {
                this.hideEventModal();
            }
        } else {
            const success = await this.addEvent(eventData);
            if (success) {
                this.hideEventModal();
            }
        }
    }
    
    // 添加事件
    async addEvent(eventData) {
        try {
            const response = await fetch('/api/schedule/events', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(eventData)
            });
            
            const result = await response.json();
            if (result.success) {
                // 更新本地数据
                const dateKey = this.formatDateKey(new Date(eventData.date));
                if (!this.scheduleData[dateKey]) {
                    this.scheduleData[dateKey] = [];
                }
                this.scheduleData[dateKey].push(result.event);
                
                this.renderEventsList();
                this.renderCalendar();
                return true;
            } else {
                console.error('添加事件失败:', result.error);
                return false;
            }
        } catch (error) {
            console.error('添加事件时发生错误:', error);
            // 回退到本地存储
            const dateKey = this.formatDateKey(new Date(eventData.date));
            if (!this.scheduleData[dateKey]) {
                this.scheduleData[dateKey] = [];
            }
            this.scheduleData[dateKey].push(eventData);
            this.saveScheduleData();
            this.renderEventsList();
            this.renderCalendar();
            return true;
        }
    }
    
    // 更新事件
    updateEvent(eventData, index) {
        const oldDateKey = this.formatDateKey(this.selectedDate);
        const newDateKey = this.formatDateKey(new Date(eventData.date));
        
        // 从旧日期删除
        if (this.scheduleData[oldDateKey]) {
            this.scheduleData[oldDateKey].splice(index, 1);
            if (this.scheduleData[oldDateKey].length === 0) {
                delete this.scheduleData[oldDateKey];
            }
        }
        
        // 添加到新日期
        if (!this.scheduleData[newDateKey]) {
            this.scheduleData[newDateKey] = [];
        }
        this.scheduleData[newDateKey].push(eventData);
        
        this.saveScheduleData();
        this.renderEventsList();
        this.renderCalendar();
    }
    
    // 编辑事件
    editEvent(event, index) {
        this.showEventModal(event, index);
    }
    
    // 删除事件
    deleteEvent(index) {
        if (confirm('Are you sure you want to delete this event?')) {
            const dateKey = this.formatDateKey(this.selectedDate);
            if (this.scheduleData[dateKey]) {
                this.scheduleData[dateKey].splice(index, 1);
                if (this.scheduleData[dateKey].length === 0) {
                    delete this.scheduleData[dateKey];
                }
                this.saveScheduleData();
                this.renderEventsList();
                this.renderCalendar();
            }
        }
    }
    
    // 导航月份
    navigateMonth(direction) {
        this.currentDate.setMonth(this.currentDate.getMonth() + direction);
        this.renderCalendar();
    }
    
    // 回到今天
    goToToday() {
        this.currentDate = new Date(2025, 9, 20); // 固定为2025年10月20日
        this.renderCalendar();
    }
    
    // 工具函数
    formatDateKey(date) {
        // 使用本地时区格式化日期，避免UTC转换导致的日期偏移
        const year = date.getFullYear();
        const month = String(date.getMonth() + 1).padStart(2, '0');
        const day = String(date.getDate()).padStart(2, '0');
        return `${year}-${month}-${day}`;
    }
    
    formatDateForInput(date) {
        // 使用本地时区格式化日期，避免UTC转换导致的日期偏移
        const year = date.getFullYear();
        const month = String(date.getMonth() + 1).padStart(2, '0');
        const day = String(date.getDate()).padStart(2, '0');
        return `${year}-${month}-${day}`;
    }
    
    isSameDay(date1, date2) {
        return date1.toDateString() === date2.toDateString();
    }
    
    // 从服务器加载日程数据
    async loadScheduleData() {
        try {
            const response = await fetch('/api/schedule');
            const result = await response.json();
            
            if (result.success) {
                this.scheduleData = result.data || {};
                this.renderCalendar();
            } else {
                console.error('加载日程数据失败:', result.error);
                this.scheduleData = {};
            }
        } catch (error) {
            console.error('加载日程数据时发生错误:', error);
            this.scheduleData = {};
        }
    }

    // 保存日程数据到服务器
    async saveScheduleDataToServer() {
        try {
            const response = await fetch('/api/schedule', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(this.scheduleData)
            });
            
            const result = await response.json();
            if (!result.success) {
                console.error('保存日程数据失败:', result.error);
            }
        } catch (error) {
            console.error('保存日程数据时发生错误:', error);
        }
    }

    // 保存日程数据（兼容本地存储）
    saveScheduleData() {
        // 保存到服务器
        this.saveScheduleDataToServer();
        // 同时保存到本地存储作为备份
        localStorage.setItem('scheduleData', JSON.stringify(this.scheduleData));
    }

    updateNotificationCount(stats) {
        if (stats) {
            // 使用API返回的统计数据
            const unreadCount = stats.unread_count;
            
            // 更新菜单项上的未读数量标识（如果需要的话）
            const notificationMenuItem = document.getElementById('notificationMenuItem');
            if (notificationMenuItem) {
                let badge = notificationMenuItem.querySelector('.notification-badge');

                if (unreadCount > 0) {
                    if (!badge) {
                        badge = document.createElement('span');
                        badge.className = 'notification-badge';
                        notificationMenuItem.appendChild(badge);
                    }
                    badge.textContent = unreadCount > 99 ? '99+' : unreadCount;
                } else if (badge) {
                    badge.remove();
                }
            }
        } else {
            // 兼容旧的计数方式
            const unreadCount = document.querySelectorAll('.notification-card.unread').length;
            
            const notificationMenuItem = document.getElementById('notificationMenuItem');
            if (notificationMenuItem) {
                let badge = notificationMenuItem.querySelector('.notification-badge');

                if (unreadCount > 0) {
                    if (!badge) {
                        badge = document.createElement('span');
                        badge.className = 'notification-badge';
                        notificationMenuItem.appendChild(badge);
                    }
                    badge.textContent = unreadCount > 99 ? '99+' : unreadCount;
                } else if (badge) {
                    badge.remove();
                }
            }
        }
    }

    // 添加新通知的方法（供后续扩展使用）
    addNotification(notification) {
        const notificationHTML = `
            <div class="notification-item unread" data-type="${notification.type}" data-id="${notification.id}">
                <div class="notification-dot"></div>
                <div class="notification-avatar ${notification.type}">${notification.avatar}</div>
                <div class="notification-body">
                    <div class="notification-header-text">
                        <span class="notification-title">${notification.title}</span>
                        <span class="notification-time">${notification.time}</span>
                    </div>
                    <div class="notification-content">${notification.content}</div>
                </div>
            </div>
        `;
        
        const notificationsList = document.getElementById('notificationsList');
        if (notificationsList) {
            notificationsList.insertAdjacentHTML('afterbegin', notificationHTML);
            this.updateNotificationCount();
        }
    }

    // 搜索功能相关方法
    toggleSearchInput() {
        const searchContainer = document.querySelector('.search-input-container');
        const searchInput = document.getElementById('nodeSearchInput');
        
        if (searchContainer.style.display === 'none' || !searchContainer.style.display) {
            searchContainer.style.display = 'block';
            searchInput.focus();
        } else {
            searchContainer.style.display = 'none';
            searchInput.value = '';
            this.clearSearchSuggestions();
        }
    }

    initSearchInput() {
        const searchInput = document.getElementById('nodeSearchInput');
        const searchSuggestions = document.getElementById('searchSuggestions');
        
        if (!searchInput || !searchSuggestions) return;

        let debounceTimer;
        let currentSuggestionIndex = -1;

        // 输入事件处理
        searchInput.addEventListener('input', (e) => {
            clearTimeout(debounceTimer);
            const query = e.target.value.trim();
            
            if (query.length === 0) {
                this.clearSearchSuggestions();
                return;
            }

            debounceTimer = setTimeout(() => {
                this.searchNodes(query);
            }, 300);
        });

        // 键盘导航
        searchInput.addEventListener('keydown', (e) => {
            const suggestions = searchSuggestions.querySelectorAll('.suggestion-item');
            
            switch (e.key) {
                case 'ArrowDown':
                    e.preventDefault();
                    currentSuggestionIndex = Math.min(currentSuggestionIndex + 1, suggestions.length - 1);
                    this.highlightSuggestion(suggestions, currentSuggestionIndex);
                    break;
                case 'ArrowUp':
                    e.preventDefault();
                    currentSuggestionIndex = Math.max(currentSuggestionIndex - 1, -1);
                    this.highlightSuggestion(suggestions, currentSuggestionIndex);
                    break;
                case 'Enter':
                    e.preventDefault();
                    if (currentSuggestionIndex >= 0 && suggestions[currentSuggestionIndex]) {
                        this.selectNode(suggestions[currentSuggestionIndex].dataset.nodeId);
                    }
                    break;
                case 'Escape':
                    this.toggleSearchInput();
                    break;
            }
        });

        // 点击外部关闭搜索框
        document.addEventListener('click', (e) => {
            const searchContainer = document.querySelector('.search-container');
            if (!searchContainer.contains(e.target)) {
                const searchInputContainer = document.querySelector('.search-input-container');
                if (searchInputContainer.style.display === 'block') {
                    this.toggleSearchInput();
                }
            }
        });
    }

    async searchNodes(query) {
        try {
            const response = await fetch(`/api/nodes/search?q=${encodeURIComponent(query)}&limit=10`);
            const data = await response.json();
            
            if (data.success && data.suggestions) {
                this.displaySearchSuggestions(data.suggestions);
            } else {
                this.clearSearchSuggestions();
            }
        } catch (error) {
            console.error('Error searching nodes:', error);
            this.clearSearchSuggestions();
        }
    }

    displaySearchSuggestions(suggestions) {
        const searchSuggestions = document.getElementById('searchSuggestions');
        
        if (suggestions.length === 0) {
            searchSuggestions.innerHTML = '<div class="no-suggestions">No matching nodes found</div>';
            return;
        }

        const suggestionsHtml = suggestions.map(suggestion => `
            <div class="suggestion-item" data-node-id="${suggestion.id}">
                <strong>${suggestion.id}</strong>
                ${suggestion.description ? `<br><small>${this.truncateText(suggestion.description, 60)}</small>` : ''}
                <span class="match-type">${this.getMatchTypeLabel(suggestion.match_type)}</span>
            </div>
        `).join('');

        searchSuggestions.innerHTML = suggestionsHtml;

        // 绑定点击事件
        searchSuggestions.querySelectorAll('.suggestion-item').forEach(item => {
            item.addEventListener('click', () => {
                this.selectNode(item.dataset.nodeId);
            });
        });
    }

    truncateText(text, maxLength) {
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength) + '...';
    }

    getMatchTypeLabel(matchType) {
        switch (matchType) {
            case 'id_prefix': return 'Prefix Match';
            case 'id_contains': return 'ID Contains';
            case 'description_contains': return 'Description Match';
            default: return '';
        }
    }

    clearSearchSuggestions() {
        const searchSuggestions = document.getElementById('searchSuggestions');
        if (searchSuggestions) {
            searchSuggestions.innerHTML = '';
        }
    }

    highlightSuggestion(suggestions, index) {
        suggestions.forEach((item, i) => {
            if (i === index) {
                item.classList.add('highlighted');
            } else {
                item.classList.remove('highlighted');
            }
        });
    }

    async selectNode(nodeId) {
        // 隐藏搜索框
        this.toggleSearchInput();
        
        try {
            // 获取节点详细信息
            const response = await fetch(`/api/nodes/${encodeURIComponent(nodeId)}`);
            const data = await response.json();
            
            if (data.success && data.node) {
                // 跳转到节点并显示详细信息
                if (this.mainNetwork) {
                    // 聚焦到节点
                    this.mainNetwork.focus(nodeId, {
                        scale: 1.5,
                        animation: {
                            duration: 1000,
                            easingFunction: 'easeInOutQuad'
                        }
                    });
                    
                    // 选中节点
                    this.mainNetwork.selectNodes([nodeId]);
                    
                    // 显示节点详细信息
                    this.showNodeDetailsFromAPI(data);
                }
            } else {
                console.error('无法获取节点详细信息:', data.error);
                // 尝试使用本地数据显示节点信息
                this.selectNodeFallback(nodeId);
            }
        } catch (error) {
            console.error('获取节点详细信息时出错:', error);
            // 尝试使用本地数据显示节点信息
            this.selectNodeFallback(nodeId);
        }
    }

    selectNodeFallback(nodeId) {
        // 备用方案：使用本地图数据显示节点信息
        if (this.mainNetwork) {
            // 聚焦到节点
            this.mainNetwork.focus(nodeId, {
                scale: 1.5,
                animation: {
                    duration: 1000,
                    easingFunction: 'easeInOutQuad'
                }
            });
            
            // 选中节点
            this.mainNetwork.selectNodes([nodeId]);
            
            // 显示节点详细信息
            const nodes = this.mainNetwork.body.data.nodes;
            const node = nodes.get(nodeId);
            if (node) {
                this.showNodeDetails(node);
            }
        }
    }

    showNodeDetailsFromAPI(apiData) {
        const { node, related_edges, related_nodes, connections_count } = apiData;
        
        // 构建详细信息面板内容
        const detailsPanel = document.querySelector('.element-details');
        if (!detailsPanel) return;

        const detailsContent = `
            <div class="details-header">
                <h3>节点详情</h3>
                <button class="close-details" onclick="chatApp.clearDetailsPanel()">×</button>
            </div>
            <div class="details-content">
                <div class="node-info">
                    <h4>${node.id}</h4>
                    <p class="node-description">${node.description || '暂无描述'}</p>
                    <div class="node-stats">
                        <span class="stat-item">连接数: ${connections_count}</span>
                    </div>
                </div>
                
                ${related_nodes.length > 0 ? `
                <div class="related-nodes">
                    <h5>相关节点 (${related_nodes.length})</h5>
                    <div class="related-nodes-list">
                        ${related_nodes.map(relatedNode => `
                            <div class="related-node-item" onclick="chatApp.selectNode('${relatedNode.id}')">
                                <strong>${relatedNode.id}</strong>
                                <small>${this.truncateText(relatedNode.description || '', 40)}</small>
                            </div>
                        `).join('')}
                    </div>
                </div>
                ` : ''}
                
                ${related_edges.length > 0 ? `
                <div class="related-edges">
                    <h5>相关连接 (${related_edges.length})</h5>
                    <div class="related-edges-list">
                        ${related_edges.slice(0, 5).map(edge => `
                            <div class="related-edge-item">
                                <span class="edge-direction">${edge.from === node.id ? '→' : '←'}</span>
                                <span class="edge-target">${edge.from === node.id ? edge.to : edge.from}</span>
                                ${edge.description ? `<small>${this.truncateText(edge.description, 30)}</small>` : ''}
                            </div>
                        `).join('')}
                        ${related_edges.length > 5 ? `<div class="more-edges">还有 ${related_edges.length - 5} 个连接...</div>` : ''}
                    </div>
                </div>
                ` : ''}
            </div>
        `;

        detailsPanel.innerHTML = detailsContent;
        detailsPanel.style.display = 'block';
    }
    
    // 多层视图相关方法
    switchViewMode(mode) {
        this.currentViewMode = mode;
        
        const flatViewBtn = document.getElementById('flatViewBtn');
        const layeredViewBtn = document.getElementById('layeredViewBtn');
        const layerOptions = document.getElementById('layerOptions');
        
        // 更新按钮状态
        if (flatViewBtn && layeredViewBtn) {
            flatViewBtn.classList.toggle('active', mode === 'flat');
            layeredViewBtn.classList.toggle('active', mode === 'layered');
        }
        
        // 显示/隐藏分层选项
        if (layerOptions) {
            layerOptions.style.display = mode === 'layered' ? 'block' : 'none';
        }
        
        // 重新渲染图谱
        if (mode === 'flat') {
            this.renderFlatGraph();
        } else {
            this.renderLayeredGraph();
        }
    }
    
    renderFlatGraph() {
        if (this.originalGraphData) {
            this.renderMainGraph(this.originalGraphData, true);
        }
    }
    
    renderLayeredGraph() {
        if (!this.originalGraphData) {
            console.error('No original graph data available for layered view');
            return;
        }
        
        const layeredData = this.processGraphDataForLayers(this.originalGraphData, this.currentLayerMode);
        this.layeredGraphData = layeredData;
        this.renderLayeredVisualization(layeredData);
    }
    
    processGraphDataForLayers(graphData, layerMode) {
        const layers = new Map();
        const nodesByLayer = new Map();
        
        // 处理边数据，按时间分层
        graphData.edges.forEach(edge => {
            if (!edge.timestamp) return;
            
            const date = new Date(edge.timestamp);
            let layerKey;
            
            if (layerMode === 'week') {
                // 按周分层：获取年份和周数
                const year = date.getFullYear();
                const weekNumber = this.getWeekNumber(date);
                layerKey = `${year}-W${weekNumber}`;
            } else {
                // 按月分层：获取年份和月份
                const year = date.getFullYear();
                const month = date.getMonth() + 1;
                layerKey = `${year}-${month.toString().padStart(2, '0')}`;
            }
            
            if (!layers.has(layerKey)) {
                layers.set(layerKey, {
                    key: layerKey,
                    edges: [],
                    nodes: new Set(),
                    startDate: date,
                    endDate: date
                });
            }
            
            const layer = layers.get(layerKey);
            layer.edges.push(edge);
            layer.nodes.add(edge.from);
            layer.nodes.add(edge.to);
            
            // 更新时间范围
            if (date < layer.startDate) layer.startDate = date;
            if (date > layer.endDate) layer.endDate = date;
        });
        
        // 为每层添加节点信息
        layers.forEach(layer => {
            layer.nodeDetails = [];
            layer.nodes.forEach(nodeId => {
                const nodeData = graphData.nodes.find(n => n.id === nodeId);
                if (nodeData) {
                    layer.nodeDetails.push(nodeData);
                }
            });
        });
        
        // 按时间排序层
        const sortedLayers = Array.from(layers.values()).sort((a, b) => 
            a.startDate.getTime() - b.startDate.getTime()
        );
        
        return {
            layers: sortedLayers,
            layerMode: layerMode,
            totalNodes: graphData.nodes.length,
            totalEdges: graphData.edges.length
        };
    }
    
    getWeekNumber(date) {
        const d = new Date(Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()));
        const dayNum = d.getUTCDay() || 7;
        d.setUTCDate(d.getUTCDate() + 4 - dayNum);
        const yearStart = new Date(Date.UTC(d.getUTCFullYear(), 0, 1));
        return Math.ceil((((d - yearStart) / 86400000) + 1) / 7);
    }
    
    renderLayeredVisualization(layeredData) {
        const container = document.getElementById('mainGraphContainer');
        if (!container) return;
        
        container.innerHTML = '';
        
        // 创建多层视图容器
        const layeredContainer = document.createElement('div');
        layeredContainer.className = 'layered-graph-container';
        
        // 添加层级信息头部
        const headerDiv = document.createElement('div');
        headerDiv.className = 'layered-header';
        headerDiv.innerHTML = `
            <h4>Multi-layer Temporal Graph - ${layeredData.layerMode === 'week' ? 'Layered by Week' : 'Layered by Month'}</h4>
            <div class="layer-stats">
                Total ${layeredData.layers.length} layers
            </div>
        `;
        layeredContainer.appendChild(headerDiv);
        
        // 为每一层创建可视化
        layeredData.layers.forEach((layer, index) => {
            const layerDiv = document.createElement('div');
            layerDiv.className = 'graph-layer';
            layerDiv.innerHTML = `
                <div class="layer-header">
                    <h5>Layer ${index + 1}: ${layer.key}</h5>
                    <div class="layer-info">
                        ${this.formatDateRange(layer.startDate, layer.endDate)}
                    </div>
                </div>
                <div class="layer-graph" id="layer-${index}"></div>
            `;
            layeredContainer.appendChild(layerDiv);
            
            // 渲染该层的图谱
            setTimeout(() => {
                this.renderLayerGraph(`layer-${index}`, {
                    nodes: layer.nodeDetails,
                    edges: layer.edges
                });
            }, 100 * index); // 延迟渲染以避免性能问题
        });
        
        container.appendChild(layeredContainer);
    }
    
    renderLayerGraph(containerId, graphData) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        // 转换节点数据格式
        const nodes = new vis.DataSet(graphData.nodes.map(node => ({
            id: node.id,
            label: node.id,
            description: node.description,
            color: {
                background: '#8594c8',
                border: this.darkenColor('#8594c8', 0.2),
                highlight: {
                    background: this.lightenColor('#8594c8', 0.2),
                    border: '#8594c8'
                }
            },
            font: {
                color: '#333',
                size: 12,
                face: 'Arial'
            },
            shape: 'dot',
            size: 12
        })));
        
        // 转换边数据格式
        const edges = new vis.DataSet(graphData.edges.map((edge, index) => ({
            id: edge.id || `${edge.from}-${edge.to}-${index}`,
            from: edge.from,
            to: edge.to,
            label: '',
            description: edge.description,
            timestamp: edge.timestamp,
            color: {
                color: '#848484',
                highlight: '#333'
            },
            font: {
                color: '#666',
                size: 10
            },
            arrows: {
                to: { enabled: false }
            },
            smooth: {
                type: 'continuous',
                roundness: 0.2
            }
        })));
        
        // 创建网络图
        const data = { nodes: nodes, edges: edges };
        const options = {
            layout: {
                improvedLayout: true,
                hierarchical: false
            },
            physics: {
                enabled: true,
                stabilization: { iterations: 100 },
                barnesHut: {
                    gravitationalConstant: -3000,
                    centralGravity: 0.05,
                    springLength: 150,
                    springConstant: 0.02,
                    damping: 0.15,
                    avoidOverlap: 0.3
                }
            },
            interaction: {
                hover: true,
                tooltipDelay: 200
            },
            nodes: {
                borderWidth: 1,
                shadow: {
                    enabled: true,
                    color: 'rgba(0,0,0,0.1)',
                    size: 3
                }
            },
            edges: {
                width: 1,
                shadow: {
                    enabled: true,
                    color: 'rgba(0,0,0,0.05)',
                    size: 2
                }
            }
        };
        
        const network = new vis.Network(container, data, options);
        
        // 添加点击事件
        network.on('click', (params) => {
            if (params.nodes.length > 0) {
                const nodeId = params.nodes[0];
                const node = nodes.get(nodeId);
                this.showNodeDetails(node);
            } else if (params.edges.length > 0) {
                const edgeId = params.edges[0];
                const edge = edges.get(edgeId);
                this.showEdgeDetails(edge);
            }
        });
        
        // 网络稳定后调整视图
        network.once('stabilizationIterationsDone', () => {
            network.fit({
                animation: {
                    duration: 500,
                    easingFunction: 'easeInOutQuad'
                }
            });
        });
    }
    
    formatDateRange(startDate, endDate) {
        const options = { month: 'short', day: 'numeric' };
        const start = startDate.toLocaleDateString('en-US', options);
        const end = endDate.toLocaleDateString('en-US', options);
        
        if (start === end) {
            return start;
        }
        return `${start} - ${end}`;
    }

    // Knowledge Graph Modal Functions
    async showKnowledgeGraphModal(notificationId) {
        console.log('showKnowledgeGraphModal called with ID:', notificationId);
        
        // 使用现有的getNotificationData函数，传入id和type参数
        const notificationData = await this.getNotificationData(notificationId, 'report');
        console.log('Retrieved notification data:', notificationData);
        
        if (!notificationData || !notificationData.knowledge_graph) {
            console.error('No knowledge graph data found for notification:', notificationId);
            console.log('Available data keys:', notificationData ? Object.keys(notificationData) : 'null');
            return;
        }
        
        console.log('Knowledge graph data:', notificationData.knowledge_graph);

        // Create modal overlay
        const modalOverlay = document.createElement('div');
        modalOverlay.className = 'knowledge-graph-modal-overlay';
        modalOverlay.innerHTML = `
            <div class="knowledge-graph-modal-content">
                <div class="knowledge-graph-modal-header">
                    <h3>Graph - ${notificationData.title}</h3>
                    <button class="close-modal-btn" onclick="this.closest('.knowledge-graph-modal-overlay').remove()">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <line x1="18" y1="6" x2="6" y2="18"></line>
                            <line x1="6" y1="6" x2="18" y2="18"></line>
                        </svg>
                    </button>
                </div>
                <div class="knowledge-graph-modal-body">
                    <div id="knowledge-graph-container" class="knowledge-graph-container"></div>
                </div>
            </div>
        `;

        document.body.appendChild(modalOverlay);

        // Render the knowledge graph
        setTimeout(() => {
            this.renderKnowledgeGraph('knowledge-graph-container', notificationData.knowledge_graph);
        }, 100);

        // Close modal when clicking outside
        modalOverlay.addEventListener('click', (e) => {
            if (e.target === modalOverlay) {
                modalOverlay.remove();
            }
        });

        // Close modal with Escape key
        const handleEscape = (e) => {
            if (e.key === 'Escape') {
                modalOverlay.remove();
                document.removeEventListener('keydown', handleEscape);
            }
        };
        document.addEventListener('keydown', handleEscape);
    }

    renderKnowledgeGraph(containerId, graphData) {
        console.log('=== 开始渲染知识图谱 (使用vis.js) ===');
        console.log('容器ID:', containerId);
        console.log('图谱数据:', graphData);
        
        // 等待DOM元素创建完成
        setTimeout(() => {
            const container = document.getElementById(containerId);
            if (!container) {
                console.error('知识图谱容器未找到:', containerId);
                return;
            }
            
            // 清空容器
            container.innerHTML = '';
            console.log('容器已清空');
            
            // 验证数据
            if (!graphData || !graphData.nodes || !graphData.edges) {
                console.error('无效的图谱数据:', graphData);
                container.innerHTML = '<div style="text-align: center; padding: 20px; color: #666;">无效的图谱数据</div>';
                return;
            }
            
            console.log('节点数量:', graphData.nodes.length);
            console.log('边数量:', graphData.edges.length);
            
            // 检查vis.js是否可用
            if (typeof vis === 'undefined') {
                console.error('vis.js未加载');
                container.innerHTML = '<div style="text-align: center; padding: 20px; color: #666;">vis.js未加载</div>';
                return;
            }
            
            console.log('vis.js已加载');
            
            // 转换节点数据格式 (使用与renderGraph相同的格式)
            const defaultColor = '#4A90E2'; // 默认蓝色
            const nodes = new vis.DataSet(graphData.nodes.map(node => ({
                id: node.id,
                label: node.id, // 显示节点ID
                title: this.createNodeTooltip(node), // 美化的节点浮窗
                color: {
                    background: defaultColor,
                    border: this.darkenColor(defaultColor, 0.2),
                    highlight: {
                        background: this.lightenColor(defaultColor, 0.2),
                        border: defaultColor
                    }
                },
                font: {
                    color: '#333',
                    size: 30,
                    face: 'Arial'
                },
                shape: 'dot',
                size: 25
            })));
            
            // 转换边数据格式 (使用与renderGraph相同的格式)
            const edges = new vis.DataSet(graphData.edges.map((edge, index) => ({
                id: edge.id || `${edge.from}-${edge.to}-${index}`,
                from: edge.from,
                to: edge.to,
                label: '',
                title: this.createEdgeTooltip(edge), // 美化的边浮窗
                description: edge.description,
                timestamp: edge.timestamp,
                color: {
                    color: '#848484',
                    highlight: '#333'
                },
                font: {
                    color: '#666',
                    size: 12,
                    strokeWidth: 2,
                    strokeColor: '#fff'
                },
                arrows: {
                    to: {
                        enabled: false
                    }
                },
                smooth: {
                    type: 'continuous',
                    roundness: 0.2
                }
            })));
            
            // 创建网络图 (使用与renderGraph相同的配置)
            const data = { nodes: nodes, edges: edges };
            const options = {
                layout: {
                    improvedLayout: true,
                    hierarchical: false
                },
                physics: {
                    enabled: true,
                    stabilization: { iterations: 150 },
                    barnesHut: {
                        gravitationalConstant: -800,
                        centralGravity: 0.05,
                        springLength: 180,
                        springConstant: 0.02,
                        damping: 0.15,
                        avoidOverlap: 0.3
                    }
                },
                interaction: {
                    hover: true,
                    tooltipDelay: 200,
                    hideEdgesOnDrag: false,
                    hideNodesOnDrag: false
                },
                nodes: {
                    borderWidth: 2,
                    shadow: {
                        enabled: true,
                        color: 'rgba(0,0,0,0.2)',
                        size: 5,
                        x: 2,
                        y: 2
                    }
                },
                edges: {
                    width: 2,
                    shadow: {
                        enabled: true,
                        color: 'rgba(0,0,0,0.1)',
                        size: 3,
                        x: 1,
                        y: 1
                    }
                }
            };
            
            const network = new vis.Network(container, data, options);
            
            // 添加交互事件
            network.on('click', (params) => {
                if (params.nodes.length > 0) {
                    const nodeId = params.nodes[0];
                    const node = nodes.get(nodeId);
                    console.log('点击节点:', node);
                } else if (params.edges.length > 0) {
                    const edgeId = params.edges[0];
                    const edge = edges.get(edgeId);
                    console.log('点击边:', edge);
                }
            });
            
            // 网络稳定后调整视图
            network.once('stabilizationIterationsDone', () => {
                network.fit({
                    animation: {
                        duration: 1000,
                        easingFunction: 'easeInOutQuad'
                    }
                });
            });
            
            console.log('知识图谱渲染完成 (vis.js)');
        }, 100);
    }

    getNodeColor(nodeId) {
        // Color nodes based on their type or ID
        const colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3', '#54a0ff'];
        let hash = 0;
        for (let i = 0; i < nodeId.length; i++) {
            hash = nodeId.charCodeAt(i) + ((hash << 5) - hash);
        }
        return colors[Math.abs(hash) % colors.length];
    }

    showKnowledgeGraphTooltip(event, node) {
        const tooltip = document.createElement('div');
        tooltip.className = 'knowledge-graph-tooltip';
        tooltip.innerHTML = `
            <div class="tooltip-header">${node.id}</div>
            <div class="tooltip-content">${node.description}</div>
        `;
        
        document.body.appendChild(tooltip);
        
        const rect = tooltip.getBoundingClientRect();
        tooltip.style.left = (event.pageX - rect.width / 2) + 'px';
        tooltip.style.top = (event.pageY - rect.height - 10) + 'px';
    }

    showKnowledgeGraphEdgeTooltip(event, edge) {
        const tooltip = document.createElement('div');
        tooltip.className = 'knowledge-graph-tooltip';
        tooltip.innerHTML = `
            <div class="tooltip-header">${edge.from} → ${edge.to}</div>
            <div class="tooltip-content">${edge.description}</div>
            <div class="tooltip-timestamp">${edge.timestamp}</div>
        `;
        
        document.body.appendChild(tooltip);
        
        const rect = tooltip.getBoundingClientRect();
        tooltip.style.left = (event.pageX - rect.width / 2) + 'px';
        tooltip.style.top = (event.pageY - rect.height - 10) + 'px';
    }

    hideKnowledgeGraphTooltip() {
        const tooltip = document.querySelector('.knowledge-graph-tooltip');
        if (tooltip) {
            tooltip.remove();
        }
    }

    // Apply suggestion functionality
    applySuggestion(notificationId) {
        // Show success notification
        this.showNotification('The suggestion has been successfully adopted!', 'success');
        
        // Optional: Mark the notification as read
        const notificationCard = document.querySelector(`[data-id="${notificationId}"]`);
        if (notificationCard) {
            this.markNotificationRead(notificationCard);
        }
        
        console.log(`Applied suggestion for notification: ${notificationId}`);
    }
}

// 初始化应用
document.addEventListener('DOMContentLoaded', async () => {
    window.chatApp = new ChatApp();
    
    // 初始化Memory Repository
    if (window.chatApp.initializeMemoryRepository) {
        await window.chatApp.initializeMemoryRepository();
    }
    
    // 初始化Notification功能
    if (window.chatApp.initNotificationFeatures) {
        await window.chatApp.initNotificationFeatures();
    }
});
