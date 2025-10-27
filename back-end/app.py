from flask import Flask, render_template, request, jsonify
import random
import datetime
import re
import json
import os

app = Flask(__name__)

# 日程数据存储文件路径
SCHEDULE_DATA_FILE = 'schedule_data.json'
BASE_GRAPH_DATA_FILE = 'base_graph_data.json'
MOCK_DATABASE_FILE = 'mock_database.json'
GRAPH_DATABASE_FILE = 'graph_database.json'
NOTIFICATION_DATA_FILE = 'notification_data.json'
USER_STATE_FILE = 'user_state.json'

# 加载日程数据
def load_schedule_data():
    if os.path.exists(SCHEDULE_DATA_FILE):
        try:
            with open(SCHEDULE_DATA_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

# 加载基础图数据
def load_base_graph_data():
    if os.path.exists(BASE_GRAPH_DATA_FILE):
        try:
            with open(BASE_GRAPH_DATA_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return None
    return None

# 加载mock数据库
def load_mock_database():
    if os.path.exists(MOCK_DATABASE_FILE):
        try:
            with open(MOCK_DATABASE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []

# 加载图数据库
def load_graph_database():
    if os.path.exists(GRAPH_DATABASE_FILE):
        try:
            with open(GRAPH_DATABASE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []

# 加载通知数据
def load_notification_data():
    if os.path.exists(NOTIFICATION_DATA_FILE):
        try:
            with open(NOTIFICATION_DATA_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []

# 加载用户状态数据
def load_user_state():
    if os.path.exists(USER_STATE_FILE):
        try:
            with open(USER_STATE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {"hasClickedAddToSchedule": False, "lastUpdated": None}
    return {"hasClickedAddToSchedule": False, "lastUpdated": None}

# 保存用户状态数据
def save_user_state(data):
    try:
        with open(USER_STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except:
        return False

# 保存日程数据
def save_schedule_data(data):
    try:
        with open(SCHEDULE_DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except:
        return False



# ==================== 通知数据 ====================
# 通知数据存储 - 从外部JSON文件加载
NOTIFICATION_DATA = load_notification_data()

# ==================== 用户状态数据 ====================
# 用户状态数据存储 - 从外部JSON文件加载
USER_STATE = load_user_state()

# ==================== 模拟检索数据库 ====================
# 模拟数据库，包含一些示例对话记录（用于vector检索）
# 从外部JSON文件加载数据
MOCK_DATABASE = load_mock_database()

# BASE_GRAPH用于Memory Repository界面的独立基础图谱
# 从JSON文件加载基础图数据，如果文件不存在则使用默认数据
BASE_GRAPH = load_base_graph_data() or {
    "id": "personal_life_graph",
    "title": "Personal Life Knowledge Graph",
    "description": "Comprehensive personal life information graph with multiple relationship types",
    "graph_data": {
        "center_node": "myself",
        "nodes": [],
        "edges": []
    }
}

# 图形数据库，包含节点和边的关系数据（用于graph检索）
GRAPH_DATABASE = load_graph_database()

# 简单问题列表 - 不需要检索功能，快速回复，不显示"add to schedule"
SIMPLE_QUESTIONS = {
    "Hello": "Hello! I am your AI assistant. How can I help you today?",
    "你好": "你好！很高兴与你对话，有什么我可以帮助你的吗？",
    "Hi": "Hi there! How can I assist you today?",
    "嗨": "嗨！有什么我可以帮助你的吗？",
    "再见": "再见！希望我们的对话对你有帮助，期待下次交流！",
    "Goodbye": "Goodbye! Have a great day!",
    "Bye": "Bye! Take care!",
    "谢谢": "不客气！我很高兴能够帮助到你。",
    "Thank you": "You're welcome! I'm glad I could help.",
    "Thanks": "You're welcome!",
    "时间": f"当前时间是 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "What time is it": f"Current time is {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "现在几点": f"现在是 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
}

# 预设回复库
PRESET_REPLIES = [
    "这是一个很有趣的问题！让我想想...",
    "我理解你的观点，这确实值得深入思考。",
    "根据我的理解，这个问题可以从多个角度来看。",
    "这让我想到了一个相关的概念...",
    "你提出了一个很好的点，我觉得...",
    "从技术角度来说，这个问题涉及到...",
    "我认为这个话题很重要，因为...",
    "让我为你详细解释一下这个概念。",
    "这是一个常见的疑问，很多人都会遇到。",
    "基于你的描述，我建议...",
    "这个问题的关键在于...",
    "我很乐意帮你分析这个问题。"
]

# 向量搜索专用关键词回复规则
VECTOR_KEYWORD_REPLIES = {
    "你好": "你好！我正在通过向量搜索为你提供最相关的记忆内容，有什么我可以帮助你的吗？",
    "再见": "再见！希望向量搜索帮你找到了有用的信息，期待下次交流！",
    "谢谢": "不客气！很高兴向量检索能够帮助到你。",
    "帮助": "我是一个基于向量搜索的AI助手，可以通过语义相似度为你找到最相关的对话记录和信息。请随时告诉我你需要什么帮助！",
    "天气": "抱歉，我无法获取实时天气信息，但我可以通过向量搜索帮你找到之前关于天气的对话记录。",
    "时间": f"当前时间是 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}，我可以通过向量搜索帮你找到特定时间段的记忆。",
    "编程": "编程是一门很有趣的技能！我可以通过向量搜索帮你找到之前关于编程的对话和学习记录。你想了解哪种编程语言？",
    "Python": "Python是一门非常优秀的编程语言！让我通过向量搜索为你找到相关的Python学习资料和对话记录。",
    "Recommend some activities for me on Saturday.": "How about kicking off your weekend with a rejuvenating city walking tour through a historic district? You could spend a leisurely three hours exploring charming landmarks like vintage theaters, old courthouses, and local artisan markets—just like the one downtown that uncovered hidden bookshops and cozy cafes. If you're craving something more active, consider a beach volleyball session at Santa Monica Beach for some sun, friendly competition, and teamwork under a light ocean breeze. And if you'd rather unwind in nature, a lakeside cabin or mountain lodge retreat offers peaceful trails, fresh air, and the perfect escape from digital noise. Whatever you choose, make it yours!",
}

# 图检索专用关键词回复规则
GRAPH_KEYWORD_REPLIES = {
    "你好": "你好！我正在通过知识图谱为你构建关联性记忆网络，有什么我可以帮助你的吗？",
    "再见": "再见！希望图谱检索帮你发现了有趣的关联关系，期待下次交流！",
    "谢谢": "不客气！很高兴图谱关联分析能够帮助到你。",
    "帮助": "我是一个基于知识图谱的AI助手，可以通过关系网络为你发现信息之间的深层关联。请随时告诉我你需要什么帮助！",
    "天气": "抱歉，我无法获取实时天气信息，但我可以通过图谱关联帮你找到天气相关的活动和记忆网络。",
    "时间": f"当前时间是 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}，我可以通过图谱分析帮你探索时间相关的记忆关联。",
    "编程": "编程是一门很有趣的技能！我可以通过知识图谱帮你发现编程概念之间的关联关系和学习路径。你想了解哪种编程语言？",
    "Python": "Python是一门非常优秀的编程语言！让我通过图谱关联为你展示Python相关的知识网络和学习关系。",
    "Recommend some activities for me on Saturday.": "Since you're still recovering from your leg injury, a low-key but uplifting Saturday at home with friends could be perfect. Consider inviting Alice, Chloe and David over for a relaxed afternoon: cook a simple Thai-inspired dish together (maybe something you learned from Bob’s class), brew some coffee, and follow it up with a movie—perhaps one from David’s Film Club Night lineup. Keep things cozy, seated, and stress-free so you can enjoy good company, creativity, and comfort while letting your leg heal.",
}

# 关键词回复规则
KEYWORD_REPLIES = {
    "你好": "你好！很高兴与你对话，有什么我可以帮助你的吗？",
    "再见": "再见！希望我们的对话对你有帮助，期待下次交流！",
    "谢谢": "不客气！我很高兴能够帮助到你。",
    "帮助": "我是一个AI助手，可以回答各种问题，进行对话交流。请随时告诉我你需要什么帮助！",
    "天气": "抱歉，我无法获取实时天气信息，建议你查看天气预报应用。",
    "时间": f"当前时间是 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "编程": "编程是一门很有趣的技能！你想了解哪种编程语言或者遇到了什么编程问题？",
    "Python": "Python是一门非常优秀的编程语言，语法简洁，功能强大，适合初学者学习！",
    "Recommend some activities for me on Saturday.": "How about kicking off your weekend with a rejuvenating city walking tour through a historic district? You could spend a leisurely three hours exploring charming landmarks like vintage theaters, old courthouses, and local artisan markets—just like the one downtown that uncovered hidden bookshops and cozy cafes. If you’re craving something more active, consider a beach volleyball session at Santa Monica Beach for some sun, friendly competition, and teamwork under a light ocean breeze. And if you’d rather unwind in nature, a lakeside cabin or mountain lodge retreat offers peaceful trails, fresh air, and the perfect escape from digital noise. Whatever you choose, make it yours!",
    # "Recommend some weekend relaxation and entertainment activities for me.": "Since you love coffee and movies, a relaxing weekend could start with a cozy afternoon at a local café with friends Alex Turner, Maya Patel, and Julian Foster—enjoying your favorite brew and good conversation. Given your leg injury, it’s best to avoid strenuous activities. Instead, host a movie marathon at home. Choose a theme, like classic films or feel-good stories, and set the mood with soft blankets, snacks, and soothing music. It’s a calm, enjoyable way to unwind, heal, and stay connected with friends.",
}

def generate_reply(user_message, search_mode='default'):
    """生成模拟回复
    
    Args:
        user_message (str): 用户消息
        search_mode (str): 搜索模式，'vector'、'graph' 或 'default'
    
    Returns:
        tuple: (回复内容, 是否为简单问题)
    """
    user_message_lower = user_message.lower()
    
    # 首先检查是否为简单问题（优先级最高）
    user_message_stripped = user_message.strip()
    for keyword, reply in SIMPLE_QUESTIONS.items():
        if user_message_stripped.lower() == keyword.lower():
            return reply, True  # 返回回复和简单问题标识
    
    # 根据搜索模式选择对应的关键词回复规则
    if search_mode == 'vector':
        keyword_replies = VECTOR_KEYWORD_REPLIES
    elif search_mode == 'graph':
        keyword_replies = GRAPH_KEYWORD_REPLIES
    else:
        keyword_replies = KEYWORD_REPLIES  # 默认使用原有规则
    
    # 检查关键词回复
    for keyword, reply in keyword_replies.items():
        if keyword in user_message:
            return reply, False  # 返回回复和非简单问题标识
    
    # 基于消息长度的简单规则
    if len(user_message) < 10:
        if search_mode == 'vector':
            replies = [
                "能详细说说吗？我会通过向量搜索为你找到相关信息。",
                "这个话题很有意思，你能展开讲讲吗？我可以搜索相关的记忆内容。",
                "我想了解更多细节，这样能更好地进行语义匹配。"
            ]
        elif search_mode == 'graph':
            replies = [
                "能详细说说吗？我会通过图谱关联为你发现相关信息。",
                "这个话题很有意思，你能展开讲讲吗？我可以分析相关的关系网络。",
                "我想了解更多细节，这样能更好地构建关联图谱。"
            ]
        else:
            replies = [
                "能详细说说吗？",
                "这个话题很有意思，你能展开讲讲吗？",
                "我想了解更多细节。"
            ]
        return random.choice(replies), False
    
    # 问号结尾的问题
    if user_message.endswith('？') or user_message.endswith('?'):
        if search_mode == 'vector':
            replies = [
                "这是一个很好的问题！让我通过向量搜索为你找到答案。",
                "让我仔细想想这个问题，并搜索相关的记忆内容...",
                "根据我的理解和向量检索结果，这个问题可以这样看..."
            ]
        elif search_mode == 'graph':
            replies = [
                "这是一个很好的问题！让我通过图谱分析为你找到答案。",
                "让我仔细想想这个问题，并分析相关的关联关系...",
                "根据我的理解和图谱检索结果，这个问题可以这样看..."
            ]
        else:
            replies = [
                "这是一个很好的问题！",
                "让我仔细想想这个问题...",
                "根据我的理解，这个问题可以这样看..."
            ]
        return random.choice(replies) + " " + random.choice(PRESET_REPLIES), False
    
    # 默认随机回复
    return random.choice(PRESET_REPLIES), False

# ==================== 检索功能实现 ====================
# 注意：以下为模拟实现，实际生产环境中需要集成真实的检索引擎

# 文本相似度计算函数已移除，现在使用固定分数简化检索逻辑

def vector_based_search(query, top_k=5):
    """基于向量的检索（简化模拟实现）
    直接为每条记录分配固定分数，简化计算过程
    """
    results = []
    
    # 为每条记录分配固定的相似度分数
    fixed_scores = [95.2, 88.7, 82.3]  # 对应3条记录的分数
    
    for i, item in enumerate(MOCK_DATABASE):
        # 确保type字段存在且有效
        item_type = item.get('type', 'unknown')
        if not item_type or item_type.strip() == '':
            item_type = 'unknown'
        
        # 构建基础结果对象
        result = {
            'id': item['id'],
            'content': item['content'],
            'score': fixed_scores[i] if i < len(fixed_scores) else 75.0,
            'retrieval_type': 'vector',
            'created_at': item['created_at'],
            'type': item_type,  # 使用处理后的type字段
        }
        
        # 根据类型添加特定字段
        if item_type == 'email':
            # 邮件类型的特定字段
            result.update({
                'subject': item.get('subject', 'No Subject'),
                'sender': item.get('sender', '未知发件人'),
                'recipient': item.get('recipient', '未知收件人'),
                'sender_email': item.get('sender_email', ''),
                'recipient_email': item.get('recipient_email', '')
            })
        elif item_type == 'activity':
            # 活动类型的特定字段
            result.update({
                'title': item.get('title', ''),
                'activity_type': item.get('activity_type', ''),
                'duration': item.get('duration', ''),
                'location': item.get('location', '')
            })
        else:
            # 其他类型的通用字段
            result.update({
                'title': item.get('title', ''),
                'activity_type': item.get('activity_type', ''),
                'duration': item.get('duration', ''),
                'location': item.get('location', '')
            })
            
        results.append(result)
    
    # 按相似度排序并返回top_k结果
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_k]

def graph_based_search(query, top_k=5):
    """基于图的检索（简化模拟实现）
    直接为每条记录分配固定分数，简化计算过程
    """
    results = []
    
    # 为每条记录分配固定的图关联性分数，优先显示最新的个人生活知识图谱
    fixed_scores = [95.0, 91.5, 85.3]  # 对应3条记录的分数，第3个（个人生活知识图谱）得分最高
    
    for i, item in enumerate(GRAPH_DATABASE):
        score = fixed_scores[i] if i < len(fixed_scores) else 70.0
        
        # 如果查询包含"Personal Life"关键词，给个人生活知识图谱更高的分数
        if "Personal Life" in query and item['query'] == "Personal Life Knowledge Graph":
            score = 100.0
        
        results.append({
            'id': item['id'],
            'content': item['graph_data'],  # 返回图形数据而不是对话内容
            'query': item['query'],  # 添加子问题
            'query_type': item['query_type'],  # 添加查询类型
            'time_interval': item['time_interval'],  # 添加时间区间
            'retrieval_type': 'graph',
            'score': score
        })
    
    # 按关联性分数排序并返回top_k结果
    results.sort(key=lambda x: x.get('score', 70.0), reverse=True)
    return results[:top_k]

def perform_search(query, search_mode='vector', top_k=5):
    """执行检索的统一接口
    
    Args:
        query (str): 检索查询
        search_mode (str): 检索模式，'vector' 或 'graph'
        top_k (int): 返回结果数量
    
    Returns:
        list: 检索结果列表
    """
    if not query or not query.strip():
        return []
    
    if search_mode == 'vector':
        return vector_based_search(query.strip(), top_k)
    elif search_mode == 'graph':
        return graph_based_search(query.strip(), top_k)
    else:
        # 默认使用向量检索
        return vector_based_search(query.strip(), top_k)

@app.route('/')
def index():
    """主页路由"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """处理对话请求的API端点"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        search_mode = data.get('search_mode', 'default')  # 获取搜索模式，默认为'default'
        
        if not user_message:
            return jsonify({
                'success': False,
                'error': '消息不能为空'
            }), 400
        
        # 生成回复，传入搜索模式
        reply, is_simple_question = generate_reply(user_message, search_mode)
        
        return jsonify({
            'success': True,
            'reply': reply,
            'search_mode': search_mode,  # 返回使用的搜索模式
            'is_simple_question': is_simple_question,  # 返回是否为简单问题
            'timestamp': datetime.datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'服务器错误: {str(e)}'
        }), 500

@app.route('/api/base-graph', methods=['GET'])
def get_base_graph():
    """获取Memory Repository基础图谱的独立API端点
    
    返回专门用于Memory Repository界面显示的基础知识图谱
    与搜索功能完全解耦，确保基础图谱的独立性
    支持时间范围过滤参数：start_time 和 end_time
    """
    try:
        # 获取时间过滤参数
        start_time = request.args.get('start_time')
        end_time = request.args.get('end_time')
        
        graph_data = BASE_GRAPH['graph_data']
        
        # 如果提供了时间过滤参数，则过滤边数据
        if start_time or end_time:
            filtered_edges = []
            
            for edge in graph_data['edges']:
                edge_timestamp = edge.get('timestamp')
                if not edge_timestamp:
                    # 如果边没有时间戳，默认包含
                    filtered_edges.append(edge)
                    continue
                
                # 检查时间范围
                include_edge = True
                
                if start_time and edge_timestamp < start_time:
                    include_edge = False
                
                if end_time and edge_timestamp > end_time:
                    include_edge = False
                
                if include_edge:
                    filtered_edges.append(edge)
            
            # 获取过滤后边中涉及的所有节点ID
            edge_node_ids = set()
            for edge in filtered_edges:
                edge_node_ids.add(edge['from'])
                edge_node_ids.add(edge['to'])
            
            # 过滤节点，只保留在过滤后的边中出现的节点
            filtered_nodes = [node for node in graph_data['nodes'] if node['id'] in edge_node_ids]
            
            # 构建过滤后的图谱数据
            filtered_graph_data = {
                'nodes': filtered_nodes,
                'edges': filtered_edges
            }
        else:
            # 如果没有时间过滤参数，返回完整数据
            filtered_graph_data = graph_data
        
        return jsonify({
            'success': True,
            'graph_data': filtered_graph_data,
            'title': BASE_GRAPH['title'],
            'description': BASE_GRAPH['description'],
            'timestamp': datetime.datetime.now().isoformat(),
            'filter_applied': bool(start_time or end_time),
            'start_time': start_time,
            'end_time': end_time
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Base graph service error: {str(e)}',
            'graph_data': None
        }), 500

@app.route('/api/search', methods=['POST'])
def search():
    """检索API端点
    
    接收前端的检索请求，返回相关的数据库检索结果
    注意：当前为模拟实现，实际生产环境中需要集成真实的检索系统
    """
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        search_mode = data.get('mode', 'vector')  # 默认向量检索
        top_k = data.get('top_k', 5)  # 默认返回5个结果
        
        if not query:
            return jsonify({
                'success': False,
                'error': 'Query cannot be empty',
                'results': []
            }), 400
        
        # 执行检索
        search_results = perform_search(query, search_mode, top_k)
        
        return jsonify({
            'success': True,
            'query': query,
            'mode': search_mode,
            'total_results': len(search_results),
            'results': search_results,
            'timestamp': datetime.datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Search service error: {str(e)}',
            'results': []
        }), 500

@app.route('/api/nodes/search', methods=['GET'])
def search_nodes():
    """节点搜索API端点
    
    根据前缀搜索节点ID，提供联想功能
    """
    try:
        query = request.args.get('q', '').strip()
        limit = int(request.args.get('limit', 10))
        
        if not query:
            return jsonify({
                'success': True,
                'query': query,
                'suggestions': []
            })
        
        # 加载图数据
        base_data = load_base_graph_data()
        if not base_data or 'graph_data' not in base_data:
            return jsonify({
                'success': False,
                'error': 'Graph data not available',
                'suggestions': []
            }), 500
        
        graph_data = base_data['graph_data']
        if 'nodes' not in graph_data:
            return jsonify({
                'success': False,
                'error': 'Nodes data not available',
                'suggestions': []
            }), 500
        
        # 搜索匹配的节点
        suggestions = []
        query_lower = query.lower()
        
        for node in graph_data['nodes']:
            node_id = node.get('id', '')
            node_desc = node.get('description', '')
            
            # 检查ID是否匹配前缀
            if node_id.lower().startswith(query_lower):
                suggestions.append({
                    'id': node_id,
                    'description': node_desc,
                    'match_type': 'id_prefix'
                })
            # 检查描述是否包含查询词
            elif query_lower in node_desc.lower():
                suggestions.append({
                    'id': node_id,
                    'description': node_desc,
                    'match_type': 'description'
                })
            
            # 限制结果数量
            if len(suggestions) >= limit:
                break
        
        return jsonify({
            'success': True,
            'query': query,
            'suggestions': suggestions
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'suggestions': []
        }), 500


@app.route('/api/nodes/<node_id>', methods=['GET'])
def get_node_details(node_id):
    """获取节点详细信息API端点
    
    返回指定节点的详细信息和相关连接
    """
    try:
        # 加载图数据
        base_data = load_base_graph_data()
        if not base_data or 'graph_data' not in base_data:
            return jsonify({
                'success': False,
                'error': 'Graph data not available'
            }), 500
        
        graph_data = base_data['graph_data']
        
        # 查找节点
        target_node = None
        for node in graph_data.get('nodes', []):
            if node.get('id') == node_id:
                target_node = node
                break
        
        if not target_node:
            return jsonify({
                'success': False,
                'error': f'Node {node_id} not found'
            }), 404
        
        # 查找相关边
        related_edges = []
        related_nodes = set()
        
        for edge in graph_data.get('edges', []):
            if edge.get('source') == node_id:
                related_edges.append({
                    'direction': 'out',
                    'target': edge.get('target'),
                    'relationship': edge.get('relationship', '')
                })
                related_nodes.add(edge.get('target'))
            elif edge.get('target') == node_id:
                related_edges.append({
                    'direction': 'in',
                    'source': edge.get('source'),
                    'relationship': edge.get('relationship', '')
                })
                related_nodes.add(edge.get('source'))
        
        # 获取相关节点的详细信息
        related_node_details = []
        for node in graph_data.get('nodes', []):
            if node.get('id') in related_nodes:
                related_node_details.append({
                    'id': node.get('id'),
                    'description': node.get('description', '')
                })
        
        return jsonify({
            'success': True,
            'node': target_node,
            'related_edges': related_edges,
            'related_nodes': related_node_details,
            'stats': {
                'total_connections': len(related_edges),
                'outgoing_connections': len([e for e in related_edges if e['direction'] == 'out']),
                'incoming_connections': len([e for e in related_edges if e['direction'] == 'in'])
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Node details error: {str(e)}'
        }), 500


@app.route('/api/notifications', methods=['GET'])
def get_notifications():
    """获取通知数据的API端点"""
    try:
        # 获取查询参数
        filter_type = request.args.get('filter', 'all')  # all, unread, report, reminder, suggestion
        
        # 根据过滤条件筛选通知
        filtered_notifications = []
        
        # 过滤通知数据，只有在用户点击过add to schedule后才显示Memory Update通知
        available_notifications = []
        for notification in NOTIFICATION_DATA:
            # 如果是Memory Update通知（id为1），只有在用户点击过add to schedule后才显示
            if notification.get('id') == 1 and notification.get('title') == 'Memory Update':
                if USER_STATE.get('hasClickedAddToSchedule', False):
                    available_notifications.append(notification)
            else:
                available_notifications.append(notification)
        
        if filter_type == 'all':
            filtered_notifications = available_notifications
        elif filter_type == 'unread':
            filtered_notifications = [n for n in available_notifications if not n['isRead']]
        elif filter_type in ['report', 'reminder', 'suggestion']:
            filtered_notifications = [n for n in available_notifications if n['type'] == filter_type]
        else:
            filtered_notifications = available_notifications
        
        # 计算统计信息
        total_count = len(available_notifications)
        unread_count = len([n for n in available_notifications if not n['isRead']])
        filtered_count = len(filtered_notifications)
        
        return jsonify({
            'success': True,
            'notifications': filtered_notifications,
            'stats': {
                'total': total_count,
                'unread': unread_count,
                'filtered': filtered_count
            },
            'filter': filter_type,
            'timestamp': datetime.datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'获取通知数据失败: {str(e)}',
            'notifications': [],
            'stats': {'total': 0, 'unread': 0, 'filtered': 0}
        }), 500

@app.route('/api/notifications/<int:notification_id>', methods=['PUT'])
def update_notification(notification_id):
    """更新通知状态的API端点"""
    try:
        data = request.get_json()
        action = data.get('action', '')
        
        # 查找通知
        notification = None
        for n in NOTIFICATION_DATA:
            if n['id'] == notification_id:
                notification = n
                break
        
        if not notification:
            return jsonify({
                'success': False,
                'error': '通知不存在'
            }), 404
        
        # 处理不同的操作
        if action == 'mark_read':
            notification['isRead'] = True
        elif action == 'mark_unread':
            notification['isRead'] = False
        elif action == 'dismiss':
            # 在实际应用中，这里可能会删除通知或标记为已忽略
            notification['isRead'] = True
        
        return jsonify({
            'success': True,
            'message': '通知状态更新成功',
            'notification': notification,
            'timestamp': datetime.datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'更新通知状态失败: {str(e)}'
        }), 500

@app.route('/api/notifications/batch', methods=['PUT'])
def batch_update_notifications():
    """批量更新通知状态的API端点"""
    try:
        data = request.get_json()
        action = data.get('action', '')
        notification_ids = data.get('ids', [])
        
        updated_count = 0
        
        if action == 'mark_all_read':
            # 标记所有通知为已读
            for notification in NOTIFICATION_DATA:
                if not notification['isRead']:
                    notification['isRead'] = True
                    updated_count += 1
        elif action == 'clear_all':
            # 在实际应用中，这里可能会删除所有通知
            # 这里我们只是标记为已读
            for notification in NOTIFICATION_DATA:
                if not notification['isRead']:
                    notification['isRead'] = True
                    updated_count += 1
        elif action == 'mark_selected_read' and notification_ids:
            # 标记选中的通知为已读
            for notification in NOTIFICATION_DATA:
                if notification['id'] in notification_ids and not notification['isRead']:
                    notification['isRead'] = True
                    updated_count += 1
        
        return jsonify({
            'success': True,
            'message': f'成功更新了 {updated_count} 条通知',
            'updated_count': updated_count,
            'timestamp': datetime.datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'批量更新通知失败: {str(e)}'
        }), 500

@app.route('/schedule')
def schedule():
    """日程管理页面路由"""
    return render_template('index.html')

@app.route('/notification')
def notification():
    """通知中心页面路由"""
    return render_template('index.html')

@app.route('/api/health')
def health_check():
    """健康检查端点"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.datetime.now().isoformat()
    })

@app.route('/api/schedule', methods=['GET'])
def get_schedule():
    """获取日程数据"""
    try:
        # 直接从文件加载日程数据
        schedule_data = load_schedule_data()
        
        # 检查用户是否点击过add to schedule
        user_state = load_user_state()
        if not user_state.get('hasClickedAddToSchedule', False):
            # 如果用户没有点击过add to schedule，过滤掉Friends Gathering事件
            filtered_schedule = {}
            for date, events in schedule_data.items():
                filtered_events = []
                for event in events:
                    # 过滤掉title为"Friends Gathering"的事件
                    if event.get('title') != 'Friends Gathering':
                        filtered_events.append(event)
                if filtered_events:  # 只有当日期有事件时才添加
                    filtered_schedule[date] = filtered_events
            schedule_data = filtered_schedule
        
        return jsonify({
            'success': True,
            'data': schedule_data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/schedule', methods=['POST'])
def save_schedule():
    """保存日程数据"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': '无效的数据格式'
            }), 400
        
        success = save_schedule_data(data)
        if success:
            return jsonify({
                'success': True,
                'message': '日程数据保存成功'
            })
        else:
            return jsonify({
                'success': False,
                'error': '保存失败'
            }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/schedule/events', methods=['POST'])
def add_event():
    """添加日程事件"""
    try:
        event_data = request.get_json()
        if not event_data or 'date' not in event_data:
            return jsonify({
                'success': False,
                'error': '缺少必要的事件数据'
            }), 400
        
        schedule_data = load_schedule_data()
        date_key = event_data['date']
        
        if date_key not in schedule_data:
            schedule_data[date_key] = []
        
        # 添加时间戳
        event_data['created_at'] = datetime.now().isoformat()
        event_data['id'] = len(schedule_data[date_key])
        
        schedule_data[date_key].append(event_data)
        
        success = save_schedule_data(schedule_data)
        if success:
            return jsonify({
                'success': True,
                'message': '事件添加成功',
                'event': event_data
            })
        else:
            return jsonify({
                'success': False,
                'error': '保存失败'
            }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/schedule/events/<date>/<int:event_id>', methods=['PUT'])
def update_event(date, event_id):
    """更新日程事件"""
    try:
        event_data = request.get_json()
        if not event_data:
            return jsonify({
                'success': False,
                'error': '无效的事件数据'
            }), 400
        
        schedule_data = load_schedule_data()
        
        if date not in schedule_data or event_id >= len(schedule_data[date]):
            return jsonify({
                'success': False,
                'error': '事件不存在'
            }), 404
        
        # 保留原有的创建时间和ID
        original_event = schedule_data[date][event_id]
        event_data['created_at'] = original_event.get('created_at')
        event_data['id'] = original_event.get('id', event_id)
        event_data['updated_at'] = datetime.now().isoformat()
        
        schedule_data[date][event_id] = event_data
        
        success = save_schedule_data(schedule_data)
        if success:
            return jsonify({
                'success': True,
                'message': '事件更新成功',
                'event': event_data
            })
        else:
            return jsonify({
                'success': False,
                'error': '保存失败'
            }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/schedule/events/<date>/<int:event_id>', methods=['DELETE'])
def delete_event(date, event_id):
    """删除日程事件"""
    try:
        schedule_data = load_schedule_data()
        
        if date not in schedule_data or event_id >= len(schedule_data[date]):
            return jsonify({
                'success': False,
                'error': '事件不存在'
            }), 404
        
        deleted_event = schedule_data[date].pop(event_id)
        
        # 如果该日期没有事件了，删除该日期键
        if not schedule_data[date]:
            del schedule_data[date]
        else:
            # 重新分配ID
            for i, event in enumerate(schedule_data[date]):
                event['id'] = i
        
        success = save_schedule_data(schedule_data)
        if success:
            return jsonify({
                'success': True,
                'message': '事件删除成功',
                'deleted_event': deleted_event
            })
        else:
            return jsonify({
                'success': False,
                'error': '保存失败'
            }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/add-to-schedule', methods=['POST'])
def add_to_schedule():
    """处理添加到日程的请求，并更新用户状态"""
    global USER_STATE
    try:
        data = request.get_json()
        content = data.get('content', '')
        
        # 更新用户状态，标记已点击过add to schedule
        USER_STATE['hasClickedAddToSchedule'] = True
        USER_STATE['lastUpdated'] = datetime.datetime.now().isoformat()
        
        # 保存用户状态到文件
        save_user_state(USER_STATE)
        
        # 这里可以添加实际的日程添加逻辑
        # 目前只是更新状态
        
        return jsonify({
            'success': True,
            'message': 'Activity added to schedule successfully!',
            'userState': USER_STATE
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/user-state', methods=['GET'])
def get_user_state():
    """获取用户状态"""
    return jsonify({
        'success': True,
        'userState': USER_STATE
    })

@app.route('/api/upload-file', methods=['POST'])
def upload_file():
    """处理文件上传，支持进度反馈"""
    import time
    import uuid
    
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': '没有选择文件'
            }), 400
        
        file = request.files['file']
        file_type = request.form.get('type', 'unknown')
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': '文件名为空'
            }), 400
        
        # 生成唯一的任务ID
        task_id = str(uuid.uuid4())
        
        # 模拟文件处理过程，返回任务ID
        return jsonify({
            'success': True,
            'task_id': task_id,
            'message': '文件上传开始处理'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# 全局任务状态存储
TASK_STATUS = {}

@app.route('/api/upload-progress/<task_id>', methods=['GET'])
def get_upload_progress(task_id):
    """获取文件上传进度"""
    import time
    import random
    import hashlib
    
    try:
        # 如果任务已完成，直接返回完成状态
        if task_id in TASK_STATUS and TASK_STATUS[task_id].get('completed', False):
            return jsonify({
                'success': True,
                'task_id': task_id,
                'progress': 100,
                'message': '处理完成！',
                'is_complete': True
            })
        
        # 初始化任务状态
        if task_id not in TASK_STATUS:
            TASK_STATUS[task_id] = {
                'start_time': time.time(),
                'completed': False
            }
        
        # 使用任务ID创建一个伪随机种子
        seed = int(hashlib.md5(task_id.encode()).hexdigest()[:8], 16)
        random.seed(seed)
        
        # 计算从任务开始到现在的时间
        task_start_time = TASK_STATUS[task_id]['start_time']
        current_time = time.time()
        elapsed_time = (current_time - task_start_time) * 1000  # 转换为毫秒
        
        # 创建不规律的进度阶段 - 调整为更快的填充速度
        progress_points = [0, 15, 25, 45, 60, 75, 85, 95, 100]
        time_points = [0, 800, 1600, 2500, 3500, 4300, 5000, 5500, 6000]  # 从10秒缩短到6秒
        
        # 添加随机波动
        for i in range(1, len(progress_points) - 1):
            # 随机调整进度点（±5%）
            variation = random.uniform(-5, 5)
            progress_points[i] = max(0, min(100, progress_points[i] + variation))
            
            # 随机调整时间点（±200ms）  # 减少时间变化范围以适应更快的速度
            time_variation = random.uniform(-200, 200)
            time_points[i] = max(0, time_points[i] + time_variation)
        
        # 根据当前时间计算进度
        current_progress = 0
        for i in range(len(time_points) - 1):
            if elapsed_time >= time_points[i] and elapsed_time <= time_points[i + 1]:
                # 在两个点之间进行插值，但添加随机抖动
                time_ratio = (elapsed_time - time_points[i]) / (time_points[i + 1] - time_points[i])
                base_progress = progress_points[i] + (progress_points[i + 1] - progress_points[i]) * time_ratio
                
                # 添加小幅随机抖动（±2%）
                jitter = random.uniform(-2, 2)
                current_progress = max(0, min(100, base_progress + jitter))
                break
        
        if elapsed_time >= time_points[-1]:
            current_progress = 100
            # 标记任务为已完成
            TASK_STATUS[task_id]['completed'] = True
        
        # 确保进度是整数
        current_progress = int(current_progress)
        
        # 模拟不同阶段的消息
        if current_progress < 20:
            message = '正在解析文件...'
        elif current_progress < 40:
            message = '正在提取内容...'
        elif current_progress < 65:
            message = '正在处理数据...'
        elif current_progress < 85:
            message = '正在建立关联...'
        elif current_progress < 100:
            message = '正在保存结果...'
        else:
            message = '处理完成！'
        
        # 如果进度达到100%，标记为完成
        is_complete = current_progress >= 100
        
        return jsonify({
            'success': True,
            'task_id': task_id,
            'progress': current_progress,
            'message': message,
            'is_complete': is_complete
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/update-import-status', methods=['POST'])
def update_import_status():
    """更新导入记忆状态"""
    global USER_STATE
    
    try:
        data = request.get_json()
        if 'hasImportedMemory' in data:
            USER_STATE['hasImportedMemory'] = data['hasImportedMemory']
            USER_STATE['lastUpdated'] = datetime.datetime.now().isoformat()
            
            # 保存状态到文件
            save_user_state(USER_STATE)
            
            return jsonify({
                'success': True,
                'message': 'Import status updated successfully',
                'userState': USER_STATE
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Missing hasImportedMemory field'
            }), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False)