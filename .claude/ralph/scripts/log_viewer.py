import json
import os
import threading
import time
from pathlib import Path

from flask import Flask, Response, jsonify, render_template_string
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

app = Flask(__name__)

LOGS_DIR = Path(__file__).parent.parent / "logs"
PLANS_DIR = Path(__file__).parent.parent / "plans"
RALPH_DIR = Path(__file__).parent.parent

file_changes = {}
file_changes_lock = threading.Lock()


class LogFileHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith(".jsonl"):
            with file_changes_lock:
                file_changes[event.src_path] = time.time()


def get_log_files():
    files = []
    for f in LOGS_DIR.glob("*.jsonl"):
        stat = f.stat()
        files.append(
            {
                "name": f.name,
                "size": stat.st_size,
                "mtime": stat.st_mtime,
            }
        )
    files.sort(key=lambda x: x["mtime"], reverse=True)
    return files


def parse_assistant_messages(filepath):
    messages = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get("type") == "assistant":
                        msg = data.get("message", {})
                        content = msg.get("content", [])
                        messages.append(
                            {
                                "line": line_num,
                                "content": content,
                                "model": msg.get("model", "unknown"),
                                "uuid": data.get("uuid", ""),
                            }
                        )
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return messages


def get_prd_data():
    prd_file = PLANS_DIR / "prd.json"
    try:
        with open(prd_file, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading PRD file: {e}")
        return []


def get_progress_content():
    progress_file = RALPH_DIR / "progress.txt"
    try:
        with open(progress_file, encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading progress file: {e}")
        return ""


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Claude Log Viewer</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/11.1.1/marked.min.js"></script>
    <style>
        :root {
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --text-primary: #e6edf3;
            --text-secondary: #8b949e;
            --border-color: #30363d;
            --accent-color: #58a6ff;
            --success-color: #3fb950;
            --warning-color: #d29922;
            --error-color: #f85149;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
        }
        
        /* 顶部 Tab 栏 */
        .top-tabs {
            display: flex;
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border-color);
            padding: 0 16px;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 1002;
            height: 48px;
        }
        
        .top-tab {
            padding: 12px 24px;
            cursor: pointer;
            color: var(--text-secondary);
            border-bottom: 2px solid transparent;
            transition: all 0.2s;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .top-tab:hover {
            color: var(--text-primary);
            background: var(--bg-tertiary);
        }
        
        .top-tab.active {
            color: var(--accent-color);
            border-bottom-color: var(--accent-color);
        }
        
        .app-container {
            margin-top: 48px;
            height: calc(100vh - 48px);
        }
        
        .tab-content {
            display: none;
            height: 100%;
        }
        
        .tab-content.active {
            display: block;
        }
        
        .container {
            display: flex;
            height: 100%;
        }
        
        .sidebar {
            width: 300px;
            background: var(--bg-secondary);
            border-right: 1px solid var(--border-color);
            overflow-y: auto;
            flex-shrink: 0;
        }
        
        .sidebar-header {
            padding: 16px;
            border-bottom: 1px solid var(--border-color);
            position: sticky;
            top: 0;
            background: var(--bg-secondary);
            z-index: 10;
        }
        
        .sidebar-header h1 {
            font-size: 18px;
            font-weight: 600;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .sidebar-header h1::before {
            content: "📋";
        }
        
        .file-list {
            padding: 8px;
        }
        
        .file-item {
            padding: 12px;
            border-radius: 6px;
            cursor: pointer;
            margin-bottom: 4px;
            transition: background 0.2s;
        }
        
        .file-item:hover {
            background: var(--bg-tertiary);
        }
        
        .file-item.active {
            background: var(--accent-color);
            color: white;
        }
        
        .file-item .file-name {
            font-size: 13px;
            font-weight: 500;
            word-break: break-all;
        }
        
        .file-item .file-meta {
            font-size: 11px;
            color: var(--text-secondary);
            margin-top: 4px;
        }
        
        .file-item.active .file-meta {
            color: rgba(255,255,255,0.8);
        }
        
        .main-content {
            flex: 1;
            overflow-y: auto;
            padding: 24px;
        }
        
        /* 自定义滚动条 */
        .main-content::-webkit-scrollbar,
        .sidebar::-webkit-scrollbar {
            width: 10px;
        }
        
        .main-content::-webkit-scrollbar-track,
        .sidebar::-webkit-scrollbar-track {
            background: var(--bg-primary);
        }
        
        .main-content::-webkit-scrollbar-thumb,
        .sidebar::-webkit-scrollbar-thumb {
            background: var(--border-color);
            border-radius: 5px;
        }
        
        .main-content::-webkit-scrollbar-thumb:hover,
        .sidebar::-webkit-scrollbar-thumb:hover {
            background: var(--text-secondary);
        }
        
        /* Firefox 滚动条 */
        .main-content,
        .sidebar {
            scrollbar-width: thin;
            scrollbar-color: var(--border-color) var(--bg-primary);
        }
        
        .message-list {
            max-width: 900px;
            margin: 0 auto;
        }
        
        .message {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            margin-bottom: 16px;
            overflow: hidden;
        }
        
        .message-header {
            padding: 12px 16px;
            background: var(--bg-tertiary);
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 13px;
        }
        
        .message-header .badge {
            background: var(--accent-color);
            color: white;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
        }
        
        .message-header .model {
            color: var(--text-secondary);
        }
        
        .message-header .line-num {
            color: var(--text-secondary);
            margin-left: auto;
        }
        
        .message-body {
            padding: 16px;
        }
        
        .content-item {
            margin-bottom: 12px;
        }
        
        .content-item:last-child {
            margin-bottom: 0;
        }
        
        .text-content {
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        
        .text-content p {
            margin-bottom: 12px;
        }
        
        .text-content p:last-child {
            margin-bottom: 0;
        }
        
        .text-content code {
            background: var(--bg-tertiary);
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'SF Mono', 'Consolas', monospace;
            font-size: 13px;
        }
        
        .text-content pre {
            background: var(--bg-tertiary);
            padding: 16px;
            border-radius: 8px;
            overflow-x: auto;
            margin: 12px 0;
        }
        
        .text-content pre code {
            background: transparent;
            padding: 0;
        }
        
        .tool-use {
            background: var(--bg-tertiary);
            border-radius: 8px;
            padding: 12px 16px;
            border-left: 3px solid var(--warning-color);
        }
        
        .tool-use-header {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
            font-weight: 600;
            color: var(--warning-color);
        }
        
        .tool-use-header::before {
            content: "🔧";
        }
        
        .tool-use-input {
            font-family: 'SF Mono', 'Consolas', monospace;
            font-size: 12px;
            background: var(--bg-primary);
            padding: 12px;
            border-radius: 6px;
            overflow-x: auto;
            max-height: 300px;
            overflow-y: auto;
        }
        
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: var(--text-secondary);
        }
        
        .empty-state h2 {
            font-size: 24px;
            margin-bottom: 8px;
            color: var(--text-primary);
        }
        
        .live-indicator {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 12px;
            color: var(--success-color);
            margin-left: 12px;
        }
        
        .live-dot {
            width: 8px;
            height: 8px;
            background: var(--success-color);
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .scroll-to-bottom {
            position: fixed;
            bottom: 24px;
            right: 24px;
            background: var(--accent-color);
            color: white;
            border: none;
            padding: 12px 20px;
            border-radius: 24px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            display: none;
        }
        
        .scroll-to-bottom:hover {
            background: #79b8ff;
        }
        
        /* 滚动按钮组 */
        .scroll-buttons {
            position: fixed;
            bottom: 24px;
            right: 24px;
            display: none;
            flex-direction: column;
            gap: 8px;
            z-index: 100;
        }
        
        .scroll-buttons.active {
            display: flex;
        }
        
        .scroll-btn {
            background: var(--accent-color);
            color: white;
            border: none;
            width: 44px;
            height: 44px;
            border-radius: 50%;
            cursor: pointer;
            font-size: 18px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background 0.2s, transform 0.2s;
        }
        
        .scroll-btn:hover {
            background: #79b8ff;
            transform: scale(1.1);
        }
        
        .refresh-btn {
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 6px 12px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
            margin-top: 12px;
        }
        
        .refresh-btn:hover {
            background: var(--border-color);
        }
        
        .message-new {
            animation: highlightNew 2s ease-out;
        }
        
        @keyframes highlightNew {
            0% {
                background: rgba(88, 166, 255, 0.2);
                border-color: var(--accent-color);
            }
            100% {
                background: var(--bg-secondary);
                border-color: var(--border-color);
            }
        }
        
        /* PRD 页面样式 */
        .prd-container {
            padding: 24px;
            max-width: 1200px;
            margin: 0 auto;
            height: 100%;
            overflow-y: auto;
        }
        
        .prd-header {
            margin-bottom: 24px;
        }
        
        .prd-header h1 {
            font-size: 24px;
            margin-bottom: 8px;
        }
        
        .prd-stats {
            display: flex;
            gap: 16px;
            flex-wrap: wrap;
            margin-bottom: 24px;
        }
        
        .prd-stat {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 16px 24px;
            min-width: 120px;
        }
        
        .prd-stat-value {
            font-size: 28px;
            font-weight: 600;
            color: var(--accent-color);
        }
        
        .prd-stat-label {
            font-size: 13px;
            color: var(--text-secondary);
        }
        
        .prd-stat.success .prd-stat-value { color: var(--success-color); }
        .prd-stat.warning .prd-stat-value { color: var(--warning-color); }
        .prd-stat.error .prd-stat-value { color: var(--error-color); }
        
        .prd-filter {
            display: flex;
            gap: 8px;
            margin-bottom: 16px;
            flex-wrap: wrap;
        }
        
        .prd-filter-btn {
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
            transition: all 0.2s;
        }
        
        .prd-filter-btn:hover {
            background: var(--border-color);
        }
        
        .prd-filter-btn.active {
            background: var(--accent-color);
            border-color: var(--accent-color);
        }
        
        .prd-list {
            display: grid;
            gap: 12px;
        }
        
        .prd-item {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 16px;
            transition: all 0.2s;
        }
        
        .prd-item:hover {
            border-color: var(--accent-color);
        }
        
        .prd-item-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 8px;
        }
        
        .prd-item-id {
            font-family: 'SF Mono', 'Consolas', monospace;
            font-size: 13px;
            color: var(--accent-color);
            background: var(--bg-tertiary);
            padding: 2px 8px;
            border-radius: 4px;
        }
        
        .prd-item-priority {
            font-size: 11px;
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: 600;
            text-transform: uppercase;
        }
        
        .prd-item-priority.high { background: var(--error-color); color: white; }
        .prd-item-priority.medium { background: var(--warning-color); color: black; }
        .prd-item-priority.low { background: var(--text-secondary); color: white; }
        .prd-item-priority.critical { background: #a333c8; color: white; }
        
        .prd-item-status {
            margin-left: auto;
            font-size: 18px;
        }
        
        .prd-item-title {
            font-size: 15px;
            font-weight: 500;
            margin-bottom: 8px;
        }
        
        .prd-item-requirements {
            list-style: none;
            font-size: 13px;
            color: var(--text-secondary);
        }
        
        .prd-item-requirements li {
            padding: 4px 0;
            padding-left: 20px;
            position: relative;
        }
        
        .prd-item-requirements li::before {
            content: "•";
            position: absolute;
            left: 6px;
            color: var(--text-secondary);
        }
        
        /* Progress 页面样式 */
        .progress-container {
            padding: 24px;
            max-width: 900px;
            margin: 0 auto;
            height: calc(100vh - 60px);
            overflow-y: auto;
            scrollbar-width: thin;
            scrollbar-color: var(--border-color) var(--bg-primary);
        }
        
        .progress-container::-webkit-scrollbar {
            width: 10px;
        }
        
        .progress-container::-webkit-scrollbar-track {
            background: var(--bg-primary);
        }
        
        .progress-container::-webkit-scrollbar-thumb {
            background: var(--border-color);
            border-radius: 5px;
        }
        
        .progress-container::-webkit-scrollbar-thumb:hover {
            background: var(--text-secondary);
        }
        
        .progress-content {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 32px;
        }
        
        .progress-content h1 {
            font-size: 28px;
            margin-bottom: 24px;
            padding-bottom: 16px;
            border-bottom: 1px solid var(--border-color);
        }
        
        .progress-content h2 {
            font-size: 22px;
            margin-top: 32px;
            margin-bottom: 16px;
            color: var(--accent-color);
        }
        
        .progress-content h3 {
            font-size: 18px;
            margin-top: 24px;
            margin-bottom: 12px;
        }
        
        .progress-content p {
            margin-bottom: 12px;
        }
        
        .progress-content ul, .progress-content ol {
            margin-left: 24px;
            margin-bottom: 12px;
        }
        
        .progress-content li {
            margin-bottom: 6px;
        }
        
        .progress-content code {
            background: var(--bg-tertiary);
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'SF Mono', 'Consolas', monospace;
            font-size: 13px;
            word-break: break-all;
            white-space: pre-wrap;
        }
        
        .progress-content pre {
            background: var(--bg-tertiary);
            padding: 16px;
            border-radius: 8px;
            overflow-x: auto;
            margin: 12px 0;
        }
        
        .progress-content pre code {
            background: transparent;
            padding: 0;
        }
        
        .progress-content table {
            width: 100%;
            border-collapse: collapse;
            margin: 16px 0;
        }
        
        .progress-content th, .progress-content td {
            border: 1px solid var(--border-color);
            padding: 10px 14px;
            text-align: left;
        }
        
        .progress-content th {
            background: var(--bg-tertiary);
            font-weight: 600;
        }
        
        .progress-content hr {
            border: none;
            border-top: 1px solid var(--border-color);
            margin: 24px 0;
        }
        
        .progress-content strong {
            color: var(--text-primary);
        }
        
        .progress-content blockquote {
            border-left: 3px solid var(--accent-color);
            margin: 16px 0;
            padding-left: 16px;
            color: var(--text-secondary);
        }
        
        /* 汉堡菜单按钮 */
        .menu-toggle {
            display: none;
            position: fixed;
            top: 60px;
            left: 12px;
            z-index: 1001;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            width: 44px;
            height: 44px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 20px;
            align-items: center;
            justify-content: center;
        }
        
        .menu-toggle:hover {
            background: var(--bg-tertiary);
        }
        
        /* 遮罩层 */
        .sidebar-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.5);
            z-index: 999;
        }
        
        .sidebar-overlay.active {
            display: block;
        }
        
        /* 移动端响应式 */
        @media (max-width: 768px) {
            .menu-toggle {
                display: flex;
            }
            
            .sidebar {
                position: fixed;
                top: 48px;
                left: -100%;
                height: calc(100vh - 48px);
                width: 85%;
                max-width: 320px;
                z-index: 1000;
                transition: left 0.3s ease;
            }
            
            .sidebar.active {
                left: 0;
            }
            
            .sidebar-overlay {
                top: 48px;
            }
            
            .main-content {
                padding: 70px 12px 12px 12px;
            }
            
            .top-tabs {
                padding: 0 8px;
            }
            
            .top-tab {
                padding: 12px 12px;
                font-size: 13px;
            }
            
            .prd-container, .progress-container {
                padding: 16px;
            }
            
            .prd-stat {
                padding: 12px 16px;
                min-width: 100px;
            }
            
            .prd-stat-value {
                font-size: 22px;
            }
            
            .progress-content {
                padding: 20px;
            }
            
            .message-header {
                padding: 10px 12px;
                flex-wrap: wrap;
                gap: 8px;
            }
            
            .message-header .line-num {
                margin-left: 0;
                width: 100%;
                order: 3;
            }
            
            .message-body {
                padding: 12px;
            }
            
            .text-content pre {
                padding: 12px;
                font-size: 11px;
            }
            
            .tool-use {
                padding: 10px 12px;
            }
            
            .tool-use-input {
                padding: 10px;
                font-size: 11px;
                max-height: 200px;
            }
            
            .empty-state {
                padding: 40px 16px;
            }
            
            .empty-state h2 {
                font-size: 20px;
            }
            
            .scroll-to-bottom {
                bottom: 16px;
                right: 16px;
                padding: 10px 16px;
                font-size: 13px;
            }
            
            .live-indicator {
                font-size: 11px;
                margin-left: 8px;
            }
            
            .content-header {
                flex-wrap: wrap;
            }
            
            .content-header h2 {
                font-size: 16px !important;
            }
        }
        
        /* 更小的屏幕 */
        @media (max-width: 480px) {
            .sidebar {
                width: 100%;
                max-width: none;
            }
            
            .message {
                border-radius: 8px;
                margin-bottom: 12px;
            }
            
            .message-header {
                font-size: 12px;
            }
            
            .text-content {
                font-size: 14px;
            }
            
            .text-content code {
                font-size: 12px;
            }
        }
    </style>
</head>
<body>
    <!-- 顶部 Tab 栏 -->
    <div class="top-tabs">
        <div class="top-tab active" onclick="switchTab('logs')" id="tab-logs">
            📋 日志查看
        </div>
        <div class="top-tab" onclick="switchTab('prd')" id="tab-prd">
            📝 PRD 需求
        </div>
        <div class="top-tab" onclick="switchTab('progress')" id="tab-progress">
            📊 开发进度
        </div>
    </div>
    
    <div class="app-container">
        <!-- 日志页面 -->
        <div class="tab-content active" id="content-logs">
            <button class="menu-toggle" id="menuToggle" onclick="toggleSidebar()">☰</button>
            <div class="sidebar-overlay" id="sidebarOverlay" onclick="toggleSidebar()"></div>
            <div class="container">
                <div class="sidebar" id="sidebar">
                    <div class="sidebar-header">
                        <h1>Log Viewer</h1>
                        <button class="refresh-btn" onclick="loadFiles()">🔄 刷新列表</button>
                    </div>
                    <div class="file-list" id="fileList">
                        <!-- 文件列表将在这里动态加载 -->
                    </div>
                </div>
                <div class="main-content" id="mainContent">
                    <div class="empty-state">
                        <h2>👈 选择一个日志文件</h2>
                        <p>从左侧选择一个 JSONL 文件来查看 Assistant 消息</p>
                    </div>
                </div>
            </div>
            <button class="scroll-to-bottom" id="scrollBtn" onclick="scrollToBottom()">
                ⬇️ 滚动到底部
            </button>
        </div>
        
        <!-- PRD 页面 -->
        <div class="tab-content" id="content-prd">
            <div class="prd-container" id="prdContainer">
                <div class="empty-state">
                    <h2>⏳ 加载中...</h2>
                </div>
            </div>
        </div>
        
        <!-- Progress 页面 -->
        <div class="tab-content" id="content-progress">
            <div class="progress-container" id="progressContainer">
                <div class="empty-state">
                    <h2>⏳ 加载中...</h2>
                </div>
            </div>
            <div class="scroll-buttons active" id="progressScrollBtns">
                <button class="scroll-btn" onclick="scrollProgressToTop()" title="回到顶部">⬆️</button>
                <button class="scroll-btn" onclick="scrollProgressToBottom()" title="回到底部">⬇️</button>
            </div>
        </div>
    </div>

    <script>
        marked.setOptions({
            highlight: function(code, lang) {
                if (lang && hljs.getLanguage(lang)) {
                    return hljs.highlight(code, { language: lang }).value;
                }
                return hljs.highlightAuto(code).value;
            }
        });

        let currentFile = null;
        let eventSource = null;
        let autoScroll = true;
        let loadedMessages = [];  // 已加载的消息列表
        let currentTab = 'logs';
        let prdData = null;
        let prdFilter = 'all';

        // Tab 切换功能
        function switchTab(tabName) {
            currentTab = tabName;
            
            // 更新 Tab 样式
            document.querySelectorAll('.top-tab').forEach(tab => {
                tab.classList.remove('active');
            });
            document.getElementById('tab-' + tabName).classList.add('active');
            
            // 切换内容
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });
            document.getElementById('content-' + tabName).classList.add('active');
            
            // 加载对应内容
            if (tabName === 'prd' && !prdData) {
                loadPRD();
            } else if (tabName === 'progress') {
                loadProgress();
            }
        }

        function toggleSidebar() {
            const sidebar = document.getElementById('sidebar');
            const overlay = document.getElementById('sidebarOverlay');
            const menuToggle = document.getElementById('menuToggle');
            
            sidebar.classList.toggle('active');
            overlay.classList.toggle('active');
            menuToggle.textContent = sidebar.classList.contains('active') ? '✕' : '☰';
        }
        
        function closeSidebar() {
            const sidebar = document.getElementById('sidebar');
            const overlay = document.getElementById('sidebarOverlay');
            const menuToggle = document.getElementById('menuToggle');
            
            sidebar.classList.remove('active');
            overlay.classList.remove('active');
            menuToggle.textContent = '☰';
        }

        function formatSize(bytes) {
            if (bytes < 1024) return bytes + ' B';
            if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
            return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
        }

        function formatTime(timestamp) {
            const date = new Date(timestamp * 1000);
            return date.toLocaleString('zh-CN');
        }

        async function loadFiles() {
            try {
                const response = await fetch('/api/files');
                const files = await response.json();
                const fileList = document.getElementById('fileList');
                
                fileList.innerHTML = files.map(file => `
                    <div class="file-item ${currentFile === file.name ? 'active' : ''}" 
                         onclick="loadFile('${file.name}')">
                        <div class="file-name">${file.name}</div>
                        <div class="file-meta">${formatSize(file.size)} · ${formatTime(file.mtime)}</div>
                    </div>
                `).join('');
            } catch (error) {
                console.error('Failed to load files:', error);
            }
        }

        function renderContent(content) {
            return content.map(item => {
                if (item.type === 'text') {
                    return `<div class="content-item text-content">${marked.parse(item.text || '')}</div>`;
                } else if (item.type === 'tool_use') {
                    const inputStr = JSON.stringify(item.input, null, 2);
                    return `
                        <div class="content-item tool-use">
                            <div class="tool-use-header">${item.name}</div>
                            <pre class="tool-use-input"><code>${escapeHtml(inputStr)}</code></pre>
                        </div>
                    `;
                }
                return '';
            }).join('');
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function renderMessages(messages) {
            if (messages.length === 0) {
                return `
                    <div class="empty-state">
                        <h2>📭 没有 Assistant 消息</h2>
                        <p>这个文件中没有找到 assistant 类型的消息</p>
                    </div>
                `;
            }

            return `
                <div class="message-list" id="messageList">
                    ${messages.map(msg => renderSingleMessage(msg)).join('')}
                </div>
            `;
        }

        function renderSingleMessage(msg) {
            return `
                <div class="message" id="msg-${msg.line}" data-line="${msg.line}">
                    <div class="message-header">
                        <span class="badge">Assistant</span>
                        <span class="model">${msg.model}</span>
                        <span class="line-num">Line ${msg.line}</span>
                    </div>
                    <div class="message-body">
                        ${renderContent(msg.content)}
                    </div>
                </div>
            `;
        }

        async function loadFile(filename) {
            currentFile = filename;
            loadedMessages = [];  // 重置已加载消息
            loadFiles(); // 更新选中状态
            closeSidebar(); // 在移动端关闭侧边栏
            
            // 关闭之前的 SSE 连接
            if (eventSource) {
                eventSource.close();
            }

            const mainContent = document.getElementById('mainContent');
            mainContent.innerHTML = '<div class="empty-state"><h2>⏳ 加载中...</h2></div>';

            try {
                const response = await fetch(`/api/messages/${filename}`);
                const messages = await response.json();
                loadedMessages = messages;
                
                mainContent.innerHTML = `
                    <div style="margin-bottom: 16px; display: flex; align-items: center;">
                        <h2 style="font-size: 18px;">${filename}</h2>
                        <span class="live-indicator">
                            <span class="live-dot"></span>
                            实时更新中
                        </span>
                    </div>
                    ${renderMessages(messages)}
                `;

                // 启动 SSE 监听
                startSSE(filename);
                
                // 滚动到底部
                scrollToBottom();
                
                // 代码高亮
                document.querySelectorAll('pre code').forEach(block => {
                    hljs.highlightElement(block);
                });
            } catch (error) {
                mainContent.innerHTML = `
                    <div class="empty-state">
                        <h2>❌ 加载失败</h2>
                        <p>${error.message}</p>
                    </div>
                `;
            }
        }

        async function updateMessages(filename) {
            // 增量更新：只追加新消息
            try {
                const response = await fetch(`/api/messages/${filename}`);
                const messages = await response.json();
                
                if (messages.length === 0) return;
                
                const messageList = document.getElementById('messageList');
                if (!messageList) {
                    // 如果消息列表不存在（之前是空的），重新加载
                    loadFile(filename);
                    return;
                }
                
                // 获取当前最后一条消息的行号
                const lastLoadedLine = loadedMessages.length > 0 
                    ? loadedMessages[loadedMessages.length - 1].line 
                    : 0;
                
                // 找出新消息
                const newMessages = messages.filter(msg => msg.line > lastLoadedLine);
                
                if (newMessages.length > 0) {
                    // 追加新消息
                    newMessages.forEach(msg => {
                        const msgHtml = renderSingleMessage(msg);
                        messageList.insertAdjacentHTML('beforeend', msgHtml);
                        
                        // 为新添加的消息应用代码高亮
                        const newMsgEl = document.getElementById(`msg-${msg.line}`);
                        if (newMsgEl) {
                            newMsgEl.querySelectorAll('pre code').forEach(block => {
                                hljs.highlightElement(block);
                            });
                            // 添加高亮动画效果
                            newMsgEl.classList.add('message-new');
                            setTimeout(() => newMsgEl.classList.remove('message-new'), 2000);
                        }
                    });
                    
                    // 更新已加载消息列表
                    loadedMessages = messages;
                    
                    // 如果用户在底部，自动滚动
                    if (autoScroll) {
                        scrollToBottom();
                    }
                }
            } catch (error) {
                console.error('Failed to update messages:', error);
            }
        }

        function startSSE(filename) {
            eventSource = new EventSource(`/api/stream/${filename}`);
            
            eventSource.onmessage = function(event) {
                if (event.data === 'refresh') {
                    // 使用增量更新而不是完全重新加载
                    updateMessages(filename);
                }
            };

            eventSource.onerror = function() {
                console.log('SSE connection error, will retry...');
            };
        }

        function scrollToBottom() {
            const mainContent = document.getElementById('mainContent');
            mainContent.scrollTop = mainContent.scrollHeight;
        }
        
        function scrollProgressToTop() {
            const progressContainer = document.getElementById('progressContainer');
            progressContainer.scrollTo({ top: 0, behavior: 'smooth' });
        }
        
        function scrollProgressToBottom() {
            const progressContainer = document.getElementById('progressContainer');
            progressContainer.scrollTo({ top: progressContainer.scrollHeight, behavior: 'smooth' });
        }

        // 检测滚动位置
        document.getElementById('mainContent').addEventListener('scroll', function() {
            const scrollBtn = document.getElementById('scrollBtn');
            const isAtBottom = this.scrollHeight - this.scrollTop - this.clientHeight < 100;
            scrollBtn.style.display = isAtBottom ? 'none' : 'block';
            autoScroll = isAtBottom;
        });

        // ========== PRD 功能 ==========
        async function loadPRD() {
            try {
                const response = await fetch('/api/prd');
                prdData = await response.json();
                renderPRD();
            } catch (error) {
                document.getElementById('prdContainer').innerHTML = `
                    <div class="empty-state">
                        <h2>❌ 加载失败</h2>
                        <p>${error.message}</p>
                    </div>
                `;
            }
        }

        function filterPRD(filter) {
            prdFilter = filter;
            renderPRD();
        }

        function renderPRD() {
            if (!prdData) return;
            
            const total = prdData.length;
            const passed = prdData.filter(item => item.passes).length;
            const pending = total - passed;
            const progress = Math.round((passed / total) * 100);
            
            // 根据筛选过滤
            let filteredData = prdData;
            if (prdFilter === 'passed') {
                filteredData = prdData.filter(item => item.passes);
            } else if (prdFilter === 'pending') {
                filteredData = prdData.filter(item => !item.passes);
            } else if (prdFilter !== 'all') {
                filteredData = prdData.filter(item => item.priority === prdFilter);
            }
            
            const container = document.getElementById('prdContainer');
            container.innerHTML = `
                <div class="prd-header">
                    <h1>📝 产品需求文档 (PRD)</h1>
                    <p style="color: var(--text-secondary);">共 ${total} 个需求项</p>
                </div>
                
                <div class="prd-stats">
                    <div class="prd-stat">
                        <div class="prd-stat-value">${total}</div>
                        <div class="prd-stat-label">总需求</div>
                    </div>
                    <div class="prd-stat success">
                        <div class="prd-stat-value">${passed}</div>
                        <div class="prd-stat-label">已完成</div>
                    </div>
                    <div class="prd-stat warning">
                        <div class="prd-stat-value">${pending}</div>
                        <div class="prd-stat-label">待完成</div>
                    </div>
                    <div class="prd-stat">
                        <div class="prd-stat-value">${progress}%</div>
                        <div class="prd-stat-label">完成率</div>
                    </div>
                </div>
                
                <div class="prd-filter">
                    <button class="prd-filter-btn ${prdFilter === 'all' ? 'active' : ''}" onclick="filterPRD('all')">全部</button>
                    <button class="prd-filter-btn ${prdFilter === 'passed' ? 'active' : ''}" onclick="filterPRD('passed')">✅ 已完成</button>
                    <button class="prd-filter-btn ${prdFilter === 'pending' ? 'active' : ''}" onclick="filterPRD('pending')">⏳ 待完成</button>
                    <button class="prd-filter-btn ${prdFilter === 'critical' ? 'active' : ''}" onclick="filterPRD('critical')">Critical</button>
                    <button class="prd-filter-btn ${prdFilter === 'high' ? 'active' : ''}" onclick="filterPRD('high')">High</button>
                    <button class="prd-filter-btn ${prdFilter === 'medium' ? 'active' : ''}" onclick="filterPRD('medium')">Medium</button>
                </div>
                
                <div class="prd-list">
                    ${filteredData.map(item => `
                        <div class="prd-item">
                            <div class="prd-item-header">
                                <span class="prd-item-id">${item.id}</span>
                                <span class="prd-item-priority ${item.priority}">${item.priority}</span>
                                <span class="prd-item-status">${item.passes ? '✅' : '⏳'}</span>
                            </div>
                            <div class="prd-item-title">${escapeHtml(item.story)}</div>
                            <ul class="prd-item-requirements">
                                ${item.requirements.map(req => `<li>${escapeHtml(req)}</li>`).join('')}
                            </ul>
                        </div>
                    `).join('')}
                </div>
            `;
        }

        // ========== Progress 功能 ==========
        async function loadProgress() {
            try {
                const response = await fetch('/api/progress');
                const data = await response.json();
                
                const container = document.getElementById('progressContainer');
                container.innerHTML = `
                    <div class="progress-content">
                        ${marked.parse(data.content)}
                    </div>
                `;
                
                container.querySelectorAll('pre code').forEach(block => {
                    hljs.highlightElement(block);
                });
            } catch (error) {
                document.getElementById('progressContainer').innerHTML = `
                    <div class="empty-state">
                        <h2>❌ 加载失败</h2>
                        <p>${error.message}</p>
                    </div>
                `;
            }
        }

        loadFiles();
        
        setInterval(loadFiles, 10000);
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/files")
def api_files():
    return jsonify(get_log_files())


@app.route("/api/prd")
def api_prd():
    return jsonify(get_prd_data())


@app.route("/api/progress")
def api_progress():
    return jsonify({"content": get_progress_content()})


@app.route("/api/messages/<filename>")
def api_messages(filename):
    filepath = LOGS_DIR / filename
    if not filepath.exists() or not filepath.suffix == ".jsonl":
        return jsonify({"error": "File not found"}), 404

    messages = parse_assistant_messages(filepath)
    return jsonify(messages)


@app.route("/api/stream/<filename>")
def api_stream(filename):
    filepath = LOGS_DIR / filename

    def generate():
        last_size = filepath.stat().st_size if filepath.exists() else 0

        try:
            while True:
                time.sleep(1)
                try:
                    current_size = filepath.stat().st_size
                    if current_size != last_size:
                        last_size = current_size
                        yield f"data: refresh\n\n"
                except (OSError, IOError):
                    pass
        except GeneratorExit:
            return

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def main():
    observer = Observer()
    event_handler = LogFileHandler()
    observer.schedule(event_handler, str(LOGS_DIR), recursive=False)
    observer.start()

    print("📋 Log Viewer 启动中...")
    print("📁 监控目录: {LOGS_DIR}")
    print("🌐 访问地址: http://localhost:5000")
    print("按 Ctrl+C 停止服务")

    try:
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    finally:
        observer.stop()
        observer.join()


if __name__ == "__main__":
    main()
