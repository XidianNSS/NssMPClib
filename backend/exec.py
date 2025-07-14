import sys
import traceback
from io import StringIO
from flask import jsonify, request, Flask
from flask_cors import CORS
from multiprocessing import Process, Queue
import queue
import os
import unicodedata

app = Flask(__name__)
CORS(app)
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

@app.route('/exec', methods=['POST', 'GET'])
def execute():
    code = request.data.decode('utf-8')  # 确保解码为字符串
    output_queue = Queue()
    p = Process(target=_run_code, args=(code, output_queue))
    p.start()
    try:
        output, error, stack_trace = output_queue.get(timeout=30)
    except queue.Empty:
        output, error, stack_trace = "", "执行超时（超过 30 秒）考虑通信send-receive不匹配导致的阻塞，或其中一方代码出错", ""
    finally:
        p.terminate()

    response = {
        "output": output,
        "error": error,
        "stack_trace": stack_trace if error else ""
    }
    return jsonify(response)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

@app.route('/config', methods=['GET', 'POST'])
def config_file():
    save_dir = os.path.expanduser('~/.NssMPClib')
    config_path = os.path.join(save_dir, 'config.json')
    
    if request.method == 'GET':
        # 读取config.json文件
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_content = f.read()
            return {"content": config_content, "exists": True}
        except Exception as e:
            return {"error": f"读取配置文件时出错: {str(e)}"}, 500
    
    elif request.method == 'POST':
        # 保存config.json文件
        try:
            data = request.get_json()
            if not data or 'content' not in data:
                return {"error": "缺少文件内容"}, 400
            
            config_content = data['content']
            
            # 验证JSON格式
            try:
                import json
                json.loads(config_content)
            except json.JSONDecodeError as e:
                return {"error": f"JSON格式错误: {str(e)}"}, 400
            
            # 确保目录存在
            os.makedirs(save_dir, exist_ok=True)
            
            # 保存文件
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(config_content)
            
            return {"message": "配置文件保存成功"}
        except Exception as e:
            return {"error": f"保存配置文件时出错: {str(e)}"}, 500

@app.route('/files', methods=['GET'])
def list_files():
    """列出数据目录下的所有文件和文件夹"""
    base_dir = os.path.expanduser('~/.NssMPClib/data')
    
    try:
        # 确保目录存在
        os.makedirs(base_dir, exist_ok=True)
        
        def get_file_info(path, relative_path=""):
            """递归获取文件信息"""
            items = []
            
            if not os.path.exists(path):
                return items
                
            for item in sorted(os.listdir(path)):
                item_path = os.path.join(path, item)
                relative_item_path = os.path.join(relative_path, item) if relative_path else item
                
                # 跳过隐藏文件
                if item.startswith('.'):
                    continue
                
                try:
                    stat = os.stat(item_path)
                    is_dir = os.path.isdir(item_path)
                    
                    file_info = {
                        "name": item,
                        "path": relative_item_path.replace('\\', '/'),  # 统一使用正斜杠
                        "type": "folder" if is_dir else "file",
                        "size": stat.st_size if not is_dir else 0,
                        "modified": stat.st_mtime,
                        "children": []
                    }
                    
                    # 如果是文件夹，递归获取子内容
                    if is_dir:
                        file_info["children"] = get_file_info(item_path, relative_item_path)
                    
                    items.append(file_info)
                except (OSError, PermissionError):
                    # 跳过无法访问的文件
                    continue
            
            return items
        
        files = get_file_info(base_dir)
        return {"files": files, "base_path": base_dir}
        
    except Exception as e:
        return {"error": f"读取文件列表时出错: {str(e)}"}, 500

@app.route('/files', methods=['DELETE'])
def delete_file():
    """删除指定的文件或文件夹"""
    base_dir = os.path.expanduser('~/.NssMPClib/data')
    
    try:
        data = request.get_json()
        if not data or 'path' not in data:
            return {"error": "缺少文件路径"}, 400
        
        file_path = data['path']
        
        # 安全检查：确保路径在基础目录下
        if '..' in file_path or file_path.startswith('/'):
            return {"error": "无效的文件路径"}, 400
        
        full_path = os.path.join(base_dir, file_path)
        
        # 检查文件是否存在
        if not os.path.exists(full_path):
            return {"error": "文件或文件夹不存在"}, 404
        
        # 确保路径在基础目录内
        real_base = os.path.realpath(base_dir)
        real_target = os.path.realpath(full_path)
        if not real_target.startswith(real_base):
            return {"error": "无权限删除此路径"}, 403
        
        # 删除文件或文件夹
        import shutil
        if os.path.isdir(full_path):
            shutil.rmtree(full_path)
            return {"message": f"成功删除文件夹: {file_path}"}
        else:
            os.remove(full_path)
            return {"message": f"成功删除文件: {file_path}"}
            
    except Exception as e:
        return {"error": f"删除文件时出错: {str(e)}"}, 500

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        # 读取HTML文件并返回
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            html_path = os.path.join(current_dir, 'upload.html')
            with open(html_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return {"error": "上传页面文件未找到"}, 404
        except Exception as e:
            return {"error": f"读取页面文件时出错: {str(e)}"}, 500
    
    # 如果是 POST 请求，处理文件上传
    upload_type = request.form.get('upload_type', 'file')
    
    if upload_type == 'folder':
        return handle_folder_upload()
    else:
        return handle_single_file_upload()

def handle_single_file_upload():
    """处理单个文件上传"""
    file = request.files.get('file')
    if not file or file.filename == '':
        return {"error": "没有选择文件"}, 400
    
    # 验证文件名安全性
    fname = secure_filename(file.filename)
    
    # # 检查扩展名是否允许
    # ext = fname.rsplit('.', 1)[1].lower() if '.' in fname else ''
    # if ext not in {'json', 'pkl', 'csv', 'txt'}:
    #     return {"error": f"不支持的文件类型: {ext}"}, 400
    
    # 确定保存路径
    save_dir = os.path.expanduser('~/.NssMPClib')
    data_dir = os.path.join(save_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    if fname == 'config.json':
        dest_path = os.path.join(save_dir, 'config.json')
    else:
        dest_path = os.path.join(data_dir, fname)
    
    file.save(dest_path)
    return {"message": f"文件上传成功: {fname}"}, 200

def handle_folder_upload():
    """处理文件夹上传"""
    files = request.files.getlist('files')
    # 某些浏览器即使未选择文件夹也会上传一个空文件（filename为''），需特殊处理
    if not files or all(f.filename == '' for f in files):
        return {"error": "没有选择文件夹或文件夹为空"}, 400
    
    uploaded_files = []
    errors = []
    
    # 基础保存目录
    save_dir = os.path.expanduser('~/.NssMPClib')
    data_dir = os.path.join(save_dir, 'data')
    
    for file in files:
        if file.filename == '':
            continue
            
        try:
            # 获取相对路径（浏览器会提供完整的相对路径）
            relative_path = file.filename
            
            # 清理路径中的每个部分
            path_parts = relative_path.split('/')
            clean_parts = []
            for part in path_parts:
                clean_part = secure_filename(part)
                if clean_part:  # 只有非空的部分才添加
                    clean_parts.append(clean_part)
            
            if not clean_parts:
                continue
                
            # clean_filename = clean_parts[-1]
            
            # # 检查文件扩展名
            # ext = clean_filename.rsplit('.', 1)[1].lower() if '.' in clean_filename else ''
            # if ext not in {'json', 'pkl', 'csv', 'txt'}:
            #     errors.append(f"跳过不支持的文件类型: {relative_path} ({ext})")
            #     continue
            
            dest_path = os.path.join(data_dir, *clean_parts)
            
            # 创建目录
            dest_dir = os.path.dirname(dest_path)
            os.makedirs(dest_dir, exist_ok=True)
            
            # 保存文件
            file.save(dest_path)
            uploaded_files.append(relative_path)
            
        except Exception as e:
            errors.append(f"上传文件 {file.filename} 时出错: {str(e)}")
    
    if uploaded_files:
        message = f"成功上传 {len(uploaded_files)} 个文件"
        if errors:
            message += f"，{len(errors)} 个文件有问题"
        
        response = {
            "message": message,
            "uploaded_files": uploaded_files
        }
        if errors:
            response["errors"] = errors
        
        return response, 200
    else:
        return {"error": "没有成功上传任何文件", "errors": errors}, 400

def secure_filename(filename):
    """
    清理文件名，防止目录穿越和特殊字符攻击
    
    Args:
        filename (str): 原始文件名
        
    Returns:
        str: 清理后的安全文件名
    """
    if not filename:
        return ""
    
    # 转换为字符串并标准化Unicode字符
    filename = str(filename)
    filename = unicodedata.normalize('NFKC', filename)
    
    # 移除或替换危险字符
    # 移除路径分隔符和目录遍历字符
    dangerous_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '..']
    for char in dangerous_chars:
        filename = filename.replace(char, '_')
    
    # 移除控制字符
    filename = ''.join(char for char in filename if ord(char) >= 32)
    
    # 移除前后空白字符
    filename = filename.strip()
    
    # 移除前导和尾随的点号（Windows文件名限制）
    filename = filename.strip('.')
    
    # 如果文件名为空或只包含空白字符，返回默认名称
    if not filename or filename.isspace():
        return "uploaded_file"
    
    # 限制文件名长度（包括扩展名）
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        max_name_length = 255 - len(ext)
        filename = name[:max_name_length] + ext
    
    # 确保文件名不以数字开头（避免某些系统的限制）
    if filename and filename[0].isdigit():
        filename = "file_" + filename
    
    return filename

def _run_code(code, output_queue):
    original_stdout = sys.stdout
    buffer = StringIO()
    sys.stdout = Tee(buffer, original_stdout)

    error = None
    stack_trace = None
    try:
        compiled_code = compile(code, "<string>", "exec")
        exec(compiled_code, {})
    except Exception as e:
        error = str(e)
        # 获取完整的堆栈跟踪
        exc_type, exc_value, exc_traceback = sys.exc_info()
        stack_trace = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    finally:
        sys.stdout = original_stdout
        output_queue.put((buffer.getvalue(), error, stack_trace))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="解释器服务")
    parser.add_argument("--port", type=int, help="端口", required=True)
    args = parser.parse_args()

    app.run(host='0.0.0.0', port=args.port)
