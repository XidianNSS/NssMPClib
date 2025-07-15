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
    code = request.data.decode('utf-8')  # Ensure decoding to string
    output_queue = Queue()
    p = Process(target=_run_code, args=(code, output_queue))
    p.start()
    try:
        output, error, stack_trace = output_queue.get(timeout=30)
    except queue.Empty:
        output, error, stack_trace = "", "Timeout (over 30 seconds). Possible blocking due to send-receive mismatch or an error in one side's code.", ""
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
        # read config.json file
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_content = f.read()
            return {"content": config_content, "exists": True}
        except Exception as e:
            return {"error": f"error reading config file: {str(e)}"}, 500
    
    elif request.method == 'POST':
        # save config.json file
        try:
            data = request.get_json()
            if not data or 'content' not in data:
                return {"error": "missing file content"}, 400
            
            config_content = data['content']
            
            # validate JSON format
            try:
                import json
                json.loads(config_content)
            except json.JSONDecodeError as e:
                return {"error": f"JSON format error: {str(e)}"}, 400
            
            # ensure directory exists
            os.makedirs(save_dir, exist_ok=True)
            
            # save file
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(config_content)
            
            return {"message": "config file saved successfully"}
        except Exception as e:
            return {"error": f"error saving config file: {str(e)}"}, 500

@app.route('/files', methods=['GET'])
def list_files():
    """list all files and folders in data directory"""
    base_dir = os.path.expanduser('~/.NssMPClib/data')
    
    try:
        # ensure directory exists
        os.makedirs(base_dir, exist_ok=True)
        
        def get_file_info(path, relative_path=""):
            """recursively get file information"""
            items = []
            
            if not os.path.exists(path):
                return items
                
            for item in sorted(os.listdir(path)):
                item_path = os.path.join(path, item)
                relative_item_path = os.path.join(relative_path, item) if relative_path else item
                
                # skip hidden files
                if item.startswith('.'):
                    continue
                
                try:
                    stat = os.stat(item_path)
                    is_dir = os.path.isdir(item_path)
                    
                    file_info = {
                        "name": item,
                        "path": relative_item_path.replace('\\', '/'),  # use forward slash for consistency
                        "type": "folder" if is_dir else "file",
                        "size": stat.st_size if not is_dir else 0,
                        "modified": stat.st_mtime,
                        "children": []
                    }
                    
                    # if it's a folder, recursively get sub-content
                    if is_dir:
                        file_info["children"] = get_file_info(item_path, relative_item_path)
                    
                    items.append(file_info)
                except (OSError, PermissionError):
                    # skip inaccessible files
                    continue
            
            return items
        
        files = get_file_info(base_dir)
        return {"files": files, "base_path": base_dir}
        
    except Exception as e:
        return {"error": f"error reading file list: {str(e)}"}, 500

@app.route('/files', methods=['DELETE'])
def delete_file():
    """delete specified file or folder"""
    base_dir = os.path.expanduser('~/.NssMPClib/data')
    
    try:
        data = request.get_json()
        if not data or 'path' not in data:
            return {"error": "missing file path"}, 400
        
        file_path = data['path']
        
        # security check: ensure path is within base directory
        if '..' in file_path or file_path.startswith('/'):
            return {"error": "invalid file path"}, 400
        
        full_path = os.path.join(base_dir, file_path)
        
        # check if file exists
        if not os.path.exists(full_path):
            return {"error": "file or folder does not exist"}, 404
        
        # ensure path is within base directory
        real_base = os.path.realpath(base_dir)
        real_target = os.path.realpath(full_path)
        if not real_target.startswith(real_base):
            return {"error": "no permission to delete this path"}, 403
        
        # delete file or folder
        import shutil
        if os.path.isdir(full_path):
            shutil.rmtree(full_path)
            return {"message": f"successfully deleted folder: {file_path}"}
        else:
            os.remove(full_path)
            return {"message": f"successfully deleted file: {file_path}"}
            
    except Exception as e:
        return {"error": f"error deleting file: {str(e)}"}, 500

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        # read HTML file and return
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            html_path = os.path.join(current_dir, 'upload.html')
            with open(html_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return {"error": "upload page file not found"}, 404
        except Exception as e:
            return {"error": f"error reading upload page file: {str(e)}"}, 500
    
    # if POST request, handle file upload
    upload_type = request.form.get('upload_type', 'file')
    
    if upload_type == 'folder':
        return handle_folder_upload()
    else:
        return handle_single_file_upload()

def handle_single_file_upload():
    """handle single file upload"""
    file = request.files.get('file')
    if not file or file.filename == '':
        return {"error": "no file selected"}, 400
    
    # validate file name security
    fname = secure_filename(file.filename)
    
    # # check if extension is allowed
    # ext = fname.rsplit('.', 1)[1].lower() if '.' in fname else ''
    # if ext not in {'json', 'pkl', 'csv', 'txt'}:
    #     return {"error": f"unsupported file type: {ext}"}, 400
    
    # determine save path
    save_dir = os.path.expanduser('~/.NssMPClib')
    data_dir = os.path.join(save_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    if fname == 'config.json':
        dest_path = os.path.join(save_dir, 'config.json')
    else:
        dest_path = os.path.join(data_dir, fname)
    
    file.save(dest_path)
    return {"message": f"file uploaded successfully: {fname}"}, 200

def handle_folder_upload():
    """handle folder upload"""
    files = request.files.getlist('files')
    # Some browsers may upload an empty file (with filename as '') even if no folder is selected, which needs special handling
    if not files or all(f.filename == '' for f in files):
        return {"error": "no folder selected or folder is empty"}, 400
    
    uploaded_files = []
    errors = []
    
    # base save directory
    save_dir = os.path.expanduser('~/.NssMPClib')
    data_dir = os.path.join(save_dir, 'data')
    
    for file in files:
        if file.filename == '':
            continue
            
        try:
            # get relative path (browser will provide the full relative path)
            relative_path = file.filename
            
            # clean each part of the path
            path_parts = relative_path.split('/')
            clean_parts = []
            for part in path_parts:
                clean_part = secure_filename(part)
                if clean_part:  # only add non-empty parts
                    clean_parts.append(clean_part)
            
            if not clean_parts:
                continue
                
            # clean_filename = clean_parts[-1]
            
            # # check file extension
            # ext = clean_filename.rsplit('.', 1)[1].lower() if '.' in clean_filename else ''
            # if ext not in {'json', 'pkl', 'csv', 'txt'}:
            #     errors.append(f"skip unsupported file type: {relative_path} ({ext})")
            #     continue
            
            dest_path = os.path.join(data_dir, *clean_parts)
            
            # create directory
            dest_dir = os.path.dirname(dest_path)
            os.makedirs(dest_dir, exist_ok=True)
            
            # save file
            file.save(dest_path)
            uploaded_files.append(relative_path)
            
        except Exception as e:
            errors.append(f"error uploading file {file.filename}: {str(e)}")
    
    if uploaded_files:
        message = f"successfully uploaded {len(uploaded_files)} files"
        if errors:
            message += f", {len(errors)} files have problems"
        
        response = {
            "message": message,
            "uploaded_files": uploaded_files
        }
        if errors:
            response["errors"] = errors
        
        return response, 200
    else:
        return {"error": "no files uploaded successfully", "errors": errors}, 400

def secure_filename(filename):
    """
    clean file name, prevent directory traversal and special character attacks
    
    Args:
        filename (str): original file name
        
    Returns:
        str: cleaned and secure file name
    """
    if not filename:
        return ""
    
    # convert to string and normalize Unicode characters
    filename = str(filename)
    filename = unicodedata.normalize('NFKC', filename)
    
    # remove or replace dangerous characters
    # remove path separators and directory traversal characters
    dangerous_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '..']
    for char in dangerous_chars:
        filename = filename.replace(char, '_')
    
    # remove control characters
    filename = ''.join(char for char in filename if ord(char) >= 32)
    
    # remove leading and trailing whitespace
    filename = filename.strip()
    
    # remove leading and trailing dots (Windows file name limit)
    filename = filename.strip('.')
    
    # if file name is empty or only contains whitespace, return default name
    if not filename or filename.isspace():
        return "uploaded_file"
    
    # limit file name length (including extension)
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        max_name_length = 255 - len(ext)
        filename = name[:max_name_length] + ext
    
    # ensure file name does not start with a number (to avoid system restrictions)
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
        # get full stack trace
        exc_type, exc_value, exc_traceback = sys.exc_info()
        stack_trace = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    finally:
        sys.stdout = original_stdout
        output_queue.put((buffer.getvalue(), error, stack_trace))

if __name__ == '__main__':
    # import argparse
    # parser = argparse.ArgumentParser(description="interpreter service")
    # parser.add_argument("--port", type=int, help="port", required=True)
    # args = parser.parse_args()

    # app.run(host='0.0.0.0', port=args.port)
    app.run(host='0.0.0.0', port=5000)